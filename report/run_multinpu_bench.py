#!/usr/bin/env python3
"""
HTTP /file_parse throughput probe for RapidDocCpp (fixed worker count; measure concurrency).
Writes JSON summary to ./report/ — run from project root with server already up and env set.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Scenario:
    name: str
    pdf: Path
    table_enable: bool
    extra_fields: Tuple[str, ...]


def curl_parse(
    base_url: str,
    scenario: Scenario,
    timeout_sec: int = 900,
) -> Tuple[int, Optional[Dict[str, Any]], str]:
    url = base_url.rstrip("/") + "/file_parse"
    args = [
        "curl",
        "-sS",
        "-m",
        str(timeout_sec),
        "-w",
        "\n%{http_code}",
        "-X",
        "POST",
        url,
        "-F",
        f"files=@{scenario.pdf};type=application/pdf",
        "-F",
        "return_md=true",
        "-F",
        "return_content_list=true",
        "-F",
        "return_middle_json=true",
        "-F",
        "clear_output_file=true",
        "-F",
        f"table_enable={'true' if scenario.table_enable else 'false'}",
    ]
    for f in scenario.extra_fields:
        args.extend(["-F", f])
    proc = subprocess.run(args, capture_output=True, text=True)
    raw = proc.stdout
    if "\n" not in raw:
        return 0, None, raw + proc.stderr
    body, code = raw.rsplit("\n", 1)
    try:
        http_code = int(code.strip())
    except ValueError:
        http_code = 0
    if http_code != 200:
        return http_code, None, body[:4000]
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return http_code, None, body[:4000]
    return http_code, payload, body


def extract_stats(payload: Dict[str, Any]) -> Dict[str, Any]:
    results = payload.get("results")
    if not results:
        raise ValueError("no results in payload")
    r0 = results[0]
    stats = r0.get("stats") or {}
    out = {
        "pdf_render_ms": stats.get("pdf_render_ms"),
        "layout_ms": stats.get("layout_ms"),
        "ocr_ms": stats.get("ocr_ms"),
        "table_ms": stats.get("table_ms"),
        "reading_order_ms": stats.get("reading_order_ms"),
        "output_gen_ms": stats.get("output_gen_ms"),
        "pipeline_call_ms": stats.get("pipeline_call_ms"),
        "npu_serial_ms": stats.get("npu_serial_ms"),
        "cpu_only_ms": stats.get("cpu_only_ms"),
        "npu_lock_wait_ms": stats.get("npu_lock_wait_ms"),
        "npu_lock_hold_ms": stats.get("npu_lock_hold_ms"),
        "time_ms": stats.get("time_ms"),
        "output_dir": r0.get("output_dir"),
    }
    return out


def stable_middle_json_hash(middle: Any) -> str:
    s = json.dumps(middle, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def md_hash(md: str) -> str:
    return hashlib.sha256(md.encode("utf-8")).hexdigest()


def fetch_status(base_url: str) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/status"
    proc = subprocess.run(
        ["curl", "-sS", "-m", "30", url],
        capture_output=True,
        text=True,
    )
    proc.check_returncode()
    return json.loads(proc.stdout)


def run_benchmark(
    base_url: str,
    scenario: Scenario,
    concurrency: int,
    total_requests: int,
    label: str,
) -> Dict[str, Any]:
    # Warmup (not timed)
    code, payload, err = curl_parse(base_url, scenario)
    if code != 200 or not payload:
        return {
            "label": label,
            "scenario": scenario.name,
            "concurrency": concurrency,
            "error": f"warmup_failed code={code} err={err[:2000]}",
        }

    # Integrity sample from warmup
    r0 = payload["results"][0]
    wh = md_hash(r0.get("md_content") or "")
    mj = stable_middle_json_hash(r0.get("middle_json"))

    errors: List[str] = []
    stats_list: List[Dict[str, Any]] = []

    def one(_i: int) -> Tuple[int, Optional[Dict[str, Any]], str]:
        return curl_parse(base_url, scenario)

    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(one, i) for i in range(total_requests)]
        for fut in as_completed(futures):
            code, payload, err = fut.result()
            if code != 200 or not payload:
                errors.append(f"http={code} {err[:500]}")
                continue
            try:
                stats_list.append(extract_stats(payload))
            except Exception as e:
                errors.append(str(e))

    elapsed = time.perf_counter() - t0
    ok = len(stats_list)
    qps = ok / elapsed if elapsed > 0 else 0.0

    # Mean stats
    def mean(key: str) -> Optional[float]:
        vals = [s[key] for s in stats_list if s.get(key) is not None]
        if not vals:
            return None
        return sum(vals) / len(vals)

    status_after = fetch_status(base_url)

    return {
        "label": label,
        "scenario": scenario.name,
        "pdf": str(scenario.pdf),
        "table_enable": scenario.table_enable,
        "concurrency": concurrency,
        "total_requests": total_requests,
        "completed_ok": ok,
        "errors": errors[:20],
        "error_count": len(errors),
        "wall_seconds": round(elapsed, 4),
        "qps": round(qps, 4),
        "mean_pdf_render_ms": mean("pdf_render_ms"),
        "mean_layout_ms": mean("layout_ms"),
        "mean_ocr_ms": mean("ocr_ms"),
        "mean_table_ms": mean("table_ms"),
        "mean_reading_order_ms": mean("reading_order_ms"),
        "mean_output_gen_ms": mean("output_gen_ms"),
        "mean_pipeline_call_ms": mean("pipeline_call_ms"),
        "mean_npu_serial_ms": mean("npu_serial_ms"),
        "mean_cpu_only_ms": mean("cpu_only_ms"),
        "mean_npu_lock_wait_ms": mean("npu_lock_wait_ms"),
        "mean_npu_lock_hold_ms": mean("npu_lock_hold_ms"),
        "warmup_md_sha256": wh,
        "warmup_middle_json_sha256": mj,
        "status_snapshot": status_after,
    }


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:18080"
    out_path = (
        Path(sys.argv[2]).resolve()
        if len(sys.argv) > 2
        else root / "report" / "multinpu_bench_result.json"
    )

    scenarios = [
        Scenario(
            "layout_ocr",
            root / "test_files" / "small_ocr_origin.pdf",
            False,
            (),
        ),
        Scenario(
            "layout_ocr_table",
            root / "test_files" / "比亚迪财报_origin.pdf",
            True,
            ("start_page_id=0", "end_page_id=1"),
        ),
        Scenario(
            "table_heavy",
            root / "test_files" / "表格0.pdf",
            True,
            (),
        ),
    ]

    for s in scenarios:
        if not s.pdf.exists():
            print(f"Missing PDF: {s.pdf}", file=sys.stderr)
            return 1

    report: Dict[str, Any] = {
        "base_url": base_url,
        "result_path": str(out_path),
        "note": "Single-machine run; multi-card comparison requires repeating with same script under different DXRT visible devices.",
        "server_config": "See status_snapshot.pipeline_lock / pipeline_stage_totals per row.",
        "status_before": fetch_status(base_url),
        "runs": [],
    }

    for sc in scenarios:
        for c in (2, 3, 6):
            label = f"{sc.name}_c{c}"
            print(f"Running {label} ...", flush=True)
            report["runs"].append(
                run_benchmark(
                    base_url,
                    sc,
                    concurrency=c,
                    total_requests=12,
                    label=label,
                )
            )
            time.sleep(0.5)

    report["status_after"] = fetch_status(base_url)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
