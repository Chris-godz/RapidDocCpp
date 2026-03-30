#!/usr/bin/env python3
"""
Run formal topology benchmark in small resumable chunks.

Each chunk covers one (workload, concurrency, device_count) slice and lets the
existing probe own the actual A/B execution and gating. This keeps reruns small
when a single slice crashes mid-run.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Tuple


def load_probe_module(project_root: Path):
    probe_path = project_root / "tools" / "npu_topology_probe.py"
    spec = importlib.util.spec_from_file_location("npu_topology_probe", probe_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load probe module from {probe_path}")
    module = importlib.util.module_from_spec(spec)
    # Python 3.13 dataclass evaluation expects the loading module to already be
    # registered in sys.modules while decorators run.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def read_json(path: Path) -> Optional[Dict[str, object]]:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def result_key(row: Dict[str, object]) -> Tuple[str, str, int, int]:
    return (
        str(row.get("topology", "")),
        str(row.get("workload", "")),
        int(row.get("concurrency", 0)),
        int(row.get("device_count", 0)),
    )


def chunk_id(workload: str, concurrency: int, device_count: int) -> str:
    return f"{workload}__c{concurrency}__{device_count}card"


def build_chunk_plan(
    workloads: Iterable[str],
    concurrency_list: Iterable[int],
    device_counts: Iterable[int],
) -> List[Dict[str, object]]:
    plan: List[Dict[str, object]] = []
    for workload in workloads:
        for concurrency in concurrency_list:
            for device_count in device_counts:
                plan.append({
                    "id": chunk_id(workload, concurrency, device_count),
                    "workload": workload,
                    "concurrency": concurrency,
                    "device_count": device_count,
                })
    return plan


def chunk_report_path(out_dir: Path, chunk: Dict[str, object]) -> Path:
    return out_dir / "chunks" / str(chunk["id"]) / "topology_report.json"


def chunk_log_path(out_dir: Path, chunk: Dict[str, object]) -> Path:
    return out_dir / "chunks" / str(chunk["id"]) / "runner.log"


def chunk_manifest_path(out_dir: Path, chunk: Dict[str, object]) -> Path:
    return out_dir / "chunks" / str(chunk["id"]) / "chunk_manifest.json"


def chunk_out_dir(out_dir: Path, chunk: Dict[str, object]) -> Path:
    return out_dir / "chunks" / str(chunk["id"])


def is_chunk_complete(report: Optional[Dict[str, object]]) -> bool:
    if not report:
        return False
    return str(report.get("status", "")) == "ok"


def collect_chunk_reports(
    out_dir: Path,
    plan: List[Dict[str, object]],
) -> List[Tuple[Dict[str, object], Dict[str, object]]]:
    reports: List[Tuple[Dict[str, object], Dict[str, object]]] = []
    for chunk in plan:
        report = read_json(chunk_report_path(out_dir, chunk))
        if report is not None:
            reports.append((chunk, report))
    return reports


def clone_jsonable(value: object) -> object:
    return json.loads(json.dumps(value, ensure_ascii=False))


def aggregate_results(
    report_pairs: List[Tuple[Dict[str, object], Dict[str, object]]],
) -> List[Dict[str, object]]:
    merged: Dict[Tuple[str, str, int, int], Dict[str, object]] = {}
    for _, report in report_pairs:
        for row in report.get("raw_results", []):
            if isinstance(row, dict):
                merged[result_key(row)] = row
    return list(merged.values())


def build_campaign_report(
    probe,
    args: argparse.Namespace,
    out_dir: Path,
    plan: List[Dict[str, object]],
    chunk_reports: List[Tuple[Dict[str, object], Dict[str, object]]],
    failed_chunks: List[Dict[str, object]],
) -> Dict[str, object]:
    raw_results = aggregate_results(chunk_reports)
    baseline = (
        chunk_reports[0][1].get("baseline", {})
        if chunk_reports
        else probe.verify_baseline(args.project_root, args)
    )
    env_snapshot = (
        chunk_reports[0][1].get("env_snapshot", {})
        if chunk_reports
        else probe.capture_env_snapshot()
    )
    env_mismatches = (
        chunk_reports[0][1].get("env_mismatches", [])
        if chunk_reports
        else probe.expected_env_mismatches(env_snapshot)
    )
    available_device_ids = (
        chunk_reports[0][1].get("available_device_ids", [])
        if chunk_reports
        else probe.detect_dxrt_devices()
    )
    dxrt_snapshot = (
        chunk_reports[0][1].get("dxrt_status", {})
        if chunk_reports
        else probe.dxrt_status_snapshot()
    )

    report_args = SimpleNamespace(
        workloads=args.workloads,
        concurrency_list=args.concurrency_list,
        device_counts=args.device_counts,
    )

    completed_chunk_ids = [str(chunk["id"]) for chunk, _ in chunk_reports if is_chunk_complete(_)]
    all_chunk_ids = [str(chunk["id"]) for chunk in plan]
    pending_chunk_ids = [chunk_id for chunk_id in all_chunk_ids if chunk_id not in completed_chunk_ids]

    if failed_chunks:
        status = "partial_chunk_failed"
    elif len(completed_chunk_ids) < len(all_chunk_ids):
        status = "partial"
    else:
        status = "ok"

    report: Dict[str, object] = {
        "generated_at": int(time.time()),
        "status": status,
        "baseline": baseline,
        "env_snapshot": env_snapshot,
        "env_mismatches": env_mismatches,
        "available_device_ids": available_device_ids,
        "available_device_count": len(available_device_ids),
        "dxrt_status": dxrt_snapshot,
        "preflight": {},
        "raw_results": raw_results,
        "section1": probe.build_section1(report_args, raw_results),
        "section2": probe.build_section2(raw_results),
        "section3": probe.build_section3(raw_results),
        "section4": probe.build_section4(raw_results),
        "section5": probe.build_section5(probe.build_section3(raw_results)),
        "chunk_mode": {
            "granularity": "workload_concurrency_device_count",
            "planned_chunks": all_chunk_ids,
            "completed_chunks": completed_chunk_ids,
            "pending_chunks": pending_chunk_ids,
            "failed_chunks": failed_chunks,
        },
    }
    probe.write_report(out_dir, report)
    write_formal_ab_summary(out_dir, report)
    return report


def request_pipeline_call_ms(request: Dict[str, object]) -> float:
    result = request.get("result", {})
    stats = result.get("stats", {}) if isinstance(result, dict) else {}
    if not stats:
        stats = request.get("stats", {})
    return float(stats.get("pipeline_call_ms", 0.0)) if isinstance(stats, dict) else 0.0


def active_devices_from_status_snapshot(status_snapshot: Dict[str, object]) -> int:
    per_device = status_snapshot.get("per_device", [])
    active = 0
    for item in per_device:
        if not isinstance(item, dict):
            continue
        busy_ms = float(item.get("busy_time_ms", 0.0))
        request_count = int(item.get("request_count", 0))
        if busy_ms > 0.0 or request_count > 0:
            active += 1
    return active


def overlap_summary_for_row(row: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if row is None:
        return None
    waves = row.get("waves", [])
    if not isinstance(waves, list):
        waves = []
    closer_to_sum = 0
    closer_to_max = 0
    wave_details = []
    for wave_index, wave in enumerate(waves):
        if not isinstance(wave, dict):
            continue
        requests = wave.get("requests", [])
        request_ms = [
            request_pipeline_call_ms(request)
            for request in requests
            if isinstance(request, dict)
        ]
        elapsed_ms = float(wave.get("elapsed_ms", 0.0))
        sum_request_ms = sum(request_ms)
        max_request_ms = max(request_ms) if request_ms else 0.0
        distance_to_sum = abs(elapsed_ms - sum_request_ms)
        distance_to_max = abs(elapsed_ms - max_request_ms)
        closer_to = "sum" if distance_to_sum <= distance_to_max else "max"
        if closer_to == "sum":
            closer_to_sum += 1
        else:
            closer_to_max += 1
        wave_details.append({
            "wave_index": wave_index,
            "elapsed_ms": elapsed_ms,
            "sum_request_pipeline_ms": sum_request_ms,
            "max_request_pipeline_ms": max_request_ms,
            "distance_to_sum_ms": distance_to_sum,
            "distance_to_max_ms": distance_to_max,
            "closer_to": closer_to,
        })
    status_snapshot = row.get("status_snapshot", {})
    if not isinstance(status_snapshot, dict):
        status_snapshot = {}
    return {
        "topology": row.get("topology"),
        "status": row.get("status"),
        "wave_count": len(wave_details),
        "waves_closer_to_sum": closer_to_sum,
        "waves_closer_to_max": closer_to_max,
        "active_devices": active_devices_from_status_snapshot(status_snapshot),
        "wave_details": wave_details,
    }


def overlap_check_for_report(
    report: Optional[Dict[str, object]],
    chunk: Dict[str, object],
) -> Dict[str, object]:
    if report is None:
        return {"A": None, "B": None}
    topology_a = find_result_row(
        report,
        "A",
        str(chunk["workload"]),
        int(chunk["concurrency"]),
        int(chunk["device_count"]),
    )
    topology_b = find_result_row(
        report,
        "B",
        str(chunk["workload"]),
        int(chunk["concurrency"]),
        int(chunk["device_count"]),
    )
    return {
        "A": overlap_summary_for_row(topology_a),
        "B": overlap_summary_for_row(topology_b),
    }


def write_formal_ab_summary(out_dir: Path, report: Dict[str, object]) -> None:
    topology_report_md = out_dir / "topology_report.md"
    if not topology_report_md.exists():
        return
    baseline = report.get("baseline", {})
    if not isinstance(baseline, dict):
        baseline = {}
    lines = [
        "# Formal A/B Summary",
        "",
        "- 旧 chunk 结果无效，不再引用",
        "- 当前 benchmark 基于 ingress 修复后的新 baseline",
        "- 当前实验已恢复为有效多卡实验",
        "- 只有新目录下结果可作为正式结论",
        f"- main_commit: {baseline.get('main_commit')}",
        f"- ocr_checkout_commit: {baseline.get('ocr_checkout_commit')}",
        f"- report_status: {report.get('status')}",
        "",
        topology_report_md.read_text(encoding="utf-8").rstrip(),
        "",
    ]
    write_text(out_dir / "formal_ab_summary.md", "\n".join(lines))


def resolved_binary_path(project_root: Path, raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = project_root / path
    return path.resolve()


def make_probe_cmd(
    project_root: Path,
    script_path: Path,
    chunk: Dict[str, object],
    args: argparse.Namespace,
) -> List[str]:
    cmd = [
        sys.executable,
        str(script_path),
        "--project-root", str(project_root),
        "--server-binary", args.server_binary,
        "--lb-binary", args.lb_binary,
        "--host", args.host,
        "--base-port", str(args.base_port),
        "--device-counts", str(chunk["device_count"]),
        "--concurrency-list", str(chunk["concurrency"]),
        "--workloads", str(chunk["workload"]),
        "--warmup-waves", str(args.warmup_waves),
        "--measure-waves", str(args.measure_waves),
        "--out-dir", str(chunk_out_dir(args.out_dir, chunk)),
        "--expected-main-commit", args.expected_main_commit,
        "--expected-ocr-commit", args.expected_ocr_commit,
    ]
    if args.skip_baseline_check:
        cmd.append("--skip-baseline-check")
    if args.allow_env_mismatch:
        cmd.append("--allow-env-mismatch")
    return cmd


def find_result_row(
    report: Dict[str, object],
    topology: str,
    workload: str,
    concurrency: int,
    device_count: int,
) -> Optional[Dict[str, object]]:
    for row in report.get("raw_results", []):
        if not isinstance(row, dict):
            continue
        if (
            row.get("topology") == topology
            and row.get("workload") == workload
            and int(row.get("concurrency", 0)) == concurrency
            and int(row.get("device_count", 0)) == device_count
        ):
            return clone_jsonable(row)  # type: ignore[return-value]
    return None


def build_probe_runtime_args(
    probe,
    args: argparse.Namespace,
    out_path: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        project_root=args.project_root,
        server_binary=resolved_binary_path(args.project_root, args.server_binary),
        lb_binary=resolved_binary_path(args.project_root, args.lb_binary),
        host=args.host,
        base_port=args.base_port,
        device_counts=args.device_counts,
        concurrency_list=args.concurrency_list,
        workloads=args.workloads,
        warmup_waves=args.warmup_waves,
        measure_waves=args.measure_waves,
        request_output_root=out_path / "request_outputs",
        expected_main_commit=args.expected_main_commit,
        expected_ocr_commit=args.expected_ocr_commit,
    )


def build_chunk_report_payload(
    probe,
    args: argparse.Namespace,
    chunk: Dict[str, object],
    baseline: Dict[str, object],
    env_snapshot: Dict[str, object],
    env_mismatches: List[str],
    available_device_ids: List[int],
    dxrt_snapshot: Dict[str, object],
    raw_results: List[Dict[str, object]],
    status: str,
    preflight: Dict[str, object],
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    report_args = SimpleNamespace(
        workloads=[chunk["workload"]],
        concurrency_list=[chunk["concurrency"]],
        device_counts=[chunk["device_count"]],
    )
    section3 = probe.build_section3(raw_results)
    report: Dict[str, object] = {
        "generated_at": int(time.time()),
        "status": status,
        "baseline": baseline,
        "env_snapshot": env_snapshot,
        "env_mismatches": env_mismatches,
        "available_device_ids": available_device_ids,
        "available_device_count": len(available_device_ids),
        "dxrt_status": dxrt_snapshot,
        "preflight": preflight,
        "raw_results": raw_results,
        "section1": probe.build_section1(report_args, raw_results),
        "section2": probe.build_section2(raw_results),
        "section3": section3,
        "section4": probe.build_section4(raw_results),
        "section5": probe.build_section5(section3),
        "chunk_mode": {
            "chunk_id": chunk["id"],
            "granularity": "workload_concurrency_device_count",
        },
    }
    if extra:
        report.update(extra)
    return report


def run_exact_chunk(
    probe,
    chunk: Dict[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    out_path = chunk_out_dir(args.out_dir, chunk)
    out_path.mkdir(parents=True, exist_ok=True)
    log_path = chunk_log_path(args.out_dir, chunk)
    manifest_path = chunk_manifest_path(args.out_dir, chunk)

    started_at = time.time()
    exit_code = 0
    report: Optional[Dict[str, object]] = None
    exception_text: Optional[str] = None

    try:
        with log_path.open("w", encoding="utf-8") as log_file:
            with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
                baseline = probe.verify_baseline(args.project_root, args)
                if args.skip_baseline_check:
                    baseline["status"] = "skipped"
                    baseline["failures"] = []
                env_snapshot = probe.capture_env_snapshot()
                env_mismatches = probe.expected_env_mismatches(env_snapshot)
                available_device_ids = probe.detect_dxrt_devices()
                dxrt_snapshot = probe.dxrt_status_snapshot()

                raw_results: List[Dict[str, object]] = []
                preflight: Dict[str, object] = {
                    "mode": "reuse_1card_baseline",
                }

                blocked_status: Optional[str] = None
                blocked_attribution: Optional[str] = None

                if baseline.get("status") not in ("ok", "skipped"):
                    blocked_status = str(baseline["status"])
                    blocked_attribution = "正式实验基线未恢复到已验收 checkpoint，停止开跑。"
                elif env_mismatches and not args.allow_env_mismatch:
                    blocked_status = "blocked_env_profile_mismatch"
                    blocked_attribution = "运行环境未对齐 set_env.sh 1 2 1 3 2 4，停止开跑。"
                else:
                    missing_binaries = [
                        str(path)
                        for path in (
                            resolved_binary_path(args.project_root, args.server_binary),
                            resolved_binary_path(args.project_root, args.lb_binary),
                        )
                        if not path.exists()
                    ]
                    if missing_binaries:
                        blocked_status = "blocked_binary_unavailable"
                        blocked_attribution = "正式实验所需 server/LB 二进制不存在，需先完成 build 与 ctest。"
                    elif len(available_device_ids) < chunk["device_count"]:
                        blocked_status = "blocked_device_unavailable"
                        blocked_attribution = "目标主机 DXRT device 数不足该 chunk 所需卡数，停止开跑。"

                source_report = read_json(
                    chunk_report_path(
                        args.out_dir,
                        {
                            "id": chunk_id(chunk["workload"], chunk["concurrency"], 1),
                            "workload": chunk["workload"],
                            "concurrency": chunk["concurrency"],
                            "device_count": 1,
                        },
                    )
                )
                a1 = None
                b1 = None
                if source_report is not None:
                    a1 = find_result_row(
                        source_report,
                        "A",
                        str(chunk["workload"]),
                        int(chunk["concurrency"]),
                        1,
                    )
                    b1 = find_result_row(
                        source_report,
                        "B",
                        str(chunk["workload"]),
                        int(chunk["concurrency"]),
                        1,
                    )
                    if a1 is not None:
                        raw_results.append(a1)
                    if b1 is not None:
                        raw_results.append(b1)
                    preflight["source_chunk"] = chunk_id(chunk["workload"], chunk["concurrency"], 1)

                if blocked_status is None and (a1 is None or a1.get("status") != "ok" or b1 is None):
                    blocked_status = "blocked_canonical_baseline_unavailable"
                    blocked_attribution = "同 workload/并发 的 1 卡 baseline chunk 不可用，不能直接跑 >1 卡 chunk。"

                if blocked_status is not None:
                    report = build_chunk_report_payload(
                        probe,
                        args,
                        chunk,
                        baseline,
                        env_snapshot,
                        env_mismatches,
                        available_device_ids,
                        dxrt_snapshot,
                        raw_results,
                        blocked_status,
                        preflight,
                        extra={
                            "section5": {
                                "verdict": blocked_status,
                                "best_topology": "blocked",
                                "attribution": blocked_attribution,
                            },
                        },
                    )
                else:
                    runtime_args = build_probe_runtime_args(probe, args, out_path)
                    manifest = probe.load_manifest(
                        args.project_root / probe.WORKLOAD_MANIFESTS[str(chunk["workload"])]
                    )
                    baseline_hashes = probe.make_baseline_hashes_from_result(
                        a1,  # type: ignore[arg-type]
                        manifest,
                        int(chunk["concurrency"]),
                        Path(str(a1.get("group_output_dir", out_path / "request_outputs"))),
                    )
                    device_ids = available_device_ids[: int(chunk["device_count"])]
                    a_target = probe.run_profile(
                        "A",
                        device_ids,
                        str(chunk["workload"]),
                        int(chunk["concurrency"]),
                        runtime_args,
                        baseline_hashes,
                    )
                    b_target = probe.run_profile(
                        "B",
                        device_ids,
                        str(chunk["workload"]),
                        int(chunk["concurrency"]),
                        runtime_args,
                        baseline_hashes,
                    )
                    raw_results.extend([a_target, b_target])
                    report = build_chunk_report_payload(
                        probe,
                        args,
                        chunk,
                        baseline,
                        env_snapshot,
                        env_mismatches,
                        available_device_ids,
                        dxrt_snapshot,
                        raw_results,
                        "ok",
                        preflight,
                    )

                probe.write_report(out_path, report)

    except Exception:
        exit_code = 1
        exception_text = traceback.format_exc()
        fallback_report = read_json(chunk_report_path(args.out_dir, chunk)) or {}
        baseline = fallback_report.get("baseline", {})
        env_snapshot = fallback_report.get("env_snapshot", {})
        env_mismatches = fallback_report.get("env_mismatches", [])
        available_device_ids = fallback_report.get("available_device_ids", [])
        dxrt_snapshot = fallback_report.get("dxrt_status", {})
        raw_results = fallback_report.get("raw_results", [])
        report = build_chunk_report_payload(
            probe,
            args,
            chunk,
            baseline if isinstance(baseline, dict) else {},
            env_snapshot if isinstance(env_snapshot, dict) else {},
            env_mismatches if isinstance(env_mismatches, list) else [],
            available_device_ids if isinstance(available_device_ids, list) else [],
            dxrt_snapshot if isinstance(dxrt_snapshot, dict) else {},
            raw_results if isinstance(raw_results, list) else [],
            "failed_chunk_exception",
            {"mode": "reuse_1card_baseline"},
            extra={
                "section5": {
                    "verdict": "failed_chunk_exception",
                    "best_topology": "blocked",
                    "attribution": "chunk 运行过程中抛出异常，详见 chunk runner.log。",
                },
                "chunk_exception": exception_text,
            },
        )
        probe.write_report(out_path, report)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write("\n=== chunk exception ===\n")
            log_file.write(exception_text)

    finished_at = time.time()
    report = read_json(chunk_report_path(args.out_dir, chunk))
    manifest = {
        "chunk": chunk,
        "mode": "exact_reuse_1card_baseline",
        "exit_code": exit_code,
        "started_at": int(started_at),
        "finished_at": int(finished_at),
        "elapsed_seconds": finished_at - started_at,
        "log_path": str(log_path),
        "report_path": str(chunk_report_path(args.out_dir, chunk)),
        "report_exists": report is not None,
        "report_status": report.get("status") if report else None,
        "overlap_check": overlap_check_for_report(report, chunk),
    }
    write_json(manifest_path, manifest)
    return manifest


def run_chunk(
    project_root: Path,
    probe,
    probe_script: Path,
    chunk: Dict[str, object],
    args: argparse.Namespace,
) -> Dict[str, object]:
    if int(chunk["device_count"]) > 1:
        return run_exact_chunk(probe, chunk, args)
    out_path = chunk_out_dir(args.out_dir, chunk)
    out_path.mkdir(parents=True, exist_ok=True)
    log_path = chunk_log_path(args.out_dir, chunk)
    manifest_path = chunk_manifest_path(args.out_dir, chunk)
    cmd = make_probe_cmd(project_root, probe_script, chunk, args)
    started_at = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(
            cmd,
            cwd=str(project_root),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
    finished_at = time.time()
    report = read_json(chunk_report_path(args.out_dir, chunk))
    manifest = {
        "chunk": chunk,
        "command": cmd,
        "exit_code": proc.returncode,
        "started_at": int(started_at),
        "finished_at": int(finished_at),
        "elapsed_seconds": finished_at - started_at,
        "log_path": str(log_path),
        "report_path": str(chunk_report_path(args.out_dir, chunk)),
        "report_exists": report is not None,
        "report_status": report.get("status") if report else None,
        "overlap_check": overlap_check_for_report(report, chunk),
    }
    write_json(manifest_path, manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--server-binary", default="build_Release/bin/rapid_doc_npu_test_server")
    parser.add_argument("--lb-binary", default="build_Release/bin/rapid_doc_topology_lb")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=19880)
    parser.add_argument("--device-counts", default="1,2,3")
    parser.add_argument("--concurrency-list", default="2,3,6")
    parser.add_argument("--workloads", default="")
    parser.add_argument("--warmup-waves", type=int, default=1)
    parser.add_argument("--measure-waves", type=int, default=5)
    parser.add_argument("--out-dir", default="output-benchmark/topology_attribution_chunked")
    parser.add_argument("--expected-main-commit", default="")
    parser.add_argument("--expected-ocr-commit", default="")
    parser.add_argument("--skip-baseline-check", action="store_true")
    parser.add_argument("--allow-env-mismatch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-chunks", type=int, default=0)
    parser.add_argument("--continue-on-chunk-failure", action="store_true")
    args = parser.parse_args()

    args.project_root = Path(args.project_root).resolve()
    args.out_dir = (args.project_root / args.out_dir).resolve()
    probe = load_probe_module(args.project_root)
    probe_script = args.project_root / "tools" / "npu_topology_probe.py"

    if not args.workloads:
        args.workloads = ",".join(probe.FORMAL_WORKLOADS)
    if not args.expected_ocr_commit:
        args.expected_ocr_commit = probe.EXPECTED_OCR_COMMIT
    if not args.expected_main_commit:
        head_code, head_output = probe.git_text(args.project_root, "rev-parse HEAD")
        if head_code == 0:
            args.expected_main_commit = head_output.strip()

    args.device_counts = probe.parse_csv_ints(args.device_counts)
    args.concurrency_list = probe.parse_csv_ints(args.concurrency_list)
    args.workloads = probe.parse_csv_strings(args.workloads)

    plan = build_chunk_plan(args.workloads, args.concurrency_list, args.device_counts)
    chunk_state_path = args.out_dir / "chunk_campaign_state.json"
    failed_chunks: List[Dict[str, object]] = []
    executed = 0

    for chunk in plan:
        report = read_json(chunk_report_path(args.out_dir, chunk))
        if args.resume and is_chunk_complete(report):
            continue
        if args.max_chunks > 0 and executed >= args.max_chunks:
            break

        manifest = run_chunk(args.project_root, probe, probe_script, chunk, args)
        executed += 1
        report = read_json(chunk_report_path(args.out_dir, chunk))
        if manifest["exit_code"] != 0 or not is_chunk_complete(report):
            failed_chunks.append({
                "chunk_id": chunk["id"],
                "exit_code": manifest["exit_code"],
                "report_status": manifest["report_status"],
                "log_path": manifest["log_path"],
            })

        chunk_reports = collect_chunk_reports(args.out_dir, plan)
        campaign_report = build_campaign_report(
            probe,
            args,
            args.out_dir,
            plan,
            chunk_reports,
            failed_chunks,
        )
        write_json(chunk_state_path, {
            "status": campaign_report["status"],
            "executed_chunks_this_run": executed,
            "planned_chunk_count": len(plan),
            "completed_chunk_count": len(campaign_report["chunk_mode"]["completed_chunks"]),
            "pending_chunk_count": len(campaign_report["chunk_mode"]["pending_chunks"]),
            "failed_chunks": failed_chunks,
        })

        if failed_chunks and not args.continue_on_chunk_failure:
            print(json.dumps({
                "status": "stopped_on_chunk_failure",
                "failed_chunk": failed_chunks[-1],
                "report": str(args.out_dir / "topology_report.json"),
            }, ensure_ascii=False))
            return 1

    chunk_reports = collect_chunk_reports(args.out_dir, plan)
    campaign_report = build_campaign_report(
        probe,
        args,
        args.out_dir,
        plan,
        chunk_reports,
        failed_chunks,
    )
    write_json(chunk_state_path, {
        "status": campaign_report["status"],
        "executed_chunks_this_run": executed,
        "planned_chunk_count": len(plan),
        "completed_chunk_count": len(campaign_report["chunk_mode"]["completed_chunks"]),
        "pending_chunk_count": len(campaign_report["chunk_mode"]["pending_chunks"]),
        "failed_chunks": failed_chunks,
    })
    print(json.dumps({
        "status": campaign_report["status"],
        "json": str(args.out_dir / "topology_report.json"),
        "markdown": str(args.out_dir / "topology_report.md"),
        "completed_chunks": len(campaign_report["chunk_mode"]["completed_chunks"]),
        "planned_chunks": len(plan),
        "failed_chunks": failed_chunks,
    }, ensure_ascii=False))
    return 0 if not failed_chunks else 1


if __name__ == "__main__":
    raise SystemExit(main())
