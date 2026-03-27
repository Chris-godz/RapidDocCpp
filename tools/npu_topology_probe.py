#!/usr/bin/env python3
"""
Run multi-device topology attribution experiments for RapidDocCpp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import signal
import statistics
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROFILE_MANIFESTS = {
    "mixed_baseline": "tools/topology_workloads/mixed_baseline.json",
    "layout_only": "tools/topology_workloads/layout_only.json",
    "ocr_only": "tools/topology_workloads/ocr_only.json",
    "table_only": "tools/topology_workloads/table_only.json",
}

PROFILE_FLAGS = {
    "mixed_baseline": [],
    "layout_only": ["--no-ocr", "--no-table"],
    "ocr_only": ["--no-table"],
    "table_only": ["--enable-table", "--no-ocr"],
}

THROUGHPUT_KEYS = [
    "pipeline_call_ms",
    "npu_serial_ms",
    "npu_lock_wait_ms",
    "cpu_only_ms",
    "route_queue_ms",
    "lb_proxy_ms",
]


@dataclass
class CurlResponse:
    exit_code: int
    http_code: int
    body: str
    raw_output: str


def run_shell(cmd: str) -> Tuple[int, str]:
    proc = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout


def shell_escape(value: str) -> str:
    return shlex.quote(value)


def run_curl_with_status(cmd: str) -> CurlResponse:
    exit_code, output = run_shell(cmd + " -w '\\n%{http_code}'")
    body = output
    http_code = 0
    if "\n" in output:
        body, code_text = output.rsplit("\n", 1)
        try:
            http_code = int(code_text.strip())
        except Exception:
            http_code = 0
    return CurlResponse(exit_code, http_code, body, output)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: object) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, ensure_ascii=False))


def summarize(values: List[float]) -> Dict[str, object]:
    if not values:
        return {
            "samples": 0,
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    ordered = sorted(values)
    def percentile(p: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        idx = (len(ordered) - 1) * p
        lo = math.floor(idx)
        hi = math.ceil(idx)
        if lo == hi:
            return ordered[lo]
        return ordered[lo] + (ordered[hi] - ordered[lo]) * (idx - lo)
    return {
        "samples": len(values),
        "mean": statistics.fmean(values),
        "p50": percentile(0.50),
        "p95": percentile(0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


def detect_dxrt_device_count() -> int:
    code, output = run_shell("dxrt-cli --status")
    if code != 0:
        return 0
    pattern = re.compile(r"^\s*\*\s*Device\s+\d+:")
    return sum(1 for line in output.splitlines() if pattern.search(line))


class ServerHandle:
    def __init__(self, argv: List[str], ready_url: str, log_path: Path) -> None:
        self.argv = argv
        self.ready_url = ready_url
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen[str]] = None

    def start(self) -> None:
        with self.log_path.open("w") as log:
            self.proc = subprocess.Popen(
                self.argv,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
            )

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=8)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=8)
        self.proc = None

    def wait_ready(self, timeout_sec: int = 120) -> None:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                log_tail = self.log_path.read_text(errors="ignore")[-2000:]
                raise RuntimeError(
                    f"server exited before ready, code={self.proc.returncode}, log_tail:\n{log_tail}"
                )
            res = run_curl_with_status(f"curl -sS --max-time 2 {shell_escape(self.ready_url)}")
            if res.exit_code == 0 and res.http_code == 200:
                return
            time.sleep(0.25)
        raise TimeoutError(f"timeout waiting for readiness: {self.ready_url}")

    def status(self) -> Dict[str, object]:
        res = run_curl_with_status(f"curl -sS --max-time 10 {shell_escape(self.ready_url)}")
        if res.exit_code != 0 or res.http_code != 200:
            raise RuntimeError(f"status request failed: {res.raw_output}")
        return json.loads(res.body)


def load_manifest(path: Path) -> List[Dict[str, object]]:
    return json.loads(path.read_text())


def make_request_batch(manifest: List[Dict[str, object]], concurrency: int) -> List[Dict[str, object]]:
    if not manifest:
        return []
    batch: List[Dict[str, object]] = []
    for i in range(concurrency):
        item = dict(manifest[i % len(manifest)])
        item["request_id"] = f"{item['name']}_{i}"
        batch.append(item)
    return batch


def build_file_parse_cmd(url: str, pdf_path: Path, fields: Dict[str, str]) -> str:
    parts = [
        "curl",
        "-sS",
        "--max-time 300",
        "-X POST",
        "-F",
        shell_escape(f"files=@{pdf_path};type=application/pdf"),
    ]
    for key, value in fields.items():
        parts.extend(["-F", shell_escape(f"{key}={value}")])
    parts.append(shell_escape(url))
    return " ".join(parts)


def parse_single_result(response: CurlResponse) -> Dict[str, object]:
    if response.exit_code != 0:
        raise RuntimeError(f"curl failed: {response.raw_output}")
    if response.http_code != 200:
        raise RuntimeError(f"http failed: code={response.http_code}, body={response.body}")
    payload = json.loads(response.body)
    if "results" not in payload or not payload["results"]:
        raise RuntimeError(f"unexpected payload: {payload}")
    result = payload["results"][0]
    if "error" in result:
        raise RuntimeError(f"file result error: {result['error']}")
    stats = result.get("stats", {})
    return {
        "result": result,
        "stats": stats,
        "md_hash": sha256_text(result.get("md_content", "")),
        "middle_hash": sha256_json(result.get("middle_json", {})),
        "output_dir": result.get("output_dir", ""),
    }


def run_wave(base_url: str, requests: List[Dict[str, object]]) -> Dict[str, object]:
    start = time.time()
    url = f"{base_url}/file_parse"
    with ThreadPoolExecutor(max_workers=len(requests)) as executor:
        futures = []
        for request in requests:
            cmd = build_file_parse_cmd(url, Path(request["abs_path"]), request["fields"])
            futures.append(executor.submit(run_curl_with_status, cmd))
        responses = [future.result() for future in futures]
    elapsed_ms = (time.time() - start) * 1000.0
    parsed = [parse_single_result(response) for response in responses]
    return {
        "elapsed_ms": elapsed_ms,
        "requests": parsed,
    }


def check_correctness(
    requests: List[Dict[str, object]],
    wave: Dict[str, object],
    baseline_hashes: Dict[str, Dict[str, str]],
) -> List[str]:
    errors: List[str] = []
    seen_output_dirs = set()
    for request, result in zip(requests, wave["requests"]):
        baseline = baseline_hashes.get(request["name"])
        if baseline is None:
            continue
        if baseline["md_hash"] != result["md_hash"]:
            errors.append(f"md hash mismatch for {request['name']}")
        if baseline["middle_hash"] != result["middle_hash"]:
            errors.append(f"middle_json hash mismatch for {request['name']}")
        output_dir = result["output_dir"]
        if output_dir in seen_output_dirs:
            errors.append(f"duplicate output_dir detected: {output_dir}")
        seen_output_dirs.add(output_dir)
    return errors


def isolation_gate(base_url: str, good_request: Dict[str, object], baseline: Dict[str, str]) -> Dict[str, object]:
    url = f"{base_url}/file_parse"
    good_cmd = build_file_parse_cmd(url, Path(good_request["abs_path"]), good_request["fields"])
    bad_cmd = " ".join([
        "curl",
        "-sS",
        "--max-time 30",
        "-X POST",
        "-F",
        shell_escape("return_content_list=true"),
        shell_escape(url),
    ])
    first = parse_single_result(run_curl_with_status(good_cmd))
    bad = run_curl_with_status(bad_cmd)
    second = parse_single_result(run_curl_with_status(good_cmd))
    return {
        "first_ok": first["md_hash"] == baseline["md_hash"] and first["middle_hash"] == baseline["middle_hash"],
        "bad_http_code": bad.http_code,
        "second_ok": second["md_hash"] == baseline["md_hash"] and second["middle_hash"] == baseline["middle_hash"],
    }


def gate_failures_for_result(
    result: Dict[str, object],
    correctness_errors: List[str],
    isolation: Optional[Dict[str, object]],
) -> List[str]:
    failures = list(correctness_errors)
    error_count = int(result.get("error_count", 0))
    if error_count != 0:
        failures.append(f"error_count_nonzero:{error_count}")
    if isolation is not None:
        if not isolation.get("first_ok", False):
            failures.append("isolation_first_request_mismatch")
        if isolation.get("bad_http_code", 200) == 200:
            failures.append("isolation_bad_request_unexpected_success")
        if not isolation.get("second_ok", False):
            failures.append("isolation_followup_request_mismatch")
    return failures


def make_stats_summary(waves: List[Dict[str, object]]) -> Dict[str, object]:
    all_requests = [req for wave in waves for req in wave["requests"]]
    summary = {
        "total_completion_time_ms": summarize([wave["elapsed_ms"] for wave in waves]),
        "qps": summarize([
            0.0 if wave["elapsed_ms"] <= 0 else (len(wave["requests"]) * 1000.0 / wave["elapsed_ms"])
            for wave in waves
        ]),
    }
    for key in THROUGHPUT_KEYS:
        summary[key] = summarize([float(req["stats"].get(key, 0.0)) for req in all_requests])
    return summary


def start_topology_a(
    server_binary: Path,
    host: str,
    port: int,
    device_ids: List[int],
    extra_flags: List[str],
    project_root: Path,
) -> Tuple[ServerHandle, str]:
    argv = [
        str(server_binary),
        "--host", host,
        "--port", str(port),
        "--workers", "1",
        "--topology", "single_process_multi_device",
        "--routing-policy", "least_inflight_rr",
        "--server-id", f"topology_a_{port}",
    ]
    if device_ids:
        argv.extend(["--device-ids", ",".join(str(d) for d in device_ids)])
    argv.extend(extra_flags)
    log_path = Path(tempfile.gettempdir()) / f"rapiddoc_topology_a_{port}.log"
    handle = ServerHandle(argv, f"http://{host}:{port}/status", log_path)
    handle.start()
    handle.wait_ready()
    return handle, f"http://{host}:{port}"


def start_topology_b(
    server_binary: Path,
    lb_binary: Path,
    host: str,
    port: int,
    device_ids: List[int],
    extra_flags: List[str],
) -> Tuple[List[ServerHandle], ServerHandle, str]:
    backend_handles: List[ServerHandle] = []
    backend_urls: List[str] = []
    for index, device_id in enumerate(device_ids):
        backend_port = port + 10 + index
        argv = [
            str(server_binary),
            "--host", host,
            "--port", str(backend_port),
            "--workers", "1",
            "--topology", "single_card_backend",
            "--routing-policy", "least_inflight_rr",
            "--server-id", f"backend_{index}",
            "--device-id", str(device_id),
        ]
        argv.extend(extra_flags)
        log_path = Path(tempfile.gettempdir()) / f"rapiddoc_topology_b_backend_{backend_port}.log"
        handle = ServerHandle(argv, f"http://{host}:{backend_port}/status", log_path)
        handle.start()
        handle.wait_ready()
        backend_handles.append(handle)
        backend_urls.append(f"http://{host}:{backend_port}")

    lb_argv = [
        str(lb_binary),
        "--host", host,
        "--port", str(port),
        "--workers", "1",
        "--server-id", f"front_lb_{port}",
        "--routing-policy", "least_inflight_rr",
    ]
    for backend_url in backend_urls:
        lb_argv.extend(["--backend-url", backend_url])
    lb_log = Path(tempfile.gettempdir()) / f"rapiddoc_topology_lb_{port}.log"
    lb_handle = ServerHandle(lb_argv, f"http://{host}:{port}/status", lb_log)
    lb_handle.start()
    lb_handle.wait_ready()
    return backend_handles, lb_handle, f"http://{host}:{port}"


def stop_all(handles: List[ServerHandle]) -> None:
    for handle in handles:
        handle.stop()


def run_profile(
    topology: str,
    device_ids: List[int],
    profile_name: str,
    args: argparse.Namespace,
    baseline_hashes: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, object]:
    if len(device_ids) == 0:
        return {"status": "blocked_device_unavailable"}

    manifest_path = args.project_root / PROFILE_MANIFESTS[profile_name]
    manifest = load_manifest(manifest_path)
    requests = make_request_batch(manifest, args.concurrency)
    for request in requests:
        request["abs_path"] = str((args.project_root / request["path"]).resolve())

    extra_flags = PROFILE_FLAGS[profile_name]
    port = args.base_port + (100 if topology == "A" else 300) + len(device_ids) * 10

    backend_handles: List[ServerHandle] = []
    lb_handle: Optional[ServerHandle] = None
    server_handle: Optional[ServerHandle] = None
    base_url = ""
    try:
        if topology == "A":
            server_handle, base_url = start_topology_a(
                args.server_binary,
                args.host,
                port,
                device_ids,
                extra_flags,
                args.project_root,
            )
        else:
            backend_handles, lb_handle, base_url = start_topology_b(
                args.server_binary,
                args.lb_binary,
                args.host,
                port,
                device_ids,
                extra_flags,
            )

        for _ in range(args.warmup_waves):
            run_wave(base_url, requests)

        measured_waves: List[Dict[str, object]] = []
        correctness_errors: List[str] = []
        for _ in range(args.measure_waves):
            wave = run_wave(base_url, requests)
            measured_waves.append(wave)
            if baseline_hashes is not None:
                correctness_errors.extend(check_correctness(requests, wave, baseline_hashes))

        status_snapshot = (lb_handle or server_handle).status()  # type: ignore[arg-type]
        isolation = None
        if baseline_hashes is not None:
            isolation = isolation_gate(base_url, requests[0], baseline_hashes[requests[0]["name"]])

        result = {
            "status": "ok",
            "topology": topology,
            "profile": profile_name,
            "device_count": len(device_ids),
            "device_ids": device_ids,
            "waves": measured_waves,
            "summary": make_stats_summary(measured_waves),
            "status_snapshot": status_snapshot,
            "correctness_errors": correctness_errors,
            "error_count": status_snapshot.get("errors", 0),
            "isolation": isolation,
        }
        gate_failures = gate_failures_for_result(result, correctness_errors, isolation)
        result["gate_failures"] = gate_failures
        if correctness_errors:
            result["status"] = "invalid_correctness_mismatch"
        elif any(failure.startswith("error_count_nonzero") for failure in gate_failures):
            result["status"] = "invalid_error_count_nonzero"
        elif gate_failures:
            result["status"] = "invalid_isolation_gate_failed"
        return result
    finally:
        if lb_handle is not None:
            lb_handle.stop()
        stop_all(backend_handles)
        if server_handle is not None:
            server_handle.stop()


def make_baseline_hashes(
    result: Dict[str, object],
    manifest: List[Dict[str, object]],
    concurrency: int,
) -> Dict[str, Dict[str, str]]:
    batch = make_request_batch(manifest, concurrency)
    hashes: Dict[str, Dict[str, str]] = {}
    first_wave = result["waves"][0]
    for request, parsed in zip(batch, first_wave["requests"]):
        hashes[request["name"]] = {
            "md_hash": parsed["md_hash"],
            "middle_hash": parsed["middle_hash"],
        }
    return hashes


def pick_best_topology(section1_rows: List[Dict[str, object]]) -> str:
    ok_rows = [row for row in section1_rows if row.get("status") == "ok"]
    if not ok_rows:
        return "A"
    ok_rows.sort(
        key=lambda row: (
            row["device_count"],
            row["summary"]["qps"]["mean"],
            row["topology"] == "B",
        ),
        reverse=True,
    )
    return ok_rows[0]["topology"]


def rank_bottlenecks(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return []
    baseline = next((row for row in ok_rows if row["device_count"] == 1), ok_rows[0])
    ranked = []
    for key in ["lb_proxy_ms", "route_queue_ms", "npu_lock_wait_ms", "npu_serial_ms", "cpu_only_ms"]:
        baseline_mean = baseline["summary"][key]["mean"]
        current_mean = ok_rows[-1]["summary"][key]["mean"]
        ranked.append({
            "name": key,
            "mean_ms": current_mean,
            "delta_vs_1card_ms": current_mean - baseline_mean,
        })
    ranked.sort(key=lambda item: (item["mean_ms"], item["delta_vs_1card_ms"]), reverse=True)
    return ranked


def investment_verdict(single_model_rows: List[Dict[str, object]]) -> str:
    efficiency_2 = [row for row in single_model_rows if row.get("device_count") == 2 and row.get("status") == "ok"]
    efficiency_3 = [row for row in single_model_rows if row.get("device_count") == 3 and row.get("status") == "ok"]
    if not efficiency_2 or not efficiency_3:
        return "blocked_device_unavailable"

    e2 = max(row["efficiency"] for row in efficiency_2)
    e3 = max(row["efficiency"] for row in efficiency_3)
    if e2 >= 0.75 and e3 >= 0.60:
        return "继续投资"
    if e2 > 0.0 or e3 > 0.0:
        return "保留观察"
    return "不建议继续"


def build_report(report: Dict[str, object]) -> str:
    lines = [
        "# 多卡拓扑归因战报告",
        "",
        "## 1. 单进程多卡 vs 一卡一进程对比",
    ]
    for row in report["section1"]:
        label = f"{row['topology']} {row['device_count']}卡"
        if row.get("status") != "ok":
            lines.append(f"- {label}: {row['status']}")
            continue
        summary = row["summary"]
        lines.append(
            f"- {label}: ok | total_completion_time_ms={summary['total_completion_time_ms']['mean']:.2f} | "
            f"QPS={summary['qps']['mean']:.2f} | pipeline_call_ms={summary['pipeline_call_ms']['mean']:.2f} | "
            f"npu_serial_ms={summary['npu_serial_ms']['mean']:.2f} | "
            f"npu_lock_wait_ms={summary['npu_lock_wait_ms']['mean']:.2f}"
        )
        if row.get("gate_failures"):
            lines.append(f"- {label} gates: {', '.join(row['gate_failures'])}")

    lines.extend(["", "## 2. per-device 利用分布"])
    for item in report["section2"]:
        lines.append(f"- {item['label']}: memory_status={item['memory_status']}")
        if not item["per_device"]:
            lines.append(f"- {item['label']} per_device: none")
            continue
        for device in item["per_device"]:
            lines.append(
                f"- {item['label']} device={device.get('device_id', -1)} backend={device.get('backend_id', '')} "
                f"requests={device.get('request_count', 0)} busy_ms={float(device.get('busy_time_ms', 0.0)):.2f} "
                f"avg_infer_ms={float(device.get('avg_infer_ms', 0.0)):.2f} "
                f"peak_mem={device.get('memory_peak_used_bytes')} imbalance={device.get('load_imbalance_flag', False)}"
            )

    lines.extend(["", "## 3. 单模型 1/2/3 卡扩展效率"])
    for row in report["section3"]:
        label = f"{row['profile']} {row['device_count']}卡"
        if row.get("status") != "ok":
            lines.append(f"- {label}: {row['status']}")
            continue
        lines.append(
            f"- {label}: ok | speedup={row.get('speedup', 0.0):.3f} | efficiency={row.get('efficiency', 0.0):.3f} | "
            f"QPS={row['summary']['qps']['mean']:.2f}"
        )
        if row.get("gate_failures"):
            lines.append(f"- {label} gates: {', '.join(row['gate_failures'])}")

    lines.extend(["", "## 4. 新瓶颈排序"])
    for item in report["section4"]:
        lines.append(
            f"- {item['name']}: mean_ms={item['mean_ms']:.2f}, delta_vs_1card_ms={item['delta_vs_1card_ms']:.2f}"
        )

    lines.extend(["", "## 5. 是否仍值得继续投资多卡方案"])
    lines.append(f"- 结论: {report['section5']['verdict']}")
    lines.append(f"- 最优拓扑: {report['section5']['best_topology']}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--server-binary", default="build/bin/rapid_doc_npu_test_server")
    parser.add_argument("--lb-binary", default="build/bin/rapid_doc_topology_lb")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=19880)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--warmup-waves", type=int, default=1)
    parser.add_argument("--measure-waves", type=int, default=5)
    parser.add_argument("--out-dir", default="output-benchmark/topology_attribution")
    args = parser.parse_args()

    args.project_root = Path(args.project_root).resolve()
    args.server_binary = (args.project_root / args.server_binary).resolve()
    args.lb_binary = (args.project_root / args.lb_binary).resolve()
    out_dir = (args.project_root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    available_devices = detect_dxrt_device_count()
    requested_device_counts = [1, 2, 3]

    section1: List[Dict[str, object]] = []
    baseline_manifest = load_manifest(args.project_root / PROFILE_MANIFESTS["mixed_baseline"])
    baseline_hashes: Optional[Dict[str, Dict[str, str]]] = None

    for topology in ["A", "B"]:
        for device_count in requested_device_counts:
            if device_count > available_devices:
                section1.append({
                    "topology": topology,
                    "device_count": device_count,
                    "status": "blocked_device_unavailable",
                })
                continue
            device_ids = list(range(device_count))
            result = run_profile(topology, device_ids, "mixed_baseline", args, baseline_hashes)
            section1.append(result)
            if topology == "A" and device_count == 1 and result.get("status") == "ok":
                baseline_hashes = make_baseline_hashes(result, baseline_manifest, args.concurrency)

    best_topology = pick_best_topology(section1)

    section2: List[Dict[str, object]] = []
    for row in section1:
        if row.get("status") != "ok":
            continue
        status_snapshot = row["status_snapshot"]
        section2.append({
            "label": f"{row['topology']}_{row['device_count']}card",
            "memory_status": status_snapshot.get("topology", {}).get(
                "memory_telemetry_status",
                "blocked_memory_telemetry_unavailable",
            ),
            "per_device": status_snapshot.get("per_device", []),
        })

    section3: List[Dict[str, object]] = []
    baselines_by_profile: Dict[str, Dict[str, Dict[str, str]]] = {}
    qps_baseline_by_profile: Dict[str, float] = {}
    for profile_name in ["layout_only", "ocr_only", "table_only"]:
        manifest = load_manifest(args.project_root / PROFILE_MANIFESTS[profile_name])
        for device_count in requested_device_counts:
            if device_count > available_devices:
                section3.append({
                    "profile": profile_name,
                    "device_count": device_count,
                    "status": "blocked_device_unavailable",
                })
                continue
            device_ids = list(range(device_count))
            baseline_hash = baselines_by_profile.get(profile_name)
            result = run_profile(best_topology, device_ids, profile_name, args, baseline_hash)
            if device_count == 1 and result.get("status") == "ok":
                baselines_by_profile[profile_name] = make_baseline_hashes(result, manifest, args.concurrency)
                qps_baseline_by_profile[profile_name] = result["summary"]["qps"]["mean"]
            if result.get("status") == "ok" and profile_name in qps_baseline_by_profile:
                qps = result["summary"]["qps"]["mean"]
                baseline_qps = qps_baseline_by_profile[profile_name]
                result["speedup"] = 0.0 if baseline_qps <= 0 else qps / baseline_qps
                result["efficiency"] = 0.0 if device_count <= 0 else result["speedup"] / device_count
            section3.append(result)

    section4 = rank_bottlenecks([row for row in section1 if row.get("topology") == best_topology])
    section5 = {
        "best_topology": best_topology,
        "verdict": investment_verdict(section3),
    }

    report = {
        "generated_at": int(time.time()),
        "available_device_count": available_devices,
        "section1": section1,
        "section2": section2,
        "section3": section3,
        "section4": section4,
        "section5": section5,
    }

    (out_dir / "topology_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (out_dir / "topology_report.md").write_text(build_report(report))
    print(json.dumps({
        "json": str(out_dir / "topology_report.json"),
        "markdown": str(out_dir / "topology_report.md"),
        "available_device_count": available_devices,
        "best_topology": best_topology,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
