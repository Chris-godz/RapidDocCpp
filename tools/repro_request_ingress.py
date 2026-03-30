#!/usr/bin/env python3
"""
Debug-only reproduction for request ingress serialization.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple


def load_probe_module(project_root: Path):
    probe_path = project_root / "tools" / "npu_topology_probe.py"
    spec = importlib.util.spec_from_file_location("npu_topology_probe", probe_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load probe module from {probe_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def parse_csv_ints(raw: str) -> List[int]:
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(int(token))
    return values


def summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {
            "samples": 0.0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    return {
        "samples": float(len(values)),
        "mean": statistics.fmean(values),
        "min": min(values),
        "max": max(values),
    }


def start_monitor(out_path: Path) -> Tuple[subprocess.Popen[str], object]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = out_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        ["dxrt-cli", "--monitor", "1"],
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc, log_handle


def stop_monitor(proc: Optional[subprocess.Popen[str]], log_handle: Optional[object]) -> None:
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=8)
    if log_handle is not None:
        log_handle.close()


def make_request_batch(
    probe,
    project_root: Path,
    workload: str,
    concurrency: int,
    output_dir: Path,
) -> List[Dict[str, object]]:
    manifest_path = project_root / probe.WORKLOAD_MANIFESTS[workload]
    manifest = probe.load_manifest(manifest_path)
    requests = probe.make_request_batch(manifest, concurrency, output_dir)
    for request in requests:
        request["abs_path"] = str((project_root / request["path"]).resolve())
    return requests


def run_wave_with_trace(probe, base_url: str, requests: List[Dict[str, object]]) -> Dict[str, object]:
    start_ts = time.time()
    url = f"{base_url}/file_parse"

    def invoke(index: int, request: Dict[str, object]) -> Dict[str, object]:
        cmd = probe.build_file_parse_cmd(url, Path(str(request["abs_path"])), request["fields"])
        submit_ts = time.time()
        response = probe.run_curl_with_status(cmd)
        done_ts = time.time()
        parsed = probe.parse_single_result(response)
        stats = parsed["stats"]
        result = parsed["result"]
        return {
            "request_index": index,
            "request_id": request["request_id"],
            "submit_ts": submit_ts,
            "done_ts": done_ts,
            "client_elapsed_ms": (done_ts - submit_ts) * 1000.0,
            "result": {
                "device_id": result.get("device_id"),
                "shard_id": result.get("shard_id"),
                "backend_id": result.get("backend_id", ""),
                "topology": result.get("topology"),
                "output_dir": parsed.get("output_dir", ""),
                "pipeline_call_ms": float(stats.get("pipeline_call_ms", 0.0)),
                "npu_serial_ms": float(stats.get("npu_serial_ms", 0.0)),
                "npu_lock_wait_ms": float(stats.get("npu_lock_wait_ms", 0.0)),
                "route_queue_ms": float(stats.get("route_queue_ms", 0.0)),
                "lb_proxy_ms": float(stats.get("lb_proxy_ms", 0.0)),
            },
        }

    with ThreadPoolExecutor(max_workers=len(requests)) as executor:
        futures = [
            executor.submit(invoke, index, request)
            for index, request in enumerate(requests)
        ]
        traced = [future.result() for future in futures]
    elapsed_ms = (time.time() - start_ts) * 1000.0
    traced.sort(key=lambda item: int(item["request_index"]))

    request_ms = [float(item["result"]["pipeline_call_ms"]) for item in traced]
    sum_request_ms = sum(request_ms)
    max_request_ms = max(request_ms) if request_ms else 0.0
    distance_to_sum = abs(elapsed_ms - sum_request_ms)
    distance_to_max = abs(elapsed_ms - max_request_ms)
    closer_to = "sum" if distance_to_sum <= distance_to_max else "max"

    requests_payload = []
    timeline_payload = []
    for item in traced:
        requests_payload.append({
            "request_id": item["request_id"],
            **item["result"],
        })
        timeline_payload.append({
            "request_id": item["request_id"],
            "submit_ts": item["submit_ts"],
            "done_ts": item["done_ts"],
            "client_elapsed_ms": item["client_elapsed_ms"],
        })

    return {
        "elapsed_ms": elapsed_ms,
        "sum_request_pipeline_ms": sum_request_ms,
        "max_request_pipeline_ms": max_request_ms,
        "distance_to_sum_ms": distance_to_sum,
        "distance_to_max_ms": distance_to_max,
        "closer_to": closer_to,
        "requests": requests_payload,
        "client_timeline": timeline_payload,
    }


def active_device_count(status_snapshot: Dict[str, object]) -> int:
    active = 0
    for item in status_snapshot.get("per_device", []):
        busy_ms = float(item.get("busy_time_ms", 0.0))
        request_count = int(item.get("request_count", 0))
        if busy_ms > 0.0 or request_count > 0:
            active += 1
    return active


def run_case(
    probe,
    args: argparse.Namespace,
    workers: int,
) -> Dict[str, object]:
    out_dir = args.out_dir / f"workers{workers}"
    out_dir.mkdir(parents=True, exist_ok=True)
    requests = make_request_batch(
        probe,
        args.project_root,
        args.workload,
        args.concurrency,
        out_dir / "request_outputs",
    )

    extra_flags = list(probe.WORKLOAD_SERVER_FLAGS.get(args.workload, []))
    server_handle = None
    lb_handle = None
    backend_handles: List[object] = []
    monitor_proc = None
    monitor_handle = None

    try:
        if args.topology == "A":
            server_handle, base_url = probe.start_topology_a(
                Path(args.server_binary),
                args.host,
                args.base_port + workers,
                args.device_ids,
                extra_flags,
                server_workers=workers,
            )
        else:
            backend_handles, lb_handle, base_url = probe.start_topology_b(
                Path(args.server_binary),
                Path(args.lb_binary),
                args.host,
                args.base_port + workers,
                args.device_ids,
                extra_flags,
                backend_workers=workers,
                lb_workers=workers,
            )

        write_json(
            out_dir / "server_launch.json",
            {
                "topology": args.topology,
                "workers": workers,
                "device_ids": args.device_ids,
                "server_status_url": (lb_handle or server_handle).ready_url,  # type: ignore[union-attr]
                "server_log_path": str((lb_handle or server_handle).log_path),  # type: ignore[union-attr]
                "backend_log_paths": [str(handle.log_path) for handle in backend_handles],
            },
        )

        monitor_proc, monitor_handle = start_monitor(out_dir / "dxrt_monitor.log")
        time.sleep(0.2)

        for _ in range(args.warmup_waves):
            run_wave_with_trace(probe, base_url, requests)

        waves: List[Dict[str, object]] = []
        flattened_requests: List[Dict[str, object]] = []
        flattened_timeline: List[Dict[str, object]] = []
        for wave_index in range(args.measure_waves):
            wave = run_wave_with_trace(probe, base_url, requests)
            wave["wave_index"] = wave_index
            waves.append(wave)
            for request in wave["requests"]:
                flattened_requests.append({"wave_index": wave_index, **request})
            for timeline in wave["client_timeline"]:
                flattened_timeline.append({"wave_index": wave_index, **timeline})

        status_snapshot = (lb_handle or server_handle).status()  # type: ignore[union-attr]

    finally:
        stop_monitor(monitor_proc, monitor_handle)
        if lb_handle is not None:
            lb_handle.stop()
        probe.stop_all(backend_handles)
        if server_handle is not None:
            server_handle.stop()

    waves_payload = [
        {
            "wave_index": wave["wave_index"],
            "elapsed_ms": wave["elapsed_ms"],
            "sum_request_pipeline_ms": wave["sum_request_pipeline_ms"],
            "max_request_pipeline_ms": wave["max_request_pipeline_ms"],
            "distance_to_sum_ms": wave["distance_to_sum_ms"],
            "distance_to_max_ms": wave["distance_to_max_ms"],
            "closer_to": wave["closer_to"],
        }
        for wave in waves
    ]

    write_json(out_dir / "waves.json", waves_payload)
    write_json(out_dir / "requests.json", flattened_requests)
    write_json(out_dir / "client_timeline.json", flattened_timeline)
    write_json(out_dir / "status_snapshot.json", status_snapshot)

    elapsed_values = [float(wave["elapsed_ms"]) for wave in waves]
    sum_values = [float(wave["sum_request_pipeline_ms"]) for wave in waves]
    max_values = [float(wave["max_request_pipeline_ms"]) for wave in waves]
    closer_to_sum = sum(1 for wave in waves if wave["closer_to"] == "sum")
    closer_to_max = sum(1 for wave in waves if wave["closer_to"] == "max")
    active_devices = active_device_count(status_snapshot)

    note_lines = [
        f"# workers={workers} operator note",
        "",
        f"- topology: {args.topology}",
        f"- workload: {args.workload}",
        f"- device_ids: {args.device_ids}",
        f"- concurrency: {args.concurrency}",
        f"- warmup_waves: {args.warmup_waves}",
        f"- measure_waves: {args.measure_waves}",
        f"- waves_closer_to_sum: {closer_to_sum}",
        f"- waves_closer_to_max: {closer_to_max}",
        f"- active_devices_by_status_snapshot: {active_devices}",
        f"- dxrt_monitor_log: {out_dir / 'dxrt_monitor.log'}",
        "- note: `dxrt-cli --monitor 1` does not expose per-device utilization percentages on this host; busy-device conclusion is based on request routing plus `/status` per-device busy_time.",
    ]
    write_text(out_dir / "operator_note.md", "\n".join(note_lines) + "\n")

    case_summary = {
        "workers": workers,
        "topology": args.topology,
        "workload": args.workload,
        "device_ids": args.device_ids,
        "concurrency": args.concurrency,
        "measure_waves": args.measure_waves,
        "waves_closer_to_sum": closer_to_sum,
        "waves_closer_to_max": closer_to_max,
        "elapsed_ms": summarize(elapsed_values),
        "sum_request_pipeline_ms": summarize(sum_values),
        "max_request_pipeline_ms": summarize(max_values),
        "active_devices": active_devices,
        "status_snapshot_path": str(out_dir / "status_snapshot.json"),
        "requests_path": str(out_dir / "requests.json"),
        "client_timeline_path": str(out_dir / "client_timeline.json"),
        "waves_path": str(out_dir / "waves.json"),
        "dxrt_monitor_log_path": str(out_dir / "dxrt_monitor.log"),
        "operator_note_path": str(out_dir / "operator_note.md"),
    }
    write_json(out_dir / "case_summary.json", case_summary)
    return case_summary


def build_verdict(workers1: Dict[str, object], workers2: Dict[str, object]) -> Dict[str, object]:
    workers1_sum = int(workers1["waves_closer_to_sum"])
    workers1_max = int(workers1["waves_closer_to_max"])
    workers2_sum = int(workers2["waves_closer_to_sum"])
    workers2_max = int(workers2["waves_closer_to_max"])
    workers2_active_devices = int(workers2["active_devices"])

    if workers1_sum >= workers1_max and workers2_max > workers2_sum and workers2_active_devices >= 2:
        verdict = "workers2_opens_ingress"
        attribution = "workers=2 后 wave elapsed 更接近 max(request_ms)，且至少两张卡有稳定 busy_time，问题位于系统级请求并发入口。"
    elif workers2_sum >= workers2_max:
        verdict = "not_crow_only"
        attribution = "workers=2 后仍接近 sum(request_ms)，需要继续向下查 requestMutex / 共享锁 / client 隐性串行。"
    else:
        verdict = "mixed"
        attribution = "workers=2 改善了重叠执行，但证据不够干净，需要继续复测或向下查串行门。"

    return {
        "verdict": verdict,
        "attribution": attribution,
    }


def safe_pair_verdict(
    workers1: Optional[Dict[str, object]],
    workers2: Optional[Dict[str, object]],
) -> Dict[str, object]:
    if not workers1 or not workers2:
        return {
            "verdict": "not_requested",
            "attribution": "当前运行没有同时包含 workers=1 与 workers=2，对比结论未生成。",
        }
    required = {"waves_closer_to_sum", "waves_closer_to_max", "active_devices"}
    if not required.issubset(workers1.keys()) or not required.issubset(workers2.keys()):
        return {
            "verdict": "incomplete",
            "attribution": "workers=1/2 对比数据不完整，对比结论未生成。",
        }
    return build_verdict(workers1, workers2)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--out-dir", default="output-benchmark/request_ingress_bug_main-e572512_ocr-34cda45")
    parser.add_argument("--server-binary", default="build_Release/bin/rapid_doc_npu_test_server")
    parser.add_argument("--lb-binary", default="build_Release/bin/rapid_doc_topology_lb")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=20380)
    parser.add_argument("--topology", choices=["A", "B"], default="A")
    parser.add_argument("--workload", default="layout_ocr")
    parser.add_argument("--device-ids", default="0,1,2")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--warmup-waves", type=int, default=1)
    parser.add_argument("--measure-waves", type=int, default=5)
    parser.add_argument("--workers-list", default="1,2")
    parser.add_argument("--expected-main-commit", default="e57251269f134e64febb002bb8173e4995f27bfe")
    parser.add_argument("--expected-ocr-commit", default="")
    args = parser.parse_args()

    args.project_root = Path(args.project_root).resolve()
    args.out_dir = (args.project_root / args.out_dir).resolve()
    probe = load_probe_module(args.project_root)
    if not args.expected_ocr_commit:
        args.expected_ocr_commit = probe.EXPECTED_OCR_COMMIT
    args.server_binary = str((args.project_root / args.server_binary).resolve())
    args.lb_binary = str((args.project_root / args.lb_binary).resolve())
    args.device_ids = parse_csv_ints(args.device_ids)
    workers_list = parse_csv_ints(args.workers_list)

    baseline_args = SimpleNamespace(
        expected_main_commit=args.expected_main_commit,
        expected_ocr_commit=args.expected_ocr_commit,
    )
    baseline = probe.verify_baseline(args.project_root, baseline_args)
    env_snapshot = probe.capture_env_snapshot()
    available_device_ids = probe.detect_dxrt_devices()

    root_manifest = {
        "generated_at": int(time.time()),
        "baseline": baseline,
        "env_snapshot": env_snapshot,
        "available_device_ids": available_device_ids,
        "topology": args.topology,
        "workload": args.workload,
        "device_ids": args.device_ids,
        "concurrency": args.concurrency,
        "warmup_waves": args.warmup_waves,
        "measure_waves": args.measure_waves,
        "workers_list": workers_list,
    }
    write_json(args.out_dir / "campaign_manifest.json", root_manifest)

    results: Dict[int, Dict[str, object]] = {}
    for workers in workers_list:
        results[workers] = run_case(probe, args, workers)

    comparison = {
        "baseline": baseline,
        "env_snapshot": env_snapshot,
        "available_device_ids": available_device_ids,
        "cases_by_worker": {str(key): value for key, value in sorted(results.items())},
        "workers1": results.get(1),
        "workers2": results.get(2),
        "comparison": safe_pair_verdict(results.get(1), results.get(2)),
    }
    write_json(args.out_dir / "comparison_summary.json", comparison)

    report_lines = [
        "# Request ingress reproduction summary",
        "",
        f"- topology: {args.topology}",
        f"- workload: {args.workload}",
        f"- device_ids: {args.device_ids}",
        f"- concurrency: {args.concurrency}",
        f"- baseline_main_commit: {baseline.get('main_commit')}",
        f"- baseline_ocr_checkout_commit: {baseline.get('ocr_checkout_commit')}",
        "",
        "## workers=1",
        f"- summary: {json.dumps(results.get(1, {}), ensure_ascii=False)}",
        "",
        "## workers=2",
        f"- summary: {json.dumps(results.get(2, {}), ensure_ascii=False)}",
        "",
        "## verdict",
        f"- {comparison['comparison']['verdict']}: {comparison['comparison']['attribution']}",
    ]
    write_text(args.out_dir / "comparison_summary.md", "\n".join(report_lines) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
