#!/usr/bin/env python3
"""
Run formal multi-device A/B experiments for RapidDocCpp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


EXPECTED_MAIN_COMMIT = ""
EXPECTED_OCR_COMMIT = "34cda458cddd8dbf708639cd0c8089804c5132d1"
FORMAL_WORKLOADS = ["layout_ocr", "layout_ocr_table", "table_heavy"]
FORMAL_ENV_PROFILE = {
    "CUSTOM_INTER_OP_THREADS_COUNT": "1",
    "CUSTOM_INTRA_OP_THREADS_COUNT": "2",
    "DXRT_DYNAMIC_CPU_THREAD": "ON",
    "DXRT_TASK_MAX_LOAD": "3",
    "NFH_INPUT_WORKER_THREADS": "2",
    "NFH_OUTPUT_WORKER_THREADS": "4",
}
WORKLOAD_MANIFESTS = {
    "layout_ocr": "tools/topology_workloads/layout_ocr.json",
    "layout_ocr_table": "tools/topology_workloads/layout_ocr_table.json",
    "table_heavy": "tools/topology_workloads/table_heavy.json",
    "mixed_baseline": "tools/topology_workloads/mixed_baseline.json",
    "layout_only": "tools/topology_workloads/layout_only.json",
    "ocr_only": "tools/topology_workloads/ocr_only.json",
    "table_only": "tools/topology_workloads/table_only.json",
}
WORKLOAD_LABELS = {
    "layout_ocr": "layout+ocr",
    "layout_ocr_table": "layout+ocr+table",
    "table_heavy": "table-heavy",
    "mixed_baseline": "mixed_baseline",
    "layout_only": "layout_only",
    "ocr_only": "ocr_only",
    "table_only": "table_only",
}
WORKLOAD_SERVER_FLAGS = {
    "layout_ocr": [],
    "layout_ocr_table": [],
    "table_heavy": [],
    "mixed_baseline": [],
    "layout_only": ["--no-ocr", "--no-table"],
    "ocr_only": ["--no-table"],
    "table_only": ["--no-ocr"],
}
THROUGHPUT_KEYS = [
    "pipeline_call_ms",
    "npu_serial_ms",
    "npu_lock_wait_ms",
    "cpu_only_ms",
    "route_queue_ms",
    "lb_proxy_ms",
]
BOTTLENECK_KEYS = [
    "lb_proxy_ms",
    "route_queue_ms",
    "npu_lock_wait_ms",
    "npu_serial_ms",
    "cpu_only_ms",
]
ALLOWED_DIRTY_PREFIXES = (
    "tools/",
    "report/",
    "output-benchmark/",
)


@dataclass
class CurlResponse:
    exit_code: int
    http_code: int
    body: str
    raw_output: str


def run_shell(cmd: str, cwd: Optional[Path] = None) -> Tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
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


def parse_csv_ints(raw: str) -> List[int]:
    values: List[int] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values


def parse_csv_strings(raw: str) -> List[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def capture_env_snapshot() -> Dict[str, Optional[str]]:
    return {key: os.environ.get(key) for key in FORMAL_ENV_PROFILE}


def expected_env_mismatches(snapshot: Dict[str, Optional[str]]) -> List[str]:
    mismatches: List[str] = []
    for key, expected in FORMAL_ENV_PROFILE.items():
        if snapshot.get(key) != expected:
            mismatches.append(f"{key}={snapshot.get(key)!r} expected {expected!r}")
    return mismatches


def workload_label(workload_name: str) -> str:
    return WORKLOAD_LABELS.get(workload_name, workload_name)


def format_float(value: object) -> str:
    if value is None:
        return "null"
    try:
        return f"{float(value):.2f}"
    except Exception:
        return str(value)


def git_text(project_root: Path, args: str) -> Tuple[int, str]:
    return run_shell(f"git {args}", cwd=project_root)


def git_submodule_text(project_root: Path, args: str) -> Tuple[int, str]:
    submodule_root = project_root / "3rd-party" / "DXNN-OCR-cpp"
    return run_shell(f"git {args}", cwd=submodule_root)


def parse_dirty_paths(project_root: Path) -> List[str]:
    code, output = git_text(project_root, "status --porcelain")
    if code != 0:
        return []
    paths: List[str] = []
    for line in output.splitlines():
        if len(line) < 4:
            continue
        path = line[3:]
        if "->" in path:
            path = path.split("->", 1)[1].strip()
        paths.append(path.strip())
    return paths


def relevant_core_dirty_paths(paths: List[str]) -> List[str]:
    blocked: List[str] = []
    for path in paths:
        if any(path.startswith(prefix) for prefix in ALLOWED_DIRTY_PREFIXES):
            continue
        blocked.append(path)
    return sorted(blocked)


def verify_baseline(project_root: Path, args: argparse.Namespace) -> Dict[str, object]:
    failures: List[str] = []

    main_code, main_output = git_text(project_root, "rev-parse HEAD")
    main_commit = main_output.strip() if main_code == 0 else None

    gitlink_code, gitlink_output = git_text(
        project_root,
        "ls-tree HEAD 3rd-party/DXNN-OCR-cpp",
    )
    gitlink_commit = None
    if gitlink_code == 0 and gitlink_output.strip():
        parts = gitlink_output.strip().split()
        if len(parts) >= 3:
            gitlink_commit = parts[2]

    ocr_code, ocr_output = git_submodule_text(project_root, "rev-parse HEAD")
    ocr_checkout_commit = ocr_output.strip() if ocr_code == 0 else None

    ocr_available_code, _ = git_submodule_text(
        project_root,
        f"cat-file -e {shell_escape(args.expected_ocr_commit)}^{{commit}}",
    )
    expected_ocr_available = (ocr_available_code == 0)

    dirty_paths = parse_dirty_paths(project_root)
    blocked_dirty_paths = relevant_core_dirty_paths(dirty_paths)

    if args.expected_main_commit and main_commit != args.expected_main_commit:
        failures.append(
            f"main_commit_mismatch:{main_commit or 'unknown'} expected {args.expected_main_commit}"
        )
    if gitlink_commit != args.expected_ocr_commit:
        failures.append(
            f"ocr_gitlink_mismatch:{gitlink_commit or 'unknown'} expected {args.expected_ocr_commit}"
        )
    if not expected_ocr_available:
        failures.append(f"verified_ocr_source_unavailable:{args.expected_ocr_commit}")
    if ocr_checkout_commit != args.expected_ocr_commit:
        failures.append(
            f"ocr_checkout_mismatch:{ocr_checkout_commit or 'unknown'} expected {args.expected_ocr_commit}"
        )
    if blocked_dirty_paths:
        failures.append(
            "unexpected_dirty_core_paths:" + ",".join(blocked_dirty_paths)
        )

    status = "ok"
    if any(item.startswith("verified_ocr_source_unavailable") for item in failures):
        status = "blocked_verified_source_unavailable"
    elif any(item.startswith("unexpected_dirty_core_paths") for item in failures):
        status = "blocked_unexpected_dirty_core_paths"
    elif failures:
        status = "blocked_baseline_mismatch"

    return {
        "status": status,
        "project_root": str(project_root),
        "expected_main_commit": args.expected_main_commit,
        "expected_ocr_commit": args.expected_ocr_commit,
        "main_commit": main_commit,
        "ocr_gitlink_commit": gitlink_commit,
        "ocr_checkout_commit": ocr_checkout_commit,
        "expected_ocr_available": expected_ocr_available,
        "dirty_paths": dirty_paths,
        "blocked_dirty_paths": blocked_dirty_paths,
        "failures": failures,
    }


def detect_dxrt_devices() -> List[int]:
    code, output = run_shell("dxrt-cli --status")
    if code != 0:
        return []
    pattern = re.compile(r"^\s*\*\s*Device\s+(\d+):")
    device_ids: List[int] = []
    for line in output.splitlines():
        match = pattern.search(line)
        if match:
            device_ids.append(int(match.group(1)))
    return device_ids


def dxrt_status_snapshot() -> Dict[str, object]:
    code, output = run_shell("dxrt-cli --status")
    return {
        "exit_code": code,
        "output": output,
    }


class ServerHandle:
    def __init__(self, argv: List[str], ready_url: str, log_path: Path) -> None:
        self.argv = argv
        self.ready_url = ready_url
        self.log_path = log_path
        self.proc: Optional[subprocess.Popen[str]] = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
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


def make_request_batch(
    manifest: List[Dict[str, object]],
    concurrency: int,
    output_dir: Path,
) -> List[Dict[str, object]]:
    if not manifest:
        return []
    batch: List[Dict[str, object]] = []
    output_dir = output_dir.resolve()
    for idx in range(concurrency):
        source = dict(manifest[idx % len(manifest)])
        request = dict(source)
        request["fields"] = dict(source.get("fields", {}))
        request["request_id"] = f"{request['name']}_{idx}"
        request["fields"]["output_dir"] = str(output_dir)
        batch.append(request)
    return batch


def build_file_parse_cmd(url: str, pdf_path: Path, fields: Dict[str, object]) -> str:
    parts = [
        "curl",
        "-sS",
        "--max-time",
        "300",
        "-X",
        "POST",
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


def make_hashes_from_wave(
    requests: List[Dict[str, object]],
    wave: Dict[str, object],
) -> Dict[str, Dict[str, str]]:
    hashes: Dict[str, Dict[str, str]] = {}
    for request, parsed in zip(requests, wave["requests"]):
        hashes[request["name"]] = {
            "md_hash": parsed["md_hash"],
            "middle_hash": parsed["middle_hash"],
        }
    return hashes


def validate_wave(
    requests: List[Dict[str, object]],
    wave: Dict[str, object],
    baseline_hashes: Dict[str, Dict[str, str]],
    seen_output_dirs: Set[str],
) -> List[str]:
    errors: List[str] = []
    for request, result in zip(requests, wave["requests"]):
        baseline = baseline_hashes.get(request["name"])
        if baseline is not None:
            if baseline["md_hash"] != result["md_hash"]:
                errors.append(f"md hash mismatch for {request['name']}")
            if baseline["middle_hash"] != result["middle_hash"]:
                errors.append(f"middle_json hash mismatch for {request['name']}")
        output_dir = str(result.get("output_dir", ""))
        if not output_dir:
            errors.append(f"missing output_dir for {request['name']}")
            continue
        if output_dir in seen_output_dirs:
            errors.append(f"duplicate output_dir detected: {output_dir}")
            continue
        seen_output_dirs.add(output_dir)
    return errors


def isolation_gate(
    base_url: str,
    good_request: Dict[str, object],
    baseline: Dict[str, str],
    seen_output_dirs: Set[str],
) -> Dict[str, object]:
    url = f"{base_url}/file_parse"
    good_cmd = build_file_parse_cmd(url, Path(good_request["abs_path"]), good_request["fields"])
    bad_cmd = " ".join([
        "curl",
        "-sS",
        "--max-time",
        "30",
        "-X",
        "POST",
        "-F",
        shell_escape("return_content_list=true"),
        shell_escape(url),
    ])
    first = parse_single_result(run_curl_with_status(good_cmd))
    bad = run_curl_with_status(bad_cmd)
    second = parse_single_result(run_curl_with_status(good_cmd))

    output_dir_failures: List[str] = []
    for label, parsed in (("first", first), ("second", second)):
        output_dir = str(parsed.get("output_dir", ""))
        if not output_dir:
            output_dir_failures.append(f"{label}_missing_output_dir")
            continue
        if output_dir in seen_output_dirs:
            output_dir_failures.append(f"{label}_duplicate_output_dir:{output_dir}")
            continue
        seen_output_dirs.add(output_dir)

    return {
        "first_ok": (
            first["md_hash"] == baseline["md_hash"]
            and first["middle_hash"] == baseline["middle_hash"]
        ),
        "bad_http_code": bad.http_code,
        "second_ok": (
            second["md_hash"] == baseline["md_hash"]
            and second["middle_hash"] == baseline["middle_hash"]
        ),
        "first_output_dir": first["output_dir"],
        "second_output_dir": second["output_dir"],
        "output_dir_failures": output_dir_failures,
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
    worker_count = result.get("status_snapshot", {}).get("topology", {}).get("worker_count")
    if worker_count not in (None, 1):
        failures.append(f"worker_count_not_1:{worker_count}")
    if isolation is None:
        failures.append("isolation_not_run")
        return failures
    if not isolation.get("first_ok", False):
        failures.append("isolation_first_request_mismatch")
    if isolation.get("bad_http_code", 200) == 200:
        failures.append("isolation_bad_request_unexpected_success")
    if not isolation.get("second_ok", False):
        failures.append("isolation_followup_request_mismatch")
    for item in isolation.get("output_dir_failures", []):
        failures.append(item)
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


def blocked_result(
    topology: str,
    workload: str,
    concurrency: int,
    device_count: int,
    device_ids: List[int],
    status: str,
    detail: Optional[str] = None,
) -> Dict[str, object]:
    result = {
        "status": status,
        "topology": topology,
        "workload": workload,
        "concurrency": concurrency,
        "device_count": device_count,
        "device_ids": list(device_ids),
    }
    if detail:
        result["detail"] = detail
    return result


def compute_port(args: argparse.Namespace, topology: str, workload: str, device_count: int, concurrency: int) -> int:
    workload_index = args.workloads.index(workload)
    topology_offset = 0 if topology == "A" else 10000
    return args.base_port + topology_offset + workload_index * 200 + device_count * 20 + concurrency


def run_profile(
    topology: str,
    device_ids: List[int],
    workload_name: str,
    concurrency: int,
    args: argparse.Namespace,
    baseline_hashes: Optional[Dict[str, Dict[str, str]]] = None,
) -> Dict[str, object]:
    if not device_ids:
        return blocked_result(
            topology,
            workload_name,
            concurrency,
            0,
            [],
            "blocked_device_unavailable",
        )

    manifest_path = args.project_root / WORKLOAD_MANIFESTS[workload_name]
    manifest = load_manifest(manifest_path)
    group_output_dir = (
        args.request_output_root
        / topology
        / workload_name
        / f"{len(device_ids)}card"
        / f"c{concurrency}"
    )
    requests = make_request_batch(manifest, concurrency, group_output_dir)
    for request in requests:
        request["abs_path"] = str((args.project_root / request["path"]).resolve())

    extra_flags = list(WORKLOAD_SERVER_FLAGS.get(workload_name, []))
    port = compute_port(args, topology, workload_name, len(device_ids), concurrency)

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
        seen_output_dirs: Set[str] = set()
        canonical_hashes = baseline_hashes

        for _ in range(args.measure_waves):
            wave = run_wave(base_url, requests)
            measured_waves.append(wave)
            if canonical_hashes is None:
                canonical_hashes = make_hashes_from_wave(requests, wave)
            correctness_errors.extend(
                validate_wave(requests, wave, canonical_hashes, seen_output_dirs)
            )

        status_snapshot = (lb_handle or server_handle).status()  # type: ignore[arg-type]
        if canonical_hashes is None:
            canonical_hashes = {}
        isolation = None
        if requests:
            baseline = canonical_hashes.get(requests[0]["name"])
            if baseline is not None:
                isolation = isolation_gate(base_url, requests[0], baseline, seen_output_dirs)

        result = {
            "status": "ok",
            "topology": topology,
            "workload": workload_name,
            "concurrency": concurrency,
            "device_count": len(device_ids),
            "device_ids": list(device_ids),
            "waves": measured_waves,
            "summary": make_stats_summary(measured_waves),
            "status_snapshot": status_snapshot,
            "correctness_errors": correctness_errors,
            "error_count": status_snapshot.get("errors", 0),
            "isolation": isolation,
            "env": capture_env_snapshot(),
            "group_output_dir": str(group_output_dir),
        }
        gate_failures = gate_failures_for_result(result, correctness_errors, isolation)
        result["gate_failures"] = gate_failures
        if correctness_errors:
            result["status"] = "invalid_correctness_mismatch"
        elif any(failure.startswith("error_count_nonzero") for failure in gate_failures):
            result["status"] = "invalid_error_count_nonzero"
        elif any(failure.startswith("worker_count_not_1") for failure in gate_failures):
            result["status"] = "invalid_worker_count"
        elif gate_failures:
            result["status"] = "invalid_isolation_gate_failed"
        return result
    finally:
        if lb_handle is not None:
            lb_handle.stop()
        stop_all(backend_handles)
        if server_handle is not None:
            server_handle.stop()


def make_baseline_hashes_from_result(
    result: Dict[str, object],
    manifest: List[Dict[str, object]],
    concurrency: int,
    output_dir: Path,
) -> Dict[str, Dict[str, str]]:
    batch = make_request_batch(manifest, concurrency, output_dir)
    first_wave = result["waves"][0]
    return make_hashes_from_wave(batch, first_wave)


def summarize_topology_run(row: Optional[Dict[str, object]]) -> Dict[str, object]:
    if row is None:
        return {"status": "missing"}
    summary = {
        "status": row.get("status"),
        "gate_failures": row.get("gate_failures", []),
    }
    if row.get("status") == "ok":
        metrics = row["summary"]
        summary.update({
            "total_completion_time_ms": metrics["total_completion_time_ms"]["mean"],
            "qps": metrics["qps"]["mean"],
            "npu_serial_ms": metrics["npu_serial_ms"]["mean"],
            "npu_lock_wait_ms": metrics["npu_lock_wait_ms"]["mean"],
        })
    else:
        if "detail" in row:
            summary["detail"] = row["detail"]
    return summary


def build_section1(
    args: argparse.Namespace,
    results: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    indexed = {
        (row["topology"], row["workload"], row["concurrency"], row["device_count"]): row
        for row in results
    }
    section: List[Dict[str, object]] = []
    for workload in args.workloads:
        for concurrency in args.concurrency_list:
            for device_count in args.device_counts:
                a_row = indexed.get(("A", workload, concurrency, device_count))
                b_row = indexed.get(("B", workload, concurrency, device_count))
                comparison = {
                    "workload": workload,
                    "concurrency": concurrency,
                    "device_count": device_count,
                    "A": summarize_topology_run(a_row),
                    "B": summarize_topology_run(b_row),
                }
                if a_row and b_row and a_row.get("status") == "ok" and b_row.get("status") == "ok":
                    a_qps = float(a_row["summary"]["qps"]["mean"])
                    b_qps = float(b_row["summary"]["qps"]["mean"])
                    comparison["qps_winner"] = "B" if b_qps > a_qps else "A"
                    comparison["qps_ratio_b_vs_a"] = 0.0 if a_qps <= 0 else b_qps / a_qps
                section.append(comparison)
    return section


def build_section2(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    section: List[Dict[str, object]] = []
    for row in results:
        if row.get("status") != "ok":
            continue
        status_snapshot = row["status_snapshot"]
        section.append({
            "topology": row["topology"],
            "workload": row["workload"],
            "concurrency": row["concurrency"],
            "device_count": row["device_count"],
            "memory_status": status_snapshot.get("topology", {}).get(
                "memory_telemetry_status",
                "blocked_memory_telemetry_unavailable",
            ),
            "per_device": status_snapshot.get("per_device", []),
        })
    return section


def build_section3(results: List[Dict[str, object]]) -> List[Dict[str, object]]:
    baseline_by_key: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    for row in results:
        if row.get("status") == "ok" and row.get("device_count") == 1:
            baseline_by_key[(row["topology"], row["workload"], row["concurrency"])] = row

    section: List[Dict[str, object]] = []
    for row in results:
        entry = {
            "topology": row["topology"],
            "workload": row["workload"],
            "concurrency": row["concurrency"],
            "device_count": row["device_count"],
            "status": row.get("status"),
            "gate_failures": row.get("gate_failures", []),
        }
        baseline = baseline_by_key.get((row["topology"], row["workload"], row["concurrency"]))
        if row.get("status") == "ok":
            qps = float(row["summary"]["qps"]["mean"])
            entry["qps"] = qps
            if baseline is not None and baseline.get("status") == "ok":
                baseline_qps = float(baseline["summary"]["qps"]["mean"])
                speedup = 0.0 if baseline_qps <= 0 else qps / baseline_qps
                efficiency = 0.0 if row["device_count"] <= 0 else speedup / float(row["device_count"])
                entry["speedup"] = speedup
                entry["efficiency"] = efficiency
        section.append(entry)
    return section


def build_section4(
    results: List[Dict[str, object]],
) -> List[Dict[str, object]]:
    baseline_by_key: Dict[Tuple[str, str, int], Dict[str, object]] = {}
    for row in results:
        if row.get("status") == "ok" and row.get("device_count") == 1:
            baseline_by_key[(row["topology"], row["workload"], row["concurrency"])] = row

    deltas: Dict[str, List[float]] = {key: [] for key in BOTTLENECK_KEYS}
    for row in results:
        if row.get("status") != "ok" or int(row.get("device_count", 0)) <= 1:
            continue
        baseline = baseline_by_key.get((row["topology"], row["workload"], row["concurrency"]))
        if baseline is None or baseline.get("status") != "ok":
            continue
        for key in BOTTLENECK_KEYS:
            delta = float(row["summary"][key]["mean"]) - float(baseline["summary"][key]["mean"])
            deltas[key].append(delta)

    ranked: List[Dict[str, object]] = []
    for key, values in deltas.items():
        if not values:
            continue
        ranked.append({
            "name": key,
            "samples": len(values),
            "mean_delta_ms": statistics.fmean(values),
            "max_delta_ms": max(values),
            "min_delta_ms": min(values),
        })
    ranked.sort(key=lambda item: (item["mean_delta_ms"], item["max_delta_ms"]), reverse=True)
    return ranked


def build_section5(section3: List[Dict[str, object]]) -> Dict[str, object]:
    a_eff = [
        float(item["efficiency"])
        for item in section3
        if item.get("status") == "ok"
        and item.get("topology") == "A"
        and int(item.get("device_count", 0)) > 1
        and "efficiency" in item
    ]
    b_eff = [
        float(item["efficiency"])
        for item in section3
        if item.get("status") == "ok"
        and item.get("topology") == "B"
        and int(item.get("device_count", 0)) > 1
        and "efficiency" in item
    ]

    if not a_eff and not b_eff:
        return {
            "verdict": "blocked",
            "best_topology": "insufficient_data",
            "attribution": "没有足够的有效多卡数据生成正式结论。",
        }

    a_mean = statistics.fmean(a_eff) if a_eff else 0.0
    b_mean = statistics.fmean(b_eff) if b_eff else 0.0
    best_topology = "A" if a_mean > b_mean else "B"

    if a_eff and b_eff and b_mean >= a_mean + 0.10 and b_mean >= 0.45 and a_mean < 0.45:
        return {
            "verdict": "保留观察",
            "best_topology": best_topology,
            "attribution": "B 明显优于 A，更像单进程共享 runtime / 调度拓扑问题。",
            "a_mean_efficiency": a_mean,
            "b_mean_efficiency": b_mean,
        }
    if max(a_mean, b_mean) < 0.35:
        return {
            "verdict": "不建议继续",
            "best_topology": best_topology,
            "attribution": "A/B 都扩不起来，更偏平台 / DXRT / host 侧。",
            "a_mean_efficiency": a_mean,
            "b_mean_efficiency": b_mean,
        }
    if a_mean >= 0.60 and b_mean >= 0.60:
        return {
            "verdict": "继续投资",
            "best_topology": best_topology,
            "attribution": "A/B 都能扩起来，可以继续讨论是否把 B 作为正式多卡方案。",
            "a_mean_efficiency": a_mean,
            "b_mean_efficiency": b_mean,
        }
    return {
        "verdict": "保留观察",
        "best_topology": best_topology,
        "attribution": "结果混合，需要继续结合 per-device 利用与瓶颈分布做归因。",
        "a_mean_efficiency": a_mean,
        "b_mean_efficiency": b_mean,
    }


def format_summary(summary: Dict[str, object]) -> str:
    status = summary.get("status", "unknown")
    if status != "ok":
        detail = summary.get("detail")
        if detail:
            return f"{status} ({detail})"
        return str(status)
    return (
        "ok"
        f" | total_completion_time_ms={format_float(summary.get('total_completion_time_ms'))}"
        f" | QPS={format_float(summary.get('qps'))}"
        f" | npu_serial_ms={format_float(summary.get('npu_serial_ms'))}"
        f" | npu_lock_wait_ms={format_float(summary.get('npu_lock_wait_ms'))}"
    )


def build_report(report: Dict[str, object]) -> str:
    lines = [
        "# 正式多卡 A/B 实验报告",
        "",
        f"- report_status: {report.get('status', 'unknown')}",
        f"- available_device_ids: {report.get('available_device_ids', [])}",
        f"- main_commit: {report.get('baseline', {}).get('main_commit')}",
        f"- ocr_gitlink_commit: {report.get('baseline', {}).get('ocr_gitlink_commit')}",
        f"- ocr_checkout_commit: {report.get('baseline', {}).get('ocr_checkout_commit')}",
        f"- expected_main_commit: {report.get('baseline', {}).get('expected_main_commit')}",
        f"- expected_ocr_commit: {report.get('baseline', {}).get('expected_ocr_commit')}",
        f"- env_snapshot: {json.dumps(report.get('env_snapshot', {}), ensure_ascii=False)}",
    ]
    env_mismatches = report.get("env_mismatches", [])
    if env_mismatches:
        lines.append(f"- env_mismatches: {', '.join(env_mismatches)}")
    baseline_failures = report.get("baseline", {}).get("failures", [])
    if baseline_failures:
        lines.append(f"- baseline_failures: {', '.join(baseline_failures)}")

    lines.extend(["", "## 1. 单进程多卡 vs 一卡一进程对比"])
    section1 = report.get("section1", [])
    if not section1:
        lines.append(f"- blocked: {report.get('status', 'unknown')}")
    for row in section1:
        label = (
            f"{workload_label(row['workload'])} | {row['device_count']}卡 | 并发{row['concurrency']}"
        )
        lines.append(f"- {label}: A={format_summary(row['A'])} | B={format_summary(row['B'])}")
        if "qps_ratio_b_vs_a" in row:
            lines.append(
                f"- {label} winner: {row['qps_winner']} | B/A QPS={row['qps_ratio_b_vs_a']:.3f}"
            )

    lines.extend(["", "## 2. per-device 利用分布"])
    section2 = report.get("section2", [])
    if not section2:
        lines.append(f"- blocked: {report.get('status', 'unknown')}")
    for item in section2:
        label = (
            f"{item['topology']} | {workload_label(item['workload'])}"
            f" | {item['device_count']}卡 | 并发{item['concurrency']}"
        )
        lines.append(f"- {label}: memory_status={item['memory_status']}")
        per_device = item.get("per_device", [])
        if not per_device:
            lines.append(f"- {label} per_device: none")
            continue
        for device in per_device:
            lines.append(
                f"- {label} device={device.get('device_id', -1)} backend={device.get('backend_id', '')} "
                f"requests={device.get('request_count', 0)} busy_ms={format_float(device.get('busy_time_ms'))} "
                f"avg_infer_ms={format_float(device.get('avg_infer_ms'))} "
                f"peak_mem={device.get('memory_peak_used_bytes')} imbalance={device.get('load_imbalance_flag', False)}"
            )

    lines.extend(["", "## 3. 单模型 1/2/3 卡扩展效率"])
    section3 = report.get("section3", [])
    if not section3:
        lines.append(f"- blocked: {report.get('status', 'unknown')}")
    for row in section3:
        label = (
            f"{row['topology']} | {workload_label(row['workload'])}"
            f" | {row['device_count']}卡 | 并发{row['concurrency']}"
        )
        if row.get("status") != "ok":
            lines.append(f"- {label}: {row['status']}")
            continue
        lines.append(
            f"- {label}: ok | QPS={format_float(row.get('qps'))} "
            f"| speedup={float(row.get('speedup', 0.0)):.3f} "
            f"| efficiency={float(row.get('efficiency', 0.0)):.3f}"
        )
        if row.get("gate_failures"):
            lines.append(f"- {label} gates: {', '.join(row['gate_failures'])}")

    lines.extend(["", "## 4. 新瓶颈排序"])
    section4 = report.get("section4", [])
    if not section4:
        lines.append(f"- blocked: {report.get('status', 'unknown')}")
    for item in section4:
        lines.append(
            f"- {item['name']}: mean_delta_ms={item['mean_delta_ms']:.2f}, "
            f"max_delta_ms={item['max_delta_ms']:.2f}, samples={item['samples']}"
        )

    lines.extend(["", "## 5. 是否仍值得继续投资多卡方案"])
    section5 = report.get("section5", {})
    lines.append(f"- 结论: {section5.get('verdict', report.get('status', 'unknown'))}")
    lines.append(f"- 倾向拓扑: {section5.get('best_topology', 'unknown')}")
    lines.append(f"- 归因: {section5.get('attribution', 'n/a')}")
    if "a_mean_efficiency" in section5 or "b_mean_efficiency" in section5:
        lines.append(
            f"- mean_efficiency: A={section5.get('a_mean_efficiency', 'n/a')} "
            f"| B={section5.get('b_mean_efficiency', 'n/a')}"
        )
    return "\n".join(lines) + "\n"


def write_report(out_dir: Path, report: Dict[str, object]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "topology_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False)
    )
    (out_dir / "topology_report.md").write_text(build_report(report))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--server-binary", default="build_Release/bin/rapid_doc_npu_test_server")
    parser.add_argument("--lb-binary", default="build_Release/bin/rapid_doc_topology_lb")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, default=19880)
    parser.add_argument("--device-counts", default="1,2,3")
    parser.add_argument("--concurrency-list", default="2,3,6")
    parser.add_argument("--workloads", default=",".join(FORMAL_WORKLOADS))
    parser.add_argument("--warmup-waves", type=int, default=1)
    parser.add_argument("--measure-waves", type=int, default=5)
    parser.add_argument("--out-dir", default="output-benchmark/topology_attribution")
    parser.add_argument("--expected-main-commit", default=EXPECTED_MAIN_COMMIT)
    parser.add_argument("--expected-ocr-commit", default=EXPECTED_OCR_COMMIT)
    parser.add_argument("--skip-baseline-check", action="store_true")
    parser.add_argument("--allow-env-mismatch", action="store_true")
    args = parser.parse_args()

    args.project_root = Path(args.project_root).resolve()
    args.server_binary = (args.project_root / args.server_binary).resolve()
    args.lb_binary = (args.project_root / args.lb_binary).resolve()
    out_dir = (args.project_root / args.out_dir).resolve()
    args.request_output_root = out_dir / "request_outputs"
    args.device_counts = parse_csv_ints(args.device_counts)
    args.concurrency_list = parse_csv_ints(args.concurrency_list)
    args.workloads = parse_csv_strings(args.workloads)

    if not args.expected_main_commit:
        head_code, head_output = git_text(args.project_root, "rev-parse HEAD")
        if head_code == 0:
            args.expected_main_commit = head_output.strip()

    unknown_workloads = [name for name in args.workloads if name not in WORKLOAD_MANIFESTS]
    if unknown_workloads:
        raise SystemExit(f"Unknown workloads: {unknown_workloads}")

    env_snapshot = capture_env_snapshot()
    env_mismatches = expected_env_mismatches(env_snapshot)
    available_device_ids = detect_dxrt_devices()
    dxrt_snapshot = dxrt_status_snapshot()
    baseline = verify_baseline(args.project_root, args)
    if args.skip_baseline_check:
        baseline["status"] = "skipped"
        baseline["failures"] = []

    report: Dict[str, object] = {
        "generated_at": int(time.time()),
        "status": "ok",
        "baseline": baseline,
        "env_snapshot": env_snapshot,
        "env_mismatches": env_mismatches,
        "available_device_ids": available_device_ids,
        "available_device_count": len(available_device_ids),
        "dxrt_status": dxrt_snapshot,
        "preflight": {},
        "raw_results": [],
        "section1": [],
        "section2": [],
        "section3": [],
        "section4": [],
        "section5": {},
    }

    if baseline.get("status") not in ("ok", "skipped"):
        report["status"] = baseline["status"]
        report["section5"] = {
            "verdict": baseline["status"],
            "best_topology": "blocked",
            "attribution": "正式实验基线未恢复到已验收 checkpoint，停止开跑。",
        }
        write_report(out_dir, report)
        print(json.dumps({
            "json": str(out_dir / "topology_report.json"),
            "markdown": str(out_dir / "topology_report.md"),
            "status": report["status"],
        }, ensure_ascii=False))
        return 1

    if env_mismatches and not args.allow_env_mismatch:
        report["status"] = "blocked_env_profile_mismatch"
        report["section5"] = {
            "verdict": "blocked_env_profile_mismatch",
            "best_topology": "blocked",
            "attribution": "运行环境未对齐 set_env.sh 1 2 1 3 2 4，停止开跑。",
        }
        write_report(out_dir, report)
        print(json.dumps({
            "json": str(out_dir / "topology_report.json"),
            "markdown": str(out_dir / "topology_report.md"),
            "status": report["status"],
        }, ensure_ascii=False))
        return 1

    missing_binaries = [
        str(path)
        for path in (args.server_binary, args.lb_binary)
        if not path.exists()
    ]
    if missing_binaries:
        report["status"] = "blocked_binary_unavailable"
        report["section5"] = {
            "verdict": "blocked_binary_unavailable",
            "best_topology": "blocked",
            "attribution": "正式实验所需 server/LB 二进制不存在，需先完成 build 与 ctest。",
        }
        report["missing_binaries"] = missing_binaries
        write_report(out_dir, report)
        print(json.dumps({
            "json": str(out_dir / "topology_report.json"),
            "markdown": str(out_dir / "topology_report.md"),
            "status": report["status"],
        }, ensure_ascii=False))
        return 1

    if len(available_device_ids) < 2:
        report["status"] = "blocked_device_unavailable"
        report["section5"] = {
            "verdict": "blocked_device_unavailable",
            "best_topology": "blocked",
            "attribution": "目标主机 DXRT device 数不足 2，不能做正式多卡结论。",
        }
        write_report(out_dir, report)
        print(json.dumps({
            "json": str(out_dir / "topology_report.json"),
            "markdown": str(out_dir / "topology_report.md"),
            "status": report["status"],
        }, ensure_ascii=False))
        return 1

    first_workload = args.workloads[0]
    first_concurrency = args.concurrency_list[0]
    first_manifest = load_manifest(args.project_root / WORKLOAD_MANIFESTS[first_workload])
    preflight_a = run_profile(
        "A",
        available_device_ids[:1],
        first_workload,
        first_concurrency,
        args,
        None,
    )
    preflight_b_hashes: Optional[Dict[str, Dict[str, str]]] = None
    if preflight_a.get("status") == "ok":
        preflight_a_output = (
            args.request_output_root / "A" / first_workload / "1card" / f"c{first_concurrency}"
        )
        preflight_b_hashes = make_baseline_hashes_from_result(
            preflight_a,
            first_manifest,
            first_concurrency,
            preflight_a_output,
        )
    preflight_b = run_profile(
        "B",
        available_device_ids[:1],
        first_workload,
        first_concurrency,
        args,
        preflight_b_hashes,
    )
    report["preflight"] = {
        "workload": first_workload,
        "concurrency": first_concurrency,
        "A": preflight_a,
        "B": preflight_b,
    }
    if preflight_a.get("status") != "ok" or preflight_b.get("status") != "ok":
        report["status"] = "blocked_preflight_sanity_failed"
        report["raw_results"] = [preflight_a, preflight_b]
        report["section1"] = build_section1(args, report["raw_results"])
        report["section2"] = build_section2(report["raw_results"])
        report["section3"] = build_section3(report["raw_results"])
        report["section4"] = build_section4(report["raw_results"])
        report["section5"] = {
            "verdict": "blocked_preflight_sanity_failed",
            "best_topology": "blocked",
            "attribution": "1 卡 A/B sanity 未同时通过，停止正式矩阵。",
        }
        write_report(out_dir, report)
        print(json.dumps({
            "json": str(out_dir / "topology_report.json"),
            "markdown": str(out_dir / "topology_report.md"),
            "status": report["status"],
        }, ensure_ascii=False))
        return 1

    results: List[Dict[str, object]] = []
    cache: Dict[Tuple[str, str, int, int], Dict[str, object]] = {
        ("A", first_workload, first_concurrency, 1): preflight_a,
        ("B", first_workload, first_concurrency, 1): preflight_b,
    }

    for workload in args.workloads:
        manifest = load_manifest(args.project_root / WORKLOAD_MANIFESTS[workload])
        for concurrency in args.concurrency_list:
            baseline_hashes: Optional[Dict[str, Dict[str, str]]] = None
            a1_key = ("A", workload, concurrency, 1)
            if a1_key in cache:
                a1 = cache[a1_key]
            else:
                a1 = run_profile("A", available_device_ids[:1], workload, concurrency, args, None)
                cache[a1_key] = a1
            results.append(a1)

            if a1.get("status") == "ok":
                a1_output = args.request_output_root / "A" / workload / "1card" / f"c{concurrency}"
                baseline_hashes = make_baseline_hashes_from_result(
                    a1,
                    manifest,
                    concurrency,
                    a1_output,
                )
            b1_key = ("B", workload, concurrency, 1)
            if b1_key in cache:
                b1 = cache[b1_key]
            else:
                if baseline_hashes is None:
                    b1 = blocked_result(
                        "B",
                        workload,
                        concurrency,
                        1,
                        available_device_ids[:1],
                        "blocked_canonical_baseline_unavailable",
                        "1-card A baseline unavailable",
                    )
                else:
                    b1 = run_profile("B", available_device_ids[:1], workload, concurrency, args, baseline_hashes)
                cache[b1_key] = b1
            results.append(b1)

            for device_count in args.device_counts:
                if device_count == 1:
                    continue
                if device_count > len(available_device_ids):
                    results.append(blocked_result(
                        "A",
                        workload,
                        concurrency,
                        device_count,
                        available_device_ids[:device_count],
                        "blocked_device_unavailable",
                    ))
                    results.append(blocked_result(
                        "B",
                        workload,
                        concurrency,
                        device_count,
                        available_device_ids[:device_count],
                        "blocked_device_unavailable",
                    ))
                    continue

                device_ids = available_device_ids[:device_count]
                if baseline_hashes is None:
                    results.append(blocked_result(
                        "A",
                        workload,
                        concurrency,
                        device_count,
                        device_ids,
                        "blocked_canonical_baseline_unavailable",
                        "1-card A baseline unavailable",
                    ))
                    results.append(blocked_result(
                        "B",
                        workload,
                        concurrency,
                        device_count,
                        device_ids,
                        "blocked_canonical_baseline_unavailable",
                        "1-card A baseline unavailable",
                    ))
                    continue

                results.append(run_profile("A", device_ids, workload, concurrency, args, baseline_hashes))
                results.append(run_profile("B", device_ids, workload, concurrency, args, baseline_hashes))

    report["raw_results"] = results
    report["section1"] = build_section1(args, results)
    report["section2"] = build_section2(results)
    report["section3"] = build_section3(results)
    report["section4"] = build_section4(results)
    report["section5"] = build_section5(report["section3"])
    write_report(out_dir, report)
    print(json.dumps({
        "json": str(out_dir / "topology_report.json"),
        "markdown": str(out_dir / "topology_report.md"),
        "status": report["status"],
        "available_device_count": len(available_device_ids),
        "best_topology": report["section5"].get("best_topology"),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
