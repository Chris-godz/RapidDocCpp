#!/usr/bin/env python3
"""
Collect NPU-enabled concurrency observability for RapidDoc HTTP /file_parse.

Outputs per-request:
  - npu_lock_wait_ms
  - npu_lock_hold_ms
  - npu_serial_ms
  - cpu_only_ms
  - pipeline_call_ms
and /status snapshots after each case.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple


def run_shell(cmd: str) -> Tuple[int, str]:
    proc = subprocess.run(
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return proc.returncode, proc.stdout


def run_curl_with_status(curl_cmd: str) -> Dict[str, object]:
    code, output = run_shell(curl_cmd + " -w '\\n%{http_code}'")
    body = output
    http_code = 0
    if "\n" in output:
        body, code_text = output.rsplit("\n", 1)
        try:
            http_code = int(code_text.strip())
        except Exception:
            http_code = 0
    return {
        "exit_code": code,
        "http_code": http_code,
        "body": body,
        "raw_output": output,
    }


def build_file_parse_cmd(url: str, pdf_path: Path, fields: List[str], timeout_sec: int = 300) -> str:
    parts = [
        "curl",
        "-sS",
        f"--max-time {timeout_sec}",
        "-X POST",
        "-F",
        shlex.quote(f"files=@{pdf_path};type=application/pdf"),
    ]
    for field in fields:
        parts.extend(["-F", shlex.quote(field)])
    parts.append(shlex.quote(url))
    return " ".join(parts)


class ServerHandle:
    def __init__(self, binary: Path, host: str, port: int, extra_args: List[str]) -> None:
        self.binary = binary
        self.host = host
        self.port = port
        self.extra_args = extra_args
        self.proc: subprocess.Popen | None = None
        self.log_path = Path(tempfile.gettempdir()) / f"rapid_doc_npu_probe_{os.getpid()}_{port}.log"

    def start(self) -> None:
        with self.log_path.open("w") as log:
            self.proc = subprocess.Popen(
                [str(self.binary), "--host", self.host, "--port", str(self.port), *self.extra_args],
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

    def wait_ready(self, timeout_sec: int = 90) -> None:
        status_url = f"http://{self.host}:{self.port}/status"
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                log_tail = self.log_path.read_text(errors="ignore")[-2000:]
                raise RuntimeError(
                    f"server exited before ready, code={self.proc.returncode}, log_tail:\n{log_tail}"
                )
            res = run_curl_with_status(f"curl -sS --max-time 2 {shlex.quote(status_url)}")
            if res["exit_code"] == 0 and res["http_code"] == 200:
                return
            time.sleep(0.25)
        raise TimeoutError("timeout waiting for server readiness")

    def status(self) -> Dict[str, object]:
        status_url = f"http://{self.host}:{self.port}/status"
        res = run_curl_with_status(f"curl -sS --max-time 20 {shlex.quote(status_url)}")
        if res["exit_code"] != 0 or res["http_code"] != 200:
            raise RuntimeError(f"status request failed: {res}")
        payload = json.loads(res["body"])
        return payload


def parse_single_result(resp: Dict[str, object]) -> Dict[str, object]:
    if resp["exit_code"] != 0:
        raise RuntimeError(f"curl failed: {resp['raw_output']}")
    if resp["http_code"] != 200:
        raise RuntimeError(f"http failed: code={resp['http_code']}, body={resp['body']}")
    payload = json.loads(resp["body"])
    if "results" not in payload or not isinstance(payload["results"], list) or not payload["results"]:
        raise RuntimeError(f"unexpected payload: {payload}")
    result = payload["results"][0]
    stats = result.get("stats", {})
    needed = [
        "npu_lock_wait_ms",
        "npu_lock_hold_ms",
        "npu_serial_ms",
        "cpu_only_ms",
        "pipeline_call_ms",
        "pages",
    ]
    for key in needed:
        if key not in stats:
            raise RuntimeError(f"missing stats.{key}: {payload}")
    return {
        "output_dir": result.get("output_dir", ""),
        "pages": stats.get("pages"),
        "total_pages": stats.get("total_pages"),
        "npu_lock_wait_ms": stats.get("npu_lock_wait_ms"),
        "npu_lock_hold_ms": stats.get("npu_lock_hold_ms"),
        "npu_serial_ms": stats.get("npu_serial_ms"),
        "cpu_only_ms": stats.get("cpu_only_ms"),
        "pipeline_call_ms": stats.get("pipeline_call_ms"),
    }


def run_case_parallel(case_name: str, commands: List[str]) -> Dict[str, object]:
    start = time.time()
    with ThreadPoolExecutor(max_workers=len(commands)) as ex:
        futures = [ex.submit(run_curl_with_status, cmd) for cmd in commands]
        responses = [f.result() for f in futures]
    elapsed_ms = (time.time() - start) * 1000.0
    parsed = [parse_single_result(r) for r in responses]
    return {
        "name": case_name,
        "elapsed_ms": elapsed_ms,
        "requests": parsed,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parent.parent))
    parser.add_argument("--server-binary", default="build/bin/rapid_doc_npu_test_server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19880)
    parser.add_argument("--enable-table", action="store_true")
    parser.add_argument("--three-way", action="store_true")
    parser.add_argument("--out-json", default="")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    server_binary = (project_root / args.server_binary).resolve()
    if not server_binary.exists():
        raise SystemExit(f"server binary not found: {server_binary}")

    pdf_a = project_root / "test_files" / "small_ocr_origin.pdf"
    pdf_b = project_root / "test_files" / "BVRC_Meeting_Minutes_2024-04_origin.pdf"
    if not pdf_a.exists() or not pdf_b.exists():
        raise SystemExit(f"missing fixture PDFs: {pdf_a}, {pdf_b}")

    extra_args: List[str] = []
    if args.enable_table:
        extra_args.append("--enable-table")

    server = ServerHandle(server_binary, args.host, args.port, extra_args)
    result: Dict[str, object] = {
        "config": {
            "server_binary": str(server_binary),
            "host": args.host,
            "port": args.port,
            "enable_table": args.enable_table,
            "pdf_a": str(pdf_a),
            "pdf_b": str(pdf_b),
        },
        "cases": [],
    }

    try:
        server.start()
        server.wait_ready(timeout_sec=120)

        file_parse_url = f"http://{args.host}:{args.port}/file_parse"
        case2_cmds = [
            build_file_parse_cmd(
                file_parse_url,
                pdf_a,
                [
                    "return_content_list=true",
                    "clear_output_file=true",
                    "start_page_id=0",
                    "end_page_id=0",
                ],
            ),
            build_file_parse_cmd(
                file_parse_url,
                pdf_b,
                [
                    "return_content_list=true",
                    "clear_output_file=true",
                ],
            ),
        ]
        case2 = run_case_parallel("concurrency_2", case2_cmds)
        case2["status_snapshot"] = server.status()
        result["cases"].append(case2)

        if args.three_way:
            case3_cmds = [
                build_file_parse_cmd(
                    file_parse_url,
                    pdf_a,
                    [
                        "return_content_list=true",
                        "clear_output_file=true",
                        "start_page_id=0",
                        "end_page_id=0",
                    ],
                ),
                build_file_parse_cmd(
                    file_parse_url,
                    pdf_a,
                    [
                        "return_content_list=true",
                        "clear_output_file=true",
                        "start_page_id=1",
                        "end_page_id=1",
                    ],
                ),
                build_file_parse_cmd(
                    file_parse_url,
                    pdf_b,
                    [
                        "return_content_list=true",
                        "clear_output_file=true",
                        "start_page_id=0",
                        "end_page_id=0",
                    ],
                ),
            ]
            case3 = run_case_parallel("concurrency_3", case3_cmds)
            case3["status_snapshot"] = server.status()
            result["cases"].append(case3)

    finally:
        server.stop()
        result["server_log_tail"] = server.log_path.read_text(errors="ignore")[-2000:] if server.log_path.exists() else ""

    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out_json:
        Path(args.out_json).write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
