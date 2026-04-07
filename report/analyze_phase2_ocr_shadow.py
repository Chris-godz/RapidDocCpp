#!/usr/bin/env python3
"""
Phase2 OCR shadow analysis generator (analysis-only).

Consumes profiling benchmark JSON from previous commit and emits:
1) markdown report for docs/
2) optional machine-readable summary JSON
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_div(num: float, den: float) -> float:
    if den <= 0.0:
        return 0.0
    return num / den


def yes_no(flag: bool) -> str:
    return "Yes" if flag else "No"


def format_ratio(value: float) -> str:
    return f"{value:.4f}"


def format_ms(value: float) -> str:
    return f"{value:.3f}"


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_case(inprocess_data: Dict[str, Any], case_name: str) -> Dict[str, Any]:
    for case in inprocess_data.get("cases", []):
        if case.get("name") == case_name:
            return case
    raise KeyError(f"Missing case '{case_name}' in in-process benchmark JSON")


def pick_layout_ocr_run(http_verify: Dict[str, Any]) -> Dict[str, Any]:
    runs = http_verify.get("runs", [])
    if not runs:
        raise KeyError("verify HTTP JSON missing runs[]")

    # Prefer layout_ocr at c2 because it is the smallest standard load.
    for run in runs:
        if run.get("label") == "layout_ocr_c2":
            return run

    for run in runs:
        if run.get("scenario") == "layout_ocr":
            return run

    raise KeyError("verify HTTP JSON missing layout_ocr run")


@dataclass
class WorkloadProfile:
    workload_name: str
    inprocess_case_name: str
    total_time_ms: float
    pipeline_call_ms: float
    npu_serial_ms: float
    cpu_only_ms: float
    ocr_submit_count: float
    ocr_submit_area_p50: float
    ocr_submit_area_p95: float
    ocr_submit_small_count: float
    ocr_submit_large_count: float
    ocr_submit_text_count: float
    ocr_submit_title_count: float
    ocr_submit_code_count: float
    ocr_submit_list_count: float
    npu_ms_per_ocr_submit: float
    small_ratio: float
    large_ratio: float
    category_ratio_text: float
    category_ratio_title: float
    category_ratio_code: float
    category_ratio_list: float
    uncategorized_submit: float
    p95_p50_ratio: float
    auto_verdict: str
    is_boundary_case: bool
    manual_override: str
    override_reason: str
    final_verdict: str
    markdown_sha256: str
    content_list_sha256: str

    @staticmethod
    def from_case(
        workload_name: str,
        inprocess_case_name: str,
        case: Dict[str, Any],
        override_map: Dict[str, Dict[str, str]],
    ) -> "WorkloadProfile":
        mean_stats = case.get("mean_stats", {})
        stage = case.get("mean_stage_breakdown", {})

        ocr_submit_count = safe_float(stage.get("ocr_submit_count"))
        npu_serial_ms = safe_float(mean_stats.get("npu_serial_ms"))
        small_count = safe_float(stage.get("ocr_submit_small_count"))
        large_count = safe_float(stage.get("ocr_submit_large_count"))
        text_count = safe_float(stage.get("ocr_submit_text_count"))
        title_count = safe_float(stage.get("ocr_submit_title_count"))
        code_count = safe_float(stage.get("ocr_submit_code_count"))
        list_count = safe_float(stage.get("ocr_submit_list_count"))
        p50 = safe_float(stage.get("ocr_submit_area_p50"))
        p95 = safe_float(stage.get("ocr_submit_area_p95"))

        npu_per_submit = safe_div(npu_serial_ms, ocr_submit_count)
        small_ratio = safe_div(small_count, ocr_submit_count)
        large_ratio = safe_div(large_count, ocr_submit_count)
        ratio_text = safe_div(text_count, ocr_submit_count)
        ratio_title = safe_div(title_count, ocr_submit_count)
        ratio_code = safe_div(code_count, ocr_submit_count)
        ratio_list = safe_div(list_count, ocr_submit_count)
        uncategorized = max(0.0, ocr_submit_count - (text_count + title_count + code_count + list_count))
        p95_p50_ratio = safe_div(p95, p50)

        auto_yes = (
            small_ratio >= 0.60
            and large_ratio >= 0.10
            and p95_p50_ratio >= 8.0
        )
        auto_verdict = yes_no(auto_yes)

        boundary = (
            0.55 <= small_ratio <= 0.65
            or 0.08 <= large_ratio <= 0.12
            or 7.0 <= p95_p50_ratio <= 9.0
        )

        override = override_map.get(workload_name, {})
        manual_verdict = override.get("manual_verdict")
        manual_reason = override.get("reason", "").strip()
        if boundary and manual_verdict in {"Yes", "No"} and manual_reason:
            final_verdict = manual_verdict
            manual_override = "Yes"
            override_reason = manual_reason
        else:
            final_verdict = auto_verdict
            manual_override = "No"
            if boundary:
                override_reason = (
                    "边界 case；未提供有效人工覆盖，保持 auto_verdict。"
                )
            else:
                override_reason = "非边界 case；保持 auto_verdict。"

        return WorkloadProfile(
            workload_name=workload_name,
            inprocess_case_name=inprocess_case_name,
            total_time_ms=safe_float(mean_stats.get("time_ms")),
            pipeline_call_ms=safe_float(mean_stats.get("pipeline_call_ms")),
            npu_serial_ms=npu_serial_ms,
            cpu_only_ms=safe_float(mean_stats.get("cpu_only_ms")),
            ocr_submit_count=ocr_submit_count,
            ocr_submit_area_p50=p50,
            ocr_submit_area_p95=p95,
            ocr_submit_small_count=small_count,
            ocr_submit_large_count=large_count,
            ocr_submit_text_count=text_count,
            ocr_submit_title_count=title_count,
            ocr_submit_code_count=code_count,
            ocr_submit_list_count=list_count,
            npu_ms_per_ocr_submit=npu_per_submit,
            small_ratio=small_ratio,
            large_ratio=large_ratio,
            category_ratio_text=ratio_text,
            category_ratio_title=ratio_title,
            category_ratio_code=ratio_code,
            category_ratio_list=ratio_list,
            uncategorized_submit=uncategorized,
            p95_p50_ratio=p95_p50_ratio,
            auto_verdict=auto_verdict,
            is_boundary_case=boundary,
            manual_override=manual_override,
            override_reason=override_reason,
            final_verdict=final_verdict,
            markdown_sha256=str(case.get("markdown_sha256", "")),
            content_list_sha256=str(case.get("content_list_sha256", "")),
        )


def parse_override(path: Optional[Path]) -> Dict[str, Dict[str, str]]:
    if path is None:
        return {}
    payload = load_json(path)
    if not isinstance(payload, dict):
        raise ValueError("override JSON must be an object")
    normalized: Dict[str, Dict[str, str]] = {}
    for key, value in payload.items():
        if not isinstance(value, dict):
            continue
        normalized[str(key)] = {
            "manual_verdict": str(value.get("manual_verdict", "")).strip(),
            "reason": str(value.get("reason", "")).strip(),
        }
    return normalized


def build_special_explanations(ocr_only: WorkloadProfile, layout: WorkloadProfile) -> Tuple[str, str]:
    ocr_only_big_box = (
        ocr_only.large_ratio >= 0.50
        and ocr_only.small_ratio < 0.40
        and ocr_only.p95_p50_ratio < 6.0
    )
    ocr_only_text = (
        f"`ocr_only` {'是' if ocr_only_big_box else '不是'}大框主导型 workload。"
        f"依据：large_ratio={format_ratio(ocr_only.large_ratio)}，"
        f"small_ratio={format_ratio(ocr_only.small_ratio)}，"
        f"p95/p50={format_ratio(ocr_only.p95_p50_ratio)}。"
    )

    layout_mixed = (
        layout.small_ratio >= 0.60
        and layout.large_ratio >= 0.10
        and layout.p95_p50_ratio >= 8.0
    )
    layout_text = (
        f"`layout_ocr_table` {'是' if layout_mixed else '不是'}碎片化 + 大框尾部混合型 workload。"
        f"依据：small_ratio={format_ratio(layout.small_ratio)}，"
        f"large_ratio={format_ratio(layout.large_ratio)}，"
        f"p95/p50={format_ratio(layout.p95_p50_ratio)}。"
    )
    return ocr_only_text, layout_text


def markdown_table_row(profile: WorkloadProfile) -> str:
    return (
        f"| {profile.workload_name} "
        f"| {profile.ocr_submit_count:.0f} "
        f"| {format_ms(profile.npu_serial_ms)} "
        f"| {format_ms(profile.npu_ms_per_ocr_submit)} "
        f"| {format_ratio(profile.small_ratio)} "
        f"| {format_ratio(profile.large_ratio)} "
        f"| {format_ratio(profile.p95_p50_ratio)} "
        f"| {format_ratio(profile.category_ratio_text)}/{format_ratio(profile.category_ratio_title)}/"
        f"{format_ratio(profile.category_ratio_code)}/{format_ratio(profile.category_ratio_list)} "
        f"| {profile.uncategorized_submit:.0f} "
        f"| {profile.auto_verdict} "
        f"| {profile.manual_override} "
        f"| {profile.override_reason} "
        f"| {profile.final_verdict} |"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate Phase2 OCR shadow analysis report")
    parser.add_argument(
        "--inprocess",
        default="output-benchmark/phase2_profile_after_inprocess.json",
        help="Primary in-process profiling benchmark JSON",
    )
    parser.add_argument(
        "--http",
        default="report/phase2_profile_http_single.json",
        help="Primary HTTP profiling JSON (single-sample format)",
    )
    parser.add_argument(
        "--verify-inprocess",
        default="output-benchmark/phase2_shadow_analysis_verify_inprocess.json",
        help="Verification in-process benchmark JSON",
    )
    parser.add_argument(
        "--verify-http",
        default="report/phase2_shadow_analysis_verify_http.json",
        help="Verification HTTP benchmark JSON produced by run_multinpu_bench.py",
    )
    parser.add_argument(
        "--override-json",
        default=None,
        help="Optional boundary override JSON",
    )
    parser.add_argument(
        "--output-md",
        default="docs/perf_phase2_ocr_shadow_analysis.md",
        help="Output markdown report",
    )
    parser.add_argument(
        "--output-json",
        default="report/phase2_ocr_shadow_analysis_summary.json",
        help="Output machine-readable summary",
    )
    args = parser.parse_args()

    inprocess_path = Path(args.inprocess)
    http_path = Path(args.http)
    verify_inprocess_path = Path(args.verify_inprocess)
    verify_http_path = Path(args.verify_http)
    output_md_path = Path(args.output_md)
    output_json_path = Path(args.output_json)
    override_path = Path(args.override_json) if args.override_json else None

    inprocess_data = load_json(inprocess_path)
    http_data = load_json(http_path)
    overrides = parse_override(override_path)

    ocr_only_case = get_case(inprocess_data, "ocr_only")
    layout_case = get_case(inprocess_data, "layout_ocr_table_full_chain")

    ocr_only = WorkloadProfile.from_case(
        "ocr_only",
        "ocr_only",
        ocr_only_case,
        overrides,
    )
    layout = WorkloadProfile.from_case(
        "layout_ocr_table",
        "layout_ocr_table_full_chain",
        layout_case,
        overrides,
    )

    ocr_only_explain, layout_explain = build_special_explanations(ocr_only, layout)

    verify_inprocess = load_json(verify_inprocess_path)
    verify_http = load_json(verify_http_path)

    verify_ocr_case = get_case(verify_inprocess, "ocr_only")
    verify_layout_case = get_case(verify_inprocess, "layout_ocr_table_full_chain")

    verify_ocr_stats = verify_ocr_case.get("mean_stats", {})
    verify_layout_stats = verify_layout_case.get("mean_stats", {})

    verify_layout_ocr_run = pick_layout_ocr_run(verify_http)
    verify_http_time = safe_float(verify_layout_ocr_run.get("mean_time_ms"))
    if verify_http_time <= 0.0:
        # run_multinpu_bench.py does not emit mean_time_ms; pipeline_call_ms is the closest
        # per-request wall-clock metric under the same HTTP route.
        verify_http_time = safe_float(verify_layout_ocr_run.get("mean_pipeline_call_ms"))
    verify_http_metrics = {
        "time_ms": verify_http_time,
        "pipeline_call_ms": safe_float(verify_layout_ocr_run.get("mean_pipeline_call_ms")),
        "npu_serial_ms": safe_float(verify_layout_ocr_run.get("mean_npu_serial_ms")),
        "cpu_only_ms": safe_float(verify_layout_ocr_run.get("mean_cpu_only_ms")),
    }

    main_http_metrics = {
        "time_ms": safe_float(http_data.get("time_ms")),
        "pipeline_call_ms": safe_float(http_data.get("pipeline_call_ms")),
        "npu_serial_ms": safe_float(http_data.get("npu_serial_ms")),
        "cpu_only_ms": safe_float(http_data.get("cpu_only_ms")),
    }

    verify_md_hash = str(verify_layout_ocr_run.get("warmup_md_sha256", ""))
    verify_middle_hash = str(verify_layout_ocr_run.get("warmup_middle_json_sha256", ""))
    main_md_hash = str(http_data.get("markdown_sha256", ""))
    main_middle_hash = str(http_data.get("middle_json_sha256", ""))

    final_sentence = (
        f"下一枪建议：{('进入' if layout.final_verdict == 'Yes' else '暂不进入')} "
        "shadow packing/merge 候选试验，"
        f"范围限定为 `layout_ocr_table`（`ocr_only` 判定为 {ocr_only.final_verdict}）。"
    )

    generated_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    md_lines = [
        "# Phase2 OCR Profile Shadow Analysis Report",
        "",
        f"- 生成时间：`{generated_time}`",
        "- 提交类型：`analysis-only`（不改推理逻辑）",
        f"- 主分析输入(in-process)：`{inprocess_path}`",
        f"- 主分析输入(HTTP)：`{http_path}`",
        f"- 回归验证输入(in-process)：`{verify_inprocess_path}`",
        f"- 回归验证输入(HTTP)：`{verify_http_path}`",
        "",
        "## 1. 固定派生指标定义",
        "",
        "- `npu_ms_per_ocr_submit = npu_serial_ms / ocr_submit_count`",
        "- `small_ratio = ocr_submit_small_count / ocr_submit_count`",
        "- `large_ratio = ocr_submit_large_count / ocr_submit_count`",
        "- `category_ratio_text/title/code/list = 对应 count / ocr_submit_count`",
        "- `uncategorized_submit = ocr_submit_count - (text + title + code + list)`（下限 0）",
        "",
        "## 2. Workload 画像与判定",
        "",
        "| workload | ocr_submit_count | npu_serial_ms | npu_ms_per_ocr_submit | small_ratio | large_ratio | p95/p50 | category_ratio text/title/code/list | uncategorized_submit | auto_verdict | manual_override | override_reason | final_verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|---|",
        markdown_table_row(ocr_only),
        markdown_table_row(layout),
        "",
        "自动规则：`Yes` 当且仅当 `small_ratio>=0.60 && large_ratio>=0.10 && p95/p50>=8`；否则 `No`。",
        "边界区间：`small_ratio∈[0.55,0.65]` 或 `large_ratio∈[0.08,0.12]` 或 `p95/p50∈[7,9]`，可人工 override 并必须写理由。",
        "",
        "## 3. 专项解释",
        "",
        f"- {ocr_only_explain}",
        f"- {layout_explain}",
        "",
        "## 4. Analysis-only 回归验证",
        "",
        "### 4.1 in-process hash 一致性",
        "",
        "| workload | main markdown_sha256 | verify markdown_sha256 | markdown same | main content_list_sha256 | verify content_list_sha256 | content_list same |",
        "|---|---|---|---|---|---|---|",
        (
            f"| ocr_only | {ocr_only.markdown_sha256} | {verify_ocr_case.get('markdown_sha256', '')} "
            f"| {yes_no(ocr_only.markdown_sha256 == str(verify_ocr_case.get('markdown_sha256', '')))} "
            f"| {ocr_only.content_list_sha256} | {verify_ocr_case.get('content_list_sha256', '')} "
            f"| {yes_no(ocr_only.content_list_sha256 == str(verify_ocr_case.get('content_list_sha256', '')))} |"
        ),
        (
            f"| layout_ocr_table | {layout.markdown_sha256} | {verify_layout_case.get('markdown_sha256', '')} "
            f"| {yes_no(layout.markdown_sha256 == str(verify_layout_case.get('markdown_sha256', '')))} "
            f"| {layout.content_list_sha256} | {verify_layout_case.get('content_list_sha256', '')} "
            f"| {yes_no(layout.content_list_sha256 == str(verify_layout_case.get('content_list_sha256', '')))} |"
        ),
        "",
        "### 4.2 HTTP warmup hash 一致性（layout_ocr）",
        "",
        f"- `main md_sha256`: `{main_md_hash}`",
        f"- `verify warmup md_sha256`: `{verify_md_hash}`",
        f"- `md same`: `{yes_no(main_md_hash == verify_md_hash)}`",
        f"- `main middle_json_sha256`: `{main_middle_hash}`",
        f"- `verify warmup middle_json_sha256`: `{verify_middle_hash}`",
        f"- `middle_json same`: `{yes_no(main_middle_hash == verify_middle_hash)}`",
        "",
        "### 4.3 指标展示（仅展示，不作性能门禁）",
        "",
        "| scope | workload | total_time_ms(main/verify) | pipeline_call_ms(main/verify) | npu_serial_ms(main/verify) | cpu_only_ms(main/verify) |",
        "|---|---|---:|---:|---:|---:|",
        (
            f"| in-process | ocr_only | {format_ms(ocr_only.total_time_ms)}/{format_ms(safe_float(verify_ocr_stats.get('time_ms')))} "
            f"| {format_ms(ocr_only.pipeline_call_ms)}/{format_ms(safe_float(verify_ocr_stats.get('pipeline_call_ms')))} "
            f"| {format_ms(ocr_only.npu_serial_ms)}/{format_ms(safe_float(verify_ocr_stats.get('npu_serial_ms')))} "
            f"| {format_ms(ocr_only.cpu_only_ms)}/{format_ms(safe_float(verify_ocr_stats.get('cpu_only_ms')))} |"
        ),
        (
            f"| in-process | layout_ocr_table | {format_ms(layout.total_time_ms)}/{format_ms(safe_float(verify_layout_stats.get('time_ms')))} "
            f"| {format_ms(layout.pipeline_call_ms)}/{format_ms(safe_float(verify_layout_stats.get('pipeline_call_ms')))} "
            f"| {format_ms(layout.npu_serial_ms)}/{format_ms(safe_float(verify_layout_stats.get('npu_serial_ms')))} "
            f"| {format_ms(layout.cpu_only_ms)}/{format_ms(safe_float(verify_layout_stats.get('cpu_only_ms')))} |"
        ),
        (
            f"| HTTP | layout_ocr | {format_ms(main_http_metrics['time_ms'])}/{format_ms(verify_http_metrics['time_ms'])} "
            f"| {format_ms(main_http_metrics['pipeline_call_ms'])}/{format_ms(verify_http_metrics['pipeline_call_ms'])} "
            f"| {format_ms(main_http_metrics['npu_serial_ms'])}/{format_ms(verify_http_metrics['npu_serial_ms'])} "
            f"| {format_ms(main_http_metrics['cpu_only_ms'])}/{format_ms(verify_http_metrics['cpu_only_ms'])} |"
        ),
        "",
        "## 5. 下一枪建议（Yes/No）",
        "",
        f"- `ocr_only`: `{ocr_only.final_verdict}`",
        f"- `layout_ocr_table`: `{layout.final_verdict}`",
        "",
        f"结论：{final_sentence}",
        "",
    ]

    output_md_path.parent.mkdir(parents=True, exist_ok=True)
    output_md_path.write_text("\n".join(md_lines), encoding="utf-8")

    summary = {
        "generated_at": generated_time,
        "inputs": {
            "inprocess": str(inprocess_path),
            "http": str(http_path),
            "verify_inprocess": str(verify_inprocess_path),
            "verify_http": str(verify_http_path),
            "override_json": str(override_path) if override_path else None,
        },
        "workloads": {
            "ocr_only": ocr_only.__dict__,
            "layout_ocr_table": layout.__dict__,
        },
        "hash_validation": {
            "inprocess_ocr_only_markdown_same": (
                ocr_only.markdown_sha256 == str(verify_ocr_case.get("markdown_sha256", ""))
            ),
            "inprocess_ocr_only_content_same": (
                ocr_only.content_list_sha256 == str(verify_ocr_case.get("content_list_sha256", ""))
            ),
            "inprocess_layout_markdown_same": (
                layout.markdown_sha256 == str(verify_layout_case.get("markdown_sha256", ""))
            ),
            "inprocess_layout_content_same": (
                layout.content_list_sha256 == str(verify_layout_case.get("content_list_sha256", ""))
            ),
            "http_layout_ocr_md_same": (main_md_hash == verify_md_hash),
            "http_layout_ocr_middle_same": (main_middle_hash == verify_middle_hash),
        },
        "final_sentence": final_sentence,
    }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote markdown report: {output_md_path}")
    print(f"Wrote summary JSON: {output_json_path}")
    print(final_sentence)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
