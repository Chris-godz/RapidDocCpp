#!/usr/bin/env python3
"""Build a 12-row mismatch ledger by joining candidate_mismatch_review,
formula_trace (layout_raw_boxes / layout_prefilter_boxes / filtered_candidates),
and the crop-parity A/B replay summary.

Output:
  <out-dir>/ledger.json
  <out-dir>/ledger.md

Classification rules (strict, 4 buckets):
  1. py_idx in mismatch_indices_A  -> same_crop_model_or_decode_gap
  2. best_raw_iou - best_prefilter_iou > 0.05 -> raw_has_better_box_but_thresholded_out
  3. best_prefilter_iou - iou_final     > 0.05 -> layout_postprocess_selects_worse_box
  4. otherwise -> layout_raw_box_granularity_gap

Granularity sub-label:
  - best_raw_iou < 0.80 -> hard-blocker-for-current-pipeline
  - else                -> needs-box-semantics-change
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


FORMULA_CATEGORY_ID = 7  # equation / formula
IOU_DELTA_THRESHOLD = 0.05
GRANULARITY_HARD_IOU = 0.80


_WS_RE = re.compile(r"\s+")


def normalize_latex(s: Optional[str]) -> str:
    return _WS_RE.sub("", s or "")


def iou_xyxy(a: List[float], b: List[float]) -> float:
    ax0, ay0, ax1, ay1 = a[0], a[1], a[2], a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[2], b[3]
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    iw = max(0.0, ix1 - ix0)
    ih = max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, ax1 - ax0) * max(0.0, ay1 - ay0)
    area_b = max(0.0, bx1 - bx0) * max(0.0, by1 - by0)
    denom = area_a + area_b - inter
    if denom <= 0.0:
        return 0.0
    return inter / denom


def best_match(
    py_bbox: List[float],
    boxes: List[Dict[str, Any]],
    index_key: str,
) -> Tuple[float, Optional[Dict[str, Any]]]:
    best_iou = 0.0
    best = None
    for b in boxes:
        if b.get("category_id") != FORMULA_CATEGORY_ID:
            continue
        iou = iou_xyxy(py_bbox, b["bbox"])
        if iou > best_iou:
            best_iou = iou
            best = b
    return best_iou, best


def classify(
    py_idx: int,
    mismatch_a: set,
    best_raw_iou: float,
    best_prefilter_iou: float,
    iou_final: float,
) -> Tuple[str, bool, str, str]:
    if py_idx in mismatch_a:
        return (
            "same_crop_model_or_decode_gap",
            False,
            "normalize_or_decode",
            "fails on Python crops; not a layout issue",
        )
    gap_threshold = best_raw_iou - best_prefilter_iou
    gap_postprocess = best_prefilter_iou - iou_final
    if gap_threshold > IOU_DELTA_THRESHOLD or gap_postprocess > IOU_DELTA_THRESHOLD:
        if gap_postprocess >= gap_threshold and gap_postprocess > IOU_DELTA_THRESHOLD:
            return (
                "layout_postprocess_selects_worse_box",
                True,
                "bbox_selection",
                (
                    f"prefilter IoU={best_prefilter_iou:.3f} > final={iou_final:.3f} "
                    f"(gap={gap_postprocess:.3f}); raw gap={gap_threshold:.3f}"
                ),
            )
        return (
            "raw_has_better_box_but_thresholded_out",
            True,
            "threshold",
            (
                f"raw IoU={best_raw_iou:.3f} > prefilter={best_prefilter_iou:.3f} "
                f"(gap={gap_threshold:.3f}); post gap={gap_postprocess:.3f}"
            ),
        )
    sub = (
        "hard-blocker-for-current-pipeline"
        if best_raw_iou < GRANULARITY_HARD_IOU
        else "needs-box-semantics-change"
    )
    return (
        "layout_raw_box_granularity_gap",
        False,
        sub,
        f"raw IoU ceiling={best_raw_iou:.3f}; no better box anywhere upstream",
    )


def geometry_profile(bbox: List[float]) -> Dict[str, float]:
    x0, y0, x1, y1 = bbox
    w = x1 - x0
    h = y1 - y0
    return {"x0": x0, "y0": y0, "x1": x1, "y1": y1, "w": w, "h": h}


def bbox_pixel_delta(py: List[float], cpp: List[float]) -> Dict[str, float]:
    return {
        "dx0": cpp[0] - py[0],
        "dy0": cpp[1] - py[1],
        "dx1": cpp[2] - py[2],
        "dy1": cpp[3] - py[3],
    }


def find_refinement_for_candidate(
    refinements: List[Dict[str, Any]],
    refined_index: int,
) -> Optional[Dict[str, Any]]:
    if refined_index < 0 or refined_index >= len(refinements):
        return None
    return refinements[refined_index]


def build_ledger(
    trace: Dict[str, Any],
    review: Dict[str, Any],
    crop_parity: Dict[str, Any],
) -> Dict[str, Any]:
    raw_boxes = trace["layout_raw_boxes"]
    prefilter_boxes = trace["layout_prefilter_boxes"]
    filtered_candidates = trace["filtered_candidates"]
    refinements = trace.get("candidate_refinements", [])

    matches_by_py = {m["python_index"]: m for m in review["matches"]}
    mismatch_a = set(crop_parity.get("mismatch_indices_A", []))
    mismatch_b = list(crop_parity.get("mismatch_indices_B", []))

    rows: List[Dict[str, Any]] = []
    for py_idx in sorted(mismatch_b):
        m = matches_by_py.get(py_idx)
        if m is None:
            rows.append({
                "python_index": py_idx,
                "error": "not found in candidate_mismatch_review.matches",
            })
            continue

        py_bbox = m["python_bbox"]
        cpp_bbox = m["cpp_bbox"]
        iou_final = m["iou"]

        best_raw_iou, best_raw = best_match(py_bbox, raw_boxes, "layout_raw_index")
        best_pre_iou, best_pre = best_match(py_bbox, prefilter_boxes, "layout_prefilter_index")

        fc_idx = m.get("filtered_candidate_index")
        fc = filtered_candidates[fc_idx] if fc_idx is not None else None
        refined_idx = fc.get("refined_candidate_index") if fc else None
        refinement = (
            find_refinement_for_candidate(refinements, refined_idx)
            if refined_idx is not None
            else None
        )

        category, fixable, fix_layer, reason = classify(
            py_idx, mismatch_a, best_raw_iou, best_pre_iou, iou_final
        )

        row = {
            "python_index": py_idx,
            "a_same_crop_pass": py_idx not in mismatch_a,
            "python_bbox": py_bbox,
            "python_bbox_profile": geometry_profile(py_bbox),
            "python_latex": m["python_latex"],
            "cpp_final_bbox": cpp_bbox,
            "cpp_final_latex": m["cpp_latex"],
            "iou_final": iou_final,
            "best_raw_iou": best_raw_iou,
            "best_raw_bbox": best_raw["bbox"] if best_raw else None,
            "best_raw_score": best_raw["confidence"] if best_raw else None,
            "best_raw_threshold": best_raw["confidence_threshold"] if best_raw else None,
            "best_raw_passed_threshold": best_raw["passed_confidence_threshold"] if best_raw else None,
            "best_raw_index": best_raw["layout_raw_index"] if best_raw else None,
            "best_prefilter_iou": best_pre_iou,
            "best_prefilter_bbox": best_pre["bbox"] if best_pre else None,
            "best_prefilter_score": best_pre["confidence"] if best_pre else None,
            "filtered_candidate_index": fc_idx,
            "refined_candidate_action": refinement.get("action") if refinement else None,
            "refined_candidate_sources": refinement.get("source_raw_candidate_indices") if refinement else None,
            "bbox_px_delta_py_to_cpp": bbox_pixel_delta(py_bbox, cpp_bbox),
            "category": category,
            "fixable_this_round": fixable,
            "fix_layer": fix_layer,
            "reason": reason,
            "normalized_python_latex": normalize_latex(m["python_latex"]),
            "normalized_cpp_latex": normalize_latex(m["cpp_latex"]),
        }
        rows.append(row)

    counts: Dict[str, int] = {}
    for r in rows:
        counts[r.get("category", "unknown")] = counts.get(r.get("category", "unknown"), 0) + 1

    expected_counts = {
        "layout_raw_box_granularity_gap": 7,
        "raw_has_better_box_but_thresholded_out": 2,
        "layout_postprocess_selects_worse_box": 1,
        "same_crop_model_or_decode_gap": 2,
    }
    delta = {k: counts.get(k, 0) - v for k, v in expected_counts.items()}

    return {
        "normalization_spec": "re.sub(r'\\s+', '', s or '')",
        "inputs": {
            "formula_trace": trace.get("_source_path"),
            "candidate_mismatch_review": review.get("_source_path"),
            "crop_parity_summary": crop_parity.get("_source_path"),
        },
        "classification_rules": {
            "iou_delta_threshold": IOU_DELTA_THRESHOLD,
            "granularity_hard_iou": GRANULARITY_HARD_IOU,
        },
        "expected_counts": expected_counts,
        "actual_counts": counts,
        "delta_vs_expected": delta,
        "rows": rows,
    }


def render_markdown(ledger: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# 12-Mismatch Root-Cause Ledger")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append("| Category | Expected | Actual | Delta |")
    lines.append("|---|---:|---:|---:|")
    for k, exp in ledger["expected_counts"].items():
        act = ledger["actual_counts"].get(k, 0)
        lines.append(f"| `{k}` | {exp} | {act} | {act - exp:+d} |")
    lines.append("")
    lines.append("## Rows")
    lines.append("")
    lines.append(
        "| py | A-pass | iou_final | best_pre | best_raw | raw_passed | score | thr | category | fixable | fix_layer |"
    )
    lines.append(
        "|---:|:---:|---:|---:|---:|:---:|---:|---:|---|:---:|---|"
    )
    for r in ledger["rows"]:
        lines.append(
            "| {py} | {a} | {iouf:.3f} | {pre:.3f} | {raw:.3f} | {passed} | {score:.3f} | {thr:.3f} | `{cat}` | {fix} | {layer} |".format(
                py=r["python_index"],
                a="Y" if r.get("a_same_crop_pass") else "N",
                iouf=r.get("iou_final", 0.0) or 0.0,
                pre=r.get("best_prefilter_iou", 0.0) or 0.0,
                raw=r.get("best_raw_iou", 0.0) or 0.0,
                passed="Y" if r.get("best_raw_passed_threshold") else "N",
                score=r.get("best_raw_score", 0.0) or 0.0,
                thr=r.get("best_raw_threshold", 0.0) or 0.0,
                cat=r.get("category"),
                fix="Y" if r.get("fixable_this_round") else "N",
                layer=r.get("fix_layer"),
            )
        )
    lines.append("")
    lines.append("## Per-row detail")
    lines.append("")
    for r in ledger["rows"]:
        lines.append(f"### py_idx = {r['python_index']}  ({r['category']})")
        lines.append("")
        lines.append(f"- reason: {r.get('reason')}")
        lines.append(f"- a_same_crop_pass: {r.get('a_same_crop_pass')}")
        lines.append(f"- python_bbox: {r['python_bbox']}")
        lines.append(f"- cpp_final_bbox: {r['cpp_final_bbox']}")
        lines.append(
            f"- best_raw: iou={r['best_raw_iou']:.4f} score={r['best_raw_score']:.3f} "
            f"thr={r['best_raw_threshold']:.3f} passed={r['best_raw_passed_threshold']} "
            f"bbox={r['best_raw_bbox']} raw_idx={r['best_raw_index']}"
        )
        lines.append(
            f"- best_prefilter: iou={r['best_prefilter_iou']:.4f} "
            f"bbox={r['best_prefilter_bbox']}"
        )
        lines.append(
            f"- filtered_candidate_index={r['filtered_candidate_index']} "
            f"refinement_action={r['refined_candidate_action']} "
            f"sources={r['refined_candidate_sources']}"
        )
        lines.append(f"- bbox_px_delta_py_to_cpp: {r['bbox_px_delta_py_to_cpp']}")
        lines.append(f"- python_latex: `{r['python_latex']}`")
        lines.append(f"- cpp_latex:    `{r['cpp_final_latex']}`")
        lines.append(f"- normalized python: `{r['normalized_python_latex']}`")
        lines.append(f"- normalized cpp:    `{r['normalized_cpp_latex']}`")
        lines.append(f"- fixable_this_round: {r['fixable_this_round']} (fix_layer={r['fix_layer']})")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formula-trace", required=True)
    ap.add_argument("--candidate-review", required=True)
    ap.add_argument("--crop-parity-summary", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    trace = json.loads(Path(args.formula_trace).read_text())
    trace["_source_path"] = str(Path(args.formula_trace).resolve())
    review = json.loads(Path(args.candidate_review).read_text())
    review["_source_path"] = str(Path(args.candidate_review).resolve())
    crop_parity = json.loads(Path(args.crop_parity_summary).read_text())
    crop_parity["_source_path"] = str(Path(args.crop_parity_summary).resolve())

    ledger = build_ledger(trace, review, crop_parity)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "ledger.json").write_text(json.dumps(ledger, indent=2, ensure_ascii=False))
    (out_dir / "ledger.md").write_text(render_markdown(ledger))

    print(json.dumps({
        "out_dir": str(out_dir.resolve()),
        "actual_counts": ledger["actual_counts"],
        "delta_vs_expected": ledger["delta_vs_expected"],
    }, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
