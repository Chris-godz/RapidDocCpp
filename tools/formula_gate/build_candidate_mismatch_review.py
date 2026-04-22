#!/usr/bin/env python3
"""
Build Gate 1 evidence: candidate_mismatch_review.json from Python *_model.json
and C++ *_formula_trace.json (RAPIDDOC_FORMULA_TRACE=1).

Python formula truth: layout_dets with category_id==13 and 'latex' (same as
output-benchmark/.../build_same_crop_summary.py).

Normalized LaTeX (Gate 1 + Gate 2 parity with gate2_median_summary):
  re.sub(r'\\s+', '', s) applied to both strings before compare.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def bbox_from_poly(poly: List[float]) -> List[float]:
    xs = poly[0::2]
    ys = poly[1::2]
    return [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]


def area(b: List[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def iou(a: List[float], b: List[float]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    den = area(a) + area(b) - inter
    return inter / den if den > 0 else 0.0


def normalize_latex(s: str) -> str:
    """Aligned with historical gate2_median_summary / same-crop tooling."""
    return re.sub(r"\s+", "", s or "")


def is_formula_raw_entry(d: Dict[str, Any]) -> bool:
    lab = d.get("label")
    cat = d.get("category")
    return lab == "formula" or cat in ("equation", "interline_equation")


def python_formula_candidates(model_page: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, d in enumerate(model_page.get("layout_dets", [])):
        if d.get("category_id") == 13 and "latex" in d:
            bbox = d.get("bbox") or bbox_from_poly(d["poly"])
            out.append(
                {
                    "python_index": len(out),
                    "source_layout_det_index": idx,
                    "bbox": [float(x) for x in bbox],
                    "score": d.get("score"),
                    "latex": d.get("latex", ""),
                }
            )
    return out


def cpp_refined_kept(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for d in trace.get("filtered_candidates", []):
        if d.get("kept"):
            kept.append(d)
    return kept


def latex_by_filtered_index(trace: Dict[str, Any]) -> Dict[int, str]:
    m: Dict[int, str] = {}
    for d in trace.get("crop_outputs", []):
        idx = d.get("filtered_candidate_index")
        if idx is not None:
            m[int(idx)] = d.get("latex", "")
    return m


def best_iou_and_entry(
    bbox: List[float], entries: List[Dict[str, Any]], pred
) -> Tuple[float, Optional[Dict[str, Any]], Optional[int]]:
    best = 0.0
    best_e: Optional[Dict[str, Any]] = None
    best_i: Optional[int] = None
    for i, e in enumerate(entries):
        if not pred(e):
            continue
        bb = e.get("bbox")
        if not bb or len(bb) < 4:
            continue
        sc = iou(bbox, [float(x) for x in bb])
        if sc > best:
            best = sc
            best_e = e
            best_i = i
    return best, best_e, best_i


def classify_python_unmatched(
    bbox: List[float],
    page: int,
    trace: Dict[str, Any],
    iou_thr: float,
) -> Dict[str, Any]:
    """Layer diagnosis for one Python truth box (typically unmatched)."""

    def same_page(e: Dict[str, Any]) -> bool:
        return int(e.get("page", -1)) == int(page)

    raw_all = [x for x in trace.get("layout_raw_boxes", []) if same_page(x)]
    pre_all = [x for x in trace.get("layout_prefilter_boxes", []) if same_page(x)]
    layout_all = [x for x in trace.get("layout_boxes", []) if same_page(x)]

    raw_formula = [x for x in raw_all if is_formula_raw_entry(x)]
    best_f, hit_f, idx_f = best_iou_and_entry(bbox, raw_formula, lambda _: True)
    best_any, hit_any, idx_any = best_iou_and_entry(bbox, raw_all, lambda _: True)

    evidence: Dict[str, Any] = {
        "page": page,
        "bbox": bbox,
        "best_iou_formula_raw": best_f,
        "best_iou_any_raw": best_any,
    }

    if hit_f is not None:
        evidence["best_formula_raw_hit"] = {
            "layout_raw_index": idx_f,
            "bbox": hit_f.get("bbox"),
            "confidence": hit_f.get("confidence"),
            "confidence_threshold": hit_f.get("confidence_threshold"),
            "passed_confidence_threshold": hit_f.get("passed_confidence_threshold"),
            "label": hit_f.get("label"),
            "category": hit_f.get("category"),
        }

    # Prefilter / layout presence (IoU with same-stage boxes)
    best_pre, hit_pre, _ = best_iou_and_entry(bbox, pre_all, lambda _: True)
    best_lay, hit_lay, _ = best_iou_and_entry(bbox, layout_all, lambda _: True)
    evidence["best_iou_prefilter"] = best_pre
    evidence["best_iou_layout_box"] = best_lay

    raw_cands = trace.get("raw_candidates", [])
    best_rc, hit_rc, _ = best_iou_and_entry(
        bbox, raw_cands, lambda e: same_page(e) and is_formula_raw_entry(e)
    )
    evidence["best_iou_raw_candidates"] = best_rc

    # Classification
    if best_f < 0.25 and best_any < 0.25:
        classification = "layout_model_or_preprocess_mismatch"
        verdict_note = (
            "No layout_raw formula (and no raw box) reaches IoU>=0.25 vs Python truth."
        )
    elif hit_f is not None and hit_f.get("passed_confidence_threshold") is False:
        classification = "true_formula_missing"
        verdict_note = (
            "High-IoU formula-class raw detection exists but failed per-category "
            "confidence threshold before prefilter (layout postprocess/threshold)."
        )
    elif best_f >= iou_thr and best_pre < 0.25:
        classification = "true_formula_missing"
        verdict_note = (
            "Formula raw hit strong but no prefilter entry with similar IoU — "
            "dropped at confidence filter / not promoted to prefilter pipeline."
        )
    elif best_pre >= iou_thr and best_rc < iou_thr:
        classification = "true_formula_missing"
        verdict_note = (
            "Signal in prefilter/layout but lost before equation raw_candidates "
            "(NMS / fragment suppression / layout mapping)."
        )
    elif best_rc >= iou_thr:
        classification = "split_merge"
        verdict_note = (
            "Equation path raw_candidates still overlap — refine/filter stage "
            "reordered or merged differently vs Python."
        )
    elif 0.25 <= best_f < iou_thr:
        classification = "low_iou_equivalent"
        verdict_note = "Formula-class raw exists but IoU below matching threshold."
    else:
        classification = "cpp_noise_candidate"
        verdict_note = "Ambiguous; inspect trace manually."

    return {
        "classification": classification,
        "evidence": evidence,
        "verdict_note": verdict_note,
    }


def greedy_match(
    py: List[Dict[str, Any]],
    cpp: List[Dict[str, Any]],
    iou_thr: float,
) -> List[Tuple[float, int, int]]:
    pairs: List[Tuple[float, int, int]] = []
    for ci, c in enumerate(cpp):
        cb = [float(x) for x in c["bbox"]]
        for pi, p in enumerate(py):
            pb = p["bbox"]
            pairs.append((iou(cb, pb), ci, pi))
    pairs.sort(reverse=True)
    used_py: set[int] = set()
    used_cpp: set[int] = set()
    matches: List[Tuple[float, int, int]] = []
    for score, ci, pi in pairs:
        if score < iou_thr:
            continue
        if ci in used_cpp or pi in used_py:
            continue
        used_cpp.add(ci)
        used_py.add(pi)
        matches.append((score, ci, pi))
    return matches


def render_page_png(pdf_path: Path, out_png: Path, dpi: int = 200) -> None:
    import pypdfium2 as pdfium

    out_png.parent.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        page = doc[0]
        scale = dpi / 72.0
        bitmap = page.render(scale=scale)
        pil = bitmap.to_pil()
        pil.save(out_png)
    finally:
        doc.close()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python-model", type=Path, required=True)
    ap.add_argument("--cpp-trace", type=Path, required=True)
    ap.add_argument("--input-pdf", type=Path, required=True, help="Single-page PDF used for C++ run")
    ap.add_argument("--out-json", type=Path, required=True)
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    ap.add_argument(
        "--rendered-page-png",
        type=Path,
        default=None,
        help="Optional output path for rendered page (dpi must match run)",
    )
    args = ap.parse_args()

    model_pages = json.loads(args.python_model.read_text())
    model_page = model_pages[0]
    trace = json.loads(args.cpp_trace.read_text())

    py_cand = python_formula_candidates(model_page)
    cpp_kept = cpp_refined_kept(trace)
    latex_map = latex_by_filtered_index(trace)

    matches_idx = greedy_match(py_cand, cpp_kept, args.iou_threshold)
    used_py = {m[2] for m in matches_idx}
    used_cpp = {m[1] for m in matches_idx}

    matches: List[Dict[str, Any]] = []
    norm_eq = 0
    for score, ci, pi in matches_idx:
        p = py_cand[pi]
        c = cpp_kept[ci]
        fidx = int(c.get("filtered_candidate_index", -1))
        cpp_latex = latex_map.get(fidx, "")
        py_latex = p.get("latex", "")
        neq = normalize_latex(py_latex) == normalize_latex(cpp_latex)
        norm_eq += int(neq)
        matches.append(
            {
                "python_index": pi,
                "cpp_filtered_slot_index": ci,
                "filtered_candidate_index": fidx,
                "iou": score,
                "python_bbox": p["bbox"],
                "cpp_bbox": [float(x) for x in c["bbox"]],
                "python_latex": py_latex,
                "cpp_latex": cpp_latex,
                "normalized_latex_equal": neq,
            }
        )

    py_page = int(model_page.get("page_info", {}).get("page_idx", 0))
    # C++ single-page PDF uses logical page 0 in trace
    trace_page = int(trace.get("layout_raw_boxes", [{}])[0].get("page", 0)) if trace.get(
        "layout_raw_boxes"
    ) else 0

    py_unmatched: List[Dict[str, Any]] = []
    for pi, p in enumerate(py_cand):
        if pi in used_py:
            continue
        best = max((iou(p["bbox"], [float(x) for x in c["bbox"]]), ci) for ci, c in enumerate(cpp_kept))
        diag = classify_python_unmatched(p["bbox"], trace_page, trace, args.iou_threshold)
        py_unmatched.append(
            {
                "python_index": pi,
                "bbox": p["bbox"],
                "score": p.get("score"),
                "latex": p.get("latex"),
                "best_cpp_iou": best[0],
                "best_cpp_filtered_slot": best[1],
                "crop_path": None,
                "layers": diag["evidence"],
                "classification": diag["classification"],
                "keep_drop_conclusion": "python_truth_not_one_to_one_matched_at_iou",
                "evidence": diag["verdict_note"],
            }
        )

    cpp_unmatched: List[Dict[str, Any]] = []
    for ci, c in enumerate(cpp_kept):
        if ci in used_cpp:
            continue
        cb = [float(x) for x in c["bbox"]]
        best = max((iou(cb, p["bbox"]), pi) for pi, p in enumerate(py_cand))
        fidx = int(c.get("filtered_candidate_index", -1))
        cpp_unmatched.append(
            {
                "cpp_filtered_slot_index": ci,
                "filtered_candidate_index": fidx,
                "bbox": cb,
                "confidence": c.get("confidence"),
                "latex": latex_map.get(fidx, ""),
                "best_python_iou": best[0],
                "best_python_index": best[1],
                "crop_path": None,
                "classification": "cpp_noise_candidate",
                "keep_drop_conclusion": "cpp_kept_no_python_pair_at_iou",
                "evidence": "No Python category_id==13 latex box with IoU>=threshold.",
            }
        )

    n_match = len(matches)
    ratio = norm_eq / n_match if n_match else 0.0
    unmatched_evidence_ok = (
        all(bool(u.get("classification")) and bool(u.get("evidence")) for u in py_unmatched)
        and all(bool(u.get("classification")) and bool(u.get("evidence")) for u in cpp_unmatched)
    )
    gate1_pass = (
        len(py_cand) == 30
        and len(cpp_kept) == 30
        and n_match >= 28
        and ratio >= 0.95
        and unmatched_evidence_ok
    )

    # Global blocker narrative (evidence-driven): any hard missing raw formula
    # detection upgrades to layout/model/preprocess mismatch.
    any_b = any(
        u.get("classification") == "layout_model_or_preprocess_mismatch" for u in py_unmatched
    )
    if gate1_pass:
        blocker_verdict = "none"
    elif any_b:
        blocker_verdict = "layout_model_or_preprocess_mismatch"
    else:
        blocker_verdict = "postprocess_nms_threshold_merge_blocker"

    blocker_comment = ""
    if (
        not gate1_pass
        and not py_unmatched
        and not cpp_unmatched
        and n_match == len(py_cand) == len(cpp_kept)
    ):
        blocker_comment = (
            "All Python truth boxes match a kept C++ refined candidate at IoU>=threshold; "
            "Gate1 failure is from normalized LaTeX inequality on matched pairs (Formula "
            "decode / tokenizer spacing vs Python pipeline), not missing layout raw boxes."
        )

    rendered = args.rendered_page_png
    if rendered is not None:
        render_page_png(args.input_pdf, rendered, args.dpi)

    out_obj: Dict[str, Any] = {
        "inputs": {
            "python_model_json": str(args.python_model.resolve()),
            "cpp_formula_trace_json": str(args.cpp_trace.resolve()),
            "input_pdf": str(args.input_pdf.resolve()),
            "dpi": args.dpi,
            "python_model_sha256": sha256_file(args.python_model),
            "cpp_trace_sha256": sha256_file(args.cpp_trace),
            "rendered_page_png": str(rendered.resolve()) if rendered else None,
            "python_page_idx_in_model_json": py_page,
            "cpp_trace_page_field": trace_page,
        },
        "normalization_spec": "normalize_latex(s) = re.sub(r'\\s+', '', s or ''); applied to both Python and C++ strings.",
        "gate": {
            "python_candidate_count": len(py_cand),
            "cpp_refined_candidate_count": len(cpp_kept),
            "iou_threshold": args.iou_threshold,
            "matched_pairs": n_match,
            "normalized_latex_equal": norm_eq,
            "normalized_latex_total": n_match,
            "normalized_latex_ratio": ratio,
            "gate1_pass": gate1_pass,
            "gate1_rule_checks": {
                "rule_counts_30_30": len(py_cand) == 30 and len(cpp_kept) == 30,
                "rule_matched_pairs_ge_28": n_match >= 28,
                "rule_normalized_latex_ratio_ge_0_95_on_pairs": ratio >= 0.95,
                "rule_all_unmatched_have_evidence": unmatched_evidence_ok,
            },
        },
        "blocker_verdict": blocker_verdict,
        "blocker_verdict_comment": blocker_comment or None,
        "matches": matches,
        "python_unmatched": py_unmatched,
        "cpp_unmatched": cpp_unmatched,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(out_obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(args.out_json), "gate1_pass": gate1_pass, "blocker_verdict": blocker_verdict}, indent=2))
    return 0 if gate1_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
