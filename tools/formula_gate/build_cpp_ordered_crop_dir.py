#!/usr/bin/env python3
"""
Map C++ pipeline infer-order crop PNGs (cpp_infer_order_XXX.png) to Python
formula index order (crop_000.png …) using formula_trace crop_mappings +
greedy IoU match vs Python model.json (same rules as build_candidate_mismatch_review).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List


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


def python_formula_candidates(model_page: Dict[str, Any]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, d in enumerate(model_page.get("layout_dets", [])):
        if d.get("category_id") == 13 and "latex" in d:
            bbox = d.get("bbox") or bbox_from_poly(d["poly"])
            out.append(
                {
                    "python_index": len(out),
                    "bbox": [float(x) for x in bbox],
                }
            )
    return out


def cpp_refined_kept(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [d for d in trace.get("filtered_candidates", []) if d.get("kept")]


def greedy_match(
    py: List[Dict[str, Any]],
    cpp: List[Dict[str, Any]],
    iou_thr: float,
) -> List[Tuple[float, int, int]]:
    pairs: List[Tuple[float, int, int]] = []
    for ci, c in enumerate(cpp):
        cb = [float(x) for x in c["bbox"]]
        for pi, p in enumerate(py):
            pairs.append((iou(cb, p["bbox"]), ci, pi))
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


def filtered_index_to_infer_order(trace: Dict[str, Any]) -> Dict[int, int]:
    """filtered_candidate_index -> infer sequence (cpp_infer_order_XXX)."""
    m: Dict[int, int] = {}
    for row in trace.get("crop_mappings", []):
        fci = row.get("filtered_candidate_index")
        cidx = row.get("crop_index")
        if fci is None or cidx is None:
            continue
        ic = int(cidx)
        if ic < 0:
            continue
        m[int(fci)] = ic
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer-crop-dir", type=Path, required=True, help="Dir with cpp_infer_order_*.png")
    ap.add_argument("--trace-json", type=Path, required=True)
    ap.add_argument("--python-model", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    args = ap.parse_args()

    model = json.loads(args.python_model.read_text())[0]
    trace = json.loads(args.trace_json.read_text())
    py = python_formula_candidates(model)
    cpp_kept = cpp_refined_kept(trace)
    if len(py) != 30:
        raise SystemExit(f"expected 30 python formula boxes, got {len(py)}")
    if len(cpp_kept) != 30:
        raise SystemExit(f"expected 30 cpp kept candidates, got {len(cpp_kept)}")

    fci_to_infer = filtered_index_to_infer_order(trace)
    if len(fci_to_infer) != 30:
        raise SystemExit(
            f"expected 30 crop_mappings with crop_index>=0, got {len(fci_to_infer)} keys: {sorted(fci_to_infer)}"
        )

    matches = greedy_match(py, cpp_kept, args.iou_threshold)
    if len(matches) != 30:
        raise SystemExit(f"greedy match expected 30 pairs, got {len(matches)}")

    py_to_fci: Dict[int, int] = {}
    for _score, ci_slot, pi in matches:
        fidx = int(cpp_kept[ci_slot].get("filtered_candidate_index", -1))
        if fidx < 0:
            raise SystemExit(f"bad filtered_candidate_index for slot {ci_slot}")
        py_to_fci[int(pi)] = fidx

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for pi in range(30):
        fci = py_to_fci[pi]
        infer_idx = fci_to_infer.get(fci)
        if infer_idx is None:
            raise SystemExit(f"python_index={pi} -> filtered_candidate_index={fci} has no infer mapping")
        src = args.infer_crop_dir / f"cpp_infer_order_{infer_idx:03d}.png"
        if not src.is_file():
            raise SystemExit(f"missing infer crop: {src}")
        dst = args.out_dir / f"crop_{pi:03d}.png"
        shutil.copy2(src, dst)
        manifest.append(
            {
                "python_index": pi,
                "filtered_candidate_index": fci,
                "infer_order_index": infer_idx,
                "source_png": str(src.resolve()),
            }
        )

    (args.out_dir / "cpp_ordered_manifest.json").write_text(
        json.dumps({"mappings": manifest}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"wrote": str(args.out_dir.resolve()), "count": 30}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
