#!/usr/bin/env python3
"""
Compare replay JSON latex vs Python reference; emit crop_parity_latex_summary.json.
Normalization: re.sub(r'\\s+', '', s) — same as Gate2 evidence tooling.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


def norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def count_equal(ref: Dict[int, str], replay_path: Path) -> tuple[int, List[int]]:
    r = json.loads(replay_path.read_text())
    out = {int(o["index"]): o.get("latex", "") for o in r.get("outputs", [])}
    bad: List[int] = []
    ok = 0
    for i in range(30):
        if norm(ref.get(i, "")) == norm(out.get(i, "")):
            ok += 1
        else:
            bad.append(i)
    return ok, bad


def verdict(count_b: int, count_a: int) -> str:
    if count_b >= 27:
        return "B_close_to_A_suspect_writeback_compare_output_path"
    if count_b <= 20:
        return "B_near_gate1_suspect_crop_extraction_roi_render_parity_mismatch"
    return "inconclusive_middle_band_recommend_roi_diff_followup"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--python-latex-json", type=Path, required=True, help="python_same_crop_replay.json")
    ap.add_argument("--replay-a", type=Path, required=True)
    ap.add_argument("--replay-b", type=Path, required=True)
    ap.add_argument("--out-json", type=Path, required=True)
    args = ap.parse_args()

    pyj = json.loads(args.python_latex_json.read_text())
    ref = {int(o["index"]): o.get("latex", "") for o in pyj.get("outputs", [])}

    count_a, miss_a = count_equal(ref, args.replay_a)
    count_b, miss_b = count_equal(ref, args.replay_b)

    summary = {
        "normalized_latex_equal_count_A": count_a,
        "normalized_latex_equal_count_B": count_b,
        "total": 30,
        "normalization_spec": "re.sub(r'\\s+', '', s or '')",
        "python_latex_reference": str(args.python_latex_json.resolve()),
        "replay_a": str(args.replay_a.resolve()),
        "replay_b": str(args.replay_b.resolve()),
        "mismatch_indices_A": miss_a,
        "mismatch_indices_B": miss_b,
        "verdict": verdict(count_b, count_a),
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"A": count_a, "B": count_b, "verdict": summary["verdict"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
