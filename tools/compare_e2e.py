#!/usr/bin/env python3
"""
Compare C++ vs Python E2E results for all test PDFs.

Handles different output structures:
  - Python (RapidDoc): {stem}/auto/{stem}.md, {stem}/auto/{stem}_content_list.json
  - C++:               {stem}/{stem}.md, {stem}/content_list.json

Usage:
    python tools/compare_e2e.py [--cpp test/fixtures/e2e/cpp] [--python test/fixtures/e2e/python]
"""

import argparse
import json
import os
import difflib
from pathlib import Path


def find_md(base_dir, stem):
    """Find the markdown file in either output structure."""
    candidates = [
        base_dir / f"{stem}.md",
        base_dir / "auto" / f"{stem}.md",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def find_content_list(base_dir, stem):
    """Find the content list JSON."""
    candidates = [
        base_dir / "content_list.json",
        base_dir / "auto" / f"{stem}_content_list.json",
        base_dir / f"{stem}_content_list.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def compare_markdown(cpp_dir, py_dir, stem):
    """Compare markdown outputs."""
    cpp_md_path = find_md(cpp_dir, stem)
    py_md_path = find_md(py_dir, stem)

    if not cpp_md_path or not py_md_path:
        return {"status": "missing", "cpp_found": cpp_md_path is not None, "py_found": py_md_path is not None}

    cpp_text = cpp_md_path.read_text(errors="replace")
    py_text = py_md_path.read_text(errors="replace")

    cpp_lines = cpp_text.splitlines()
    py_lines = py_text.splitlines()

    diff = list(difflib.unified_diff(py_lines, cpp_lines, lineterm="", n=0))
    adds = sum(1 for l in diff if l.startswith("+") and not l.startswith("+++"))
    dels = sum(1 for l in diff if l.startswith("-") and not l.startswith("---"))

    # Check if both have table HTML
    cpp_has_table = "<table" in cpp_text.lower()
    py_has_table = "<table" in py_text.lower()

    return {
        "cpp_chars": len(cpp_text),
        "py_chars": len(py_text),
        "ratio": round(len(cpp_text) / len(py_text), 3) if py_text else 0,
        "cpp_lines": len(cpp_lines),
        "py_lines": len(py_lines),
        "diff_adds": adds,
        "diff_dels": dels,
        "cpp_has_table": cpp_has_table,
        "py_has_table": py_has_table,
    }


def compare_content(cpp_dir, py_dir, stem):
    """Compare content list structure."""
    cpp_path = find_content_list(cpp_dir, stem)
    py_path = find_content_list(py_dir, stem)

    if not cpp_path or not py_path:
        return {"status": "missing"}

    try:
        cpp_data = json.loads(cpp_path.read_text(errors="replace"))
        py_data = json.loads(py_path.read_text(errors="replace"))
    except json.JSONDecodeError:
        return {"status": "parse_error"}

    # Flatten to text
    def extract_texts(data):
        texts = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, list):
                    for sub in item:
                        if isinstance(sub, dict):
                            texts.append(sub.get("text", ""))
                elif isinstance(item, dict):
                    texts.append(item.get("text", ""))
        return texts

    cpp_texts = extract_texts(cpp_data)
    py_texts = extract_texts(py_data)

    cpp_all = "\n".join(t for t in cpp_texts if t)
    py_all = "\n".join(t for t in py_texts if t)

    return {
        "cpp_elements": len(cpp_texts),
        "py_elements": len(py_texts),
        "cpp_text_chars": len(cpp_all),
        "py_text_chars": len(py_all),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp", default="test/fixtures/e2e/cpp")
    parser.add_argument("--python", default="test/fixtures/e2e/python")
    parser.add_argument("--output", default="test/fixtures/e2e/comparison_report.json")
    args = parser.parse_args()

    cpp_root = Path(args.cpp)
    py_root = Path(args.python)

    cpp_stems = {d.name for d in cpp_root.iterdir() if d.is_dir()}
    py_stems = {d.name for d in py_root.iterdir() if d.is_dir()}
    common = sorted(cpp_stems & py_stems)

    print(f"C++: {len(cpp_stems)} PDFs, Python: {len(py_stems)} PDFs, Common: {len(common)}")
    print(f"{'='*100}")
    print(f"{'PDF':<45} {'C++ MD':>8} {'Py MD':>8} {'Ratio':>6} {'C++ Tbl':>8} {'Py Tbl':>8} {'Status'}")
    print(f"{'-'*100}")

    report = []
    for stem in common:
        cpp_dir = cpp_root / stem
        py_dir = py_root / stem

        entry = {"pdf": stem}
        entry["markdown"] = compare_markdown(cpp_dir, py_dir, stem)
        entry["content"] = compare_content(cpp_dir, py_dir, stem)

        md = entry["markdown"]
        if "status" in md:
            status = "MISSING"
        else:
            ratio = md["ratio"]
            both_table = md["cpp_has_table"] == md["py_has_table"]
            if 0.3 < ratio < 3.0:
                status = "OK" if both_table else "TABLE_DIFF"
            else:
                status = "SIZE_DIFF"

        entry["status"] = status
        report.append(entry)

        cpp_c = md.get("cpp_chars", "?")
        py_c = md.get("py_chars", "?")
        ratio_s = f"{md.get('ratio', 0):.2f}" if "ratio" in md else "?"
        cpp_tbl = "Yes" if md.get("cpp_has_table") else "No"
        py_tbl = "Yes" if md.get("py_has_table") else "No"
        print(f"  {stem:<43} {cpp_c:>8} {py_c:>8} {ratio_s:>6} {cpp_tbl:>8} {py_tbl:>8} [{status}]")

    print(f"{'='*100}")

    ok_count = sum(1 for r in report if r["status"] == "OK")
    table_diff = sum(1 for r in report if r["status"] == "TABLE_DIFF")
    size_diff = sum(1 for r in report if r["status"] == "SIZE_DIFF")
    missing = sum(1 for r in report if r["status"] == "MISSING")

    print(f"\nSummary: OK={ok_count}, TABLE_DIFF={table_diff}, SIZE_DIFF={size_diff}, MISSING={missing}")
    print(f"  (Note: C++ processes page 0 only, Python processes all pages)")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport: {args.output}")


if __name__ == "__main__":
    main()
