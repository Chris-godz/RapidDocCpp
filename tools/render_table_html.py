#!/usr/bin/env python3
"""Render first table from C++/Python 表格1 markdown to HTML for visual comparison."""
import re
import sys
from pathlib import Path

def extract_first_table(md_path: Path) -> str | None:
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()
    # Match first <table>...</table> (allow newlines)
    m = re.search(r"<table[^>]*>.*?</table>", text, re.DOTALL)
    return m.group(0) if m else None

def wrap_html(table_fragment: str, title: str) -> str:
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>{title}</title>
<style>
table {{ border-collapse: collapse; font-size: 12px; }}
td, th {{ border: 1px solid #333; padding: 4px 6px; vertical-align: top; }}
</style>
</head><body>
<h2>{title}</h2>
{table_fragment}
</body></html>
"""

def main():
    repo = Path(__file__).resolve().parents[1]
    fixtures = repo / "test" / "fixtures" / "e2e"
    cpp_md = fixtures / "cpp" / "表格1" / "表格1.md"
    py_md = fixtures / "python" / "表格1" / "auto" / "表格1.md"
    out_dir = repo / "test" / "fixtures" / "e2e" / "table_render"
    out_dir.mkdir(parents=True, exist_ok=True)

    for label, md_path in [("C++ 表格1", cpp_md), ("Python 表格1", py_md)]:
        if not md_path.exists():
            print(f"Skip: {md_path} not found", file=sys.stderr)
            continue
        table = extract_first_table(md_path)
        if not table:
            print(f"No table in {md_path}", file=sys.stderr)
            continue
        html = wrap_html(table, label)
        out_name = "cpp_表格1.html" if "C++" in label else "python_表格1.html"
        out_path = out_dir / out_name
        out_path.write_text(html, encoding="utf-8")
        print(f"Wrote {out_path}")

if __name__ == "__main__":
    main()
