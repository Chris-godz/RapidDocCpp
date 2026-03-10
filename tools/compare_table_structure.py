#!/usr/bin/env python3
"""Parse first table from two MD files and compare row/col structure (logical columns per row)."""
import re
from pathlib import Path

def extract_first_table(md_path: Path) -> str | None:
    with open(md_path, "r", encoding="utf-8") as f:
        text = f.read()
    m = re.search(r"<table[^>]*>(.*?)</table>", text, re.DOTALL)
    return m.group(1) if m else None

def parse_row_cols(tr_fragment: str) -> list[int]:
    """Sum colspans in one <tr>...</tr> to get logical column count. Default colspan=1."""
    cols = 0
    # Match <td ...> or <td rowspan=N colspan=M>
    for m in re.finditer(r"<t[dh](?:\s[^>]*)?>", tr_fragment):
        tag = m.group(0)
        c = 1
        if "colspan" in tag:
            cm = re.search(r"colspan\s*=\s*(\d+)", tag, re.I)
            if cm:
                c = int(cm.group(1))
        cols += c
    return cols

def table_rows(table_body: str) -> list[str]:
    """Split table body into <tr>...</tr> chunks (handling nested tags)."""
    rows = []
    depth = 0
    start = 0
    i = 0
    while i < len(table_body):
        if table_body[i:i+4] == "<tr>":
            if depth == 0:
                start = i
            depth += 1
            i += 4
            continue
        if table_body[i:i+5] == "</tr>":
            depth -= 1
            if depth == 0:
                rows.append(table_body[start:i+5])
            i += 5
            continue
        i += 1
    return rows

def main():
    repo = Path(__file__).resolve().parents[1]
    cpp_md = repo / "test" / "fixtures" / "e2e" / "cpp" / "表格1" / "表格1.md"
    py_md = repo / "test" / "fixtures" / "e2e" / "python" / "表格1" / "auto" / "表格1.md"

    cpp_table = extract_first_table(cpp_md)
    py_table = extract_first_table(py_md)
    if not cpp_table or not py_table:
        print("Missing table in one or both files")
        return

    cpp_rows = table_rows(cpp_table)
    py_rows = table_rows(py_table)

    print("=== 表格1 结构对比 (C++ vs Python) ===\n")
    print(f"行数: C++ {len(cpp_rows)} vs Python {len(py_rows)}\n")

    # Per-row logical column count
    print("前 15 行每行逻辑列数:")
    print("Row | C++ cols | Py cols | diff")
    print("-" * 35)
    for r in range(min(15, len(cpp_rows), len(py_rows))):
        cpp_cols = parse_row_cols(cpp_rows[r])
        py_cols = parse_row_cols(py_rows[r])
        diff = cpp_cols - py_cols
        mark = " ***" if diff != 0 else ""
        print(f"  {r+1:2} |    {cpp_cols:3}    |   {py_cols:3}   | {diff:+d}{mark}")

    if len(cpp_rows) != len(py_rows):
        print(f"\n... (C++ 共 {len(cpp_rows)} 行, Python 共 {len(py_rows)} 行)")

    # Summary
    cpp_cols_first = [parse_row_cols(r) for r in cpp_rows[:20]]
    py_cols_first = [parse_row_cols(r) for r in py_rows[:20]]
    cpp_max = max(cpp_cols_first) if cpp_cols_first else 0
    py_max = max(py_cols_first) if py_cols_first else 0
    print(f"\n前20行最大逻辑列数: C++ {cpp_max} vs Python {py_max}")
    if cpp_max > py_max:
        print("→ C++ 列数多于 Python，可能竖线检测过多，导致网格过细。")

if __name__ == "__main__":
    main()
