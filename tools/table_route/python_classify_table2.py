#!/usr/bin/env python3
"""Run RapidDoc Python's TableCls + default (UNET_SLANET_PLUS) backends
on the three C++-dumped crops of 表格2.pdf to determine which lane the
Python reference would put each sample into, and whether HTML is produced."""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import cv2

CROP_DIR = Path("/home/deepx/Desktop/RapidDocCpp/output-benchmark/table_route_20260417/head/crops")
OUT_JSON = Path(
    "/home/deepx/Desktop/RapidDocCpp/output-benchmark/table_route_20260417/compare_python/PYTHON_RESULT.json"
)

from rapid_doc.model.table.rapid_table_self.table_cls import TableCls
from rapid_doc.model.table.rapid_table_self import (  # noqa: E402
    ModelType,
    RapidTable,
    RapidTableInput,
)


def main() -> int:
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    crop_files = sorted(CROP_DIR.glob("table_p*.png"))
    if not crop_files:
        print("No crops found under", CROP_DIR)
        return 1

    print("Loading TableCls + UNET + SLANET_PLUS (downloads on first run)...")
    table_cls = TableCls()
    wired_model = RapidTable(RapidTableInput(model_type=ModelType.UNET, use_ocr=False))
    wireless_model = RapidTable(RapidTableInput(model_type=ModelType.SLANETPLUS, use_ocr=False))

    results = []
    for crop_path in crop_files:
        entry = {"crop": crop_path.name}
        try:
            bgr = cv2.imread(str(crop_path))
            if bgr is None:
                entry["error"] = "cv2.imread returned None"
                results.append(entry)
                continue
            entry["crop_shape"] = list(bgr.shape)
            cls_label, cls_ms = table_cls(bgr)
            entry["table_cls"] = cls_label
            entry["table_cls_ms"] = float(cls_ms)

            backend = wired_model if cls_label == "wired" else wireless_model
            entry["backend"] = "UNET" if cls_label == "wired" else "SLANETPLUS"

            try:
                ocr_stub = [[], [], []]
                table_results = backend(bgr, ocr_stub)
                html = getattr(table_results, "pred_html", None) or ""
                elapse = getattr(table_results, "elapse", None)
                entry["html_present"] = bool(html.strip())
                entry["html_len"] = len(html)
                entry["html_preview"] = html[:400]
                entry["elapse"] = float(elapse) if isinstance(elapse, (int, float)) else elapse
            except Exception as exc:  # noqa: BLE001
                entry["backend_error"] = repr(exc)
                entry["backend_traceback"] = traceback.format_exc()
        except Exception as exc:  # noqa: BLE001
            entry["error"] = repr(exc)
            entry["traceback"] = traceback.format_exc()
        results.append(entry)
        print(json.dumps(entry, ensure_ascii=False))

    OUT_JSON.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print("wrote", OUT_JSON)
    return 0


if __name__ == "__main__":
    sys.exit(main())
