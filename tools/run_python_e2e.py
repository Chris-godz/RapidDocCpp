#!/usr/bin/env python3
"""
Run the REAL RapidDoc DXNN pipeline on all test PDFs.

Calls the same pipeline_doc_analyze + pipeline_union_make that demo_offline.py
uses, with table_enable=True, formula_enable=True, all engines = dxengine.

Usage:
    # Set env first:
    source /home/deepx/Desktop/RapidDoc/deepx_scripts/set_env.sh 1 2 1 3 2 4
    # Then run:
    python tools/run_python_e2e.py [--pdf-dir test_files] [--output test/fixtures/e2e/python]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── Environment ──────────────────────────────────────────────────────────
# Set required env vars if not already set (same defaults as set_env.sh 1 2 1 3 2 4)
_ENV_DEFAULTS = {
    "CUSTOM_INTER_OP_THREADS_COUNT": "1",
    "CUSTOM_INTRA_OP_THREADS_COUNT": "2",
    "DXRT_DYNAMIC_CPU_THREAD": "1",
    "DXRT_TASK_MAX_LOAD": "3",
    "NFH_INPUT_WORKER_THREADS": "2",
    "NFH_OUTPUT_WORKER_THREADS": "4",
}
for k, v in _ENV_DEFAULTS.items():
    os.environ.setdefault(k, v)

os.environ["MINERU_MODEL_SOURCE"] = "local"

RAPIDDOC_ROOT = Path(__file__).resolve().parent.parent.parent / "RapidDoc"


def main():
    parser = argparse.ArgumentParser(description="Run real RapidDoc pipeline on test PDFs")
    parser.add_argument("--pdf-dir", default="test_files")
    parser.add_argument("--output", default="test/fixtures/e2e/python")
    parser.add_argument("--rapiddoc", default=str(RAPIDDOC_ROOT))
    parser.add_argument("--start-page", type=int, default=0)
    parser.add_argument("--end-page", type=int, default=None)
    args = parser.parse_args()

    rapiddoc = Path(args.rapiddoc).resolve()
    if str(rapiddoc) not in sys.path:
        sys.path.insert(0, str(rapiddoc))
    # Also add demo/ so we can import from there
    demo_dir = rapiddoc / "demo"

    # ── Import the real pipeline ───────────────────────────────────────
    from rapid_doc.cli.common import prepare_env, read_fn
    from rapid_doc.data.data_reader_writer import FileBasedDataWriter
    from rapid_doc.utils.enum_class import MakeMode
    from rapid_doc.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
    from rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
    from rapid_doc.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
    from rapid_doc.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2

    from rapid_doc.model.layout.rapid_layout_self import ModelType as LayoutModelType
    from rapid_doc.model.layout.rapid_layout_self.utils.typings import EngineType as LayoutEngineType
    from rapid_doc.model.table.rapid_table_self import ModelType as TableModelType

    dxnn_models_dir = rapiddoc / "dxnn_models"
    onnx_models_dir = rapiddoc / "onnx_models"

    if not dxnn_models_dir.exists():
        print(f"ERROR: dxnn_models not found at {dxnn_models_dir}")
        sys.exit(1)

    # ── Build configs (same as demo_offline.py DXNN mode) ─────────────
    layout_config = {
        "model_type": LayoutModelType.PP_DOCLAYOUT_L,
        "engine_type": LayoutEngineType.DXENGINE,
        "model_dir_or_path": str(dxnn_models_dir / "pp_doclayout_l_part1.dxnn"),
        "sub_model_path": str(onnx_models_dir / "pp_doclayout_l_part2.onnx"),
    }

    ocr_config = {
        "engine_type": "dxengine",
        "use_det_mode": "auto",
        "Det.model_path": str(dxnn_models_dir / "det_v5_640_640.dxnn"),
        "Rec.model_path": str(dxnn_models_dir / "rec_v5_ratio_10.dxnn"),
        "char_dict_path": str(rapiddoc / "value_compare" / "recognition" / "character_dict_from_onnx.txt"),
        "use_multi_det_model": True,
        "Det.model_paths": {
            1: str(dxnn_models_dir / "det_v5_640_640.dxnn"),
            2: str(dxnn_models_dir / "det_v5_320_640.dxnn"),
            4: str(dxnn_models_dir / "det_v5_160_640.dxnn"),
            10: str(dxnn_models_dir / "det_v5_64_640.dxnn"),
        },
        "use_multi_rec_model": True,
        "Rec.model_paths": {
            3: str(dxnn_models_dir / "rec_v5_ratio_3.dxnn"),
            5: str(dxnn_models_dir / "rec_v5_ratio_5.dxnn"),
            10: str(dxnn_models_dir / "rec_v5_ratio_10.dxnn"),
            15: str(dxnn_models_dir / "rec_v5_ratio_15.dxnn"),
            25: str(dxnn_models_dir / "rec_v5_ratio_25.dxnn"),
            35: str(dxnn_models_dir / "rec_v5_ratio_35.dxnn"),
        },
        "save_debug_images": False,
    }

    table_config = {
        "model_type": TableModelType.UNET,
        "engine_type": "dxengine",
        "model_dir_or_path": str(dxnn_models_dir / "unet.dxnn"),
    }

    formula_enable = False
    formula_config = {}

    checkbox_config = {"checkbox_enable": False}

    # ── Discover PDFs ────────────────────────────────────────────────
    pdf_dir = Path(args.pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs in {pdf_dir}")

    output_root = Path(args.output)
    os.makedirs(output_root, exist_ok=True)

    parse_method = "auto"
    all_summaries = []

    for pdf_path in pdfs:
        stem = pdf_path.stem
        print(f"\n{'='*60}")
        print(f"Processing: {pdf_path.name}")
        print(f"{'='*60}")

        t0 = time.time()
        try:
            pdf_bytes = read_fn(pdf_path)

            # Page range
            if args.end_page is not None:
                pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(
                    pdf_bytes, args.start_page, args.end_page)

            # ── Run the real pipeline ─────────────────────────────
            infer_results, all_image_lists, all_page_dicts, lang_list, ocr_enabled_list = \
                pipeline_doc_analyze(
                    [pdf_bytes],
                    parse_method=parse_method,
                    formula_enable=formula_enable,
                    table_enable=True,
                    layout_config=layout_config,
                    ocr_config=ocr_config,
                    formula_config=formula_config,
                    table_config=table_config,
                    checkbox_config=checkbox_config,
                    use_async_pipeline=False,
                )

            # ── Post-process: middle JSON -> Markdown ─────────────
            per_pdf_output = output_root / stem / parse_method
            per_pdf_images = per_pdf_output / "images"
            os.makedirs(per_pdf_images, exist_ok=True)

            image_writer = FileBasedDataWriter(str(per_pdf_images))
            md_writer = FileBasedDataWriter(str(per_pdf_output))

            model_list = infer_results[0]
            images_list = all_image_lists[0]
            pdf_dict = all_page_dicts[0]
            _lang = lang_list[0]
            _ocr_enable = ocr_enabled_list[0]

            middle_json = pipeline_result_to_middle_json(
                model_list, images_list, pdf_dict, image_writer, _lang, _ocr_enable,
                formula_enable, ocr_config=ocr_config,
            )
            pdf_info = middle_json["pdf_info"]

            # Markdown
            image_dir = "images"
            md_content = pipeline_union_make(pdf_info, MakeMode.MM_MD, image_dir)
            md_writer.write_string(f"{stem}.md", md_content)

            # Content list
            content_list = pipeline_union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir)
            md_writer.write_string(
                f"{stem}_content_list.json",
                json.dumps(content_list, ensure_ascii=False, indent=2),
            )

            elapsed = time.time() - t0
            page_count = len(images_list)

            summary = {
                "pdf": pdf_path.name,
                "stem": stem,
                "total_pages": page_count,
                "markdown_chars": len(md_content),
                "total_time_ms": round(elapsed * 1000, 1),
            }
            md_writer.write_string("summary.json", json.dumps(summary, indent=2))
            all_summaries.append(summary)

            print(f"  Pages: {page_count}, MD: {len(md_content)} chars, "
                  f"Time: {elapsed:.1f}s")
            print(f"  Output: {per_pdf_output}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback; traceback.print_exc()
            all_summaries.append({"pdf": pdf_path.name, "error": str(e)})

    with open(str(output_root / "all_summaries.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Done! {len(all_summaries)} PDFs processed")
    print(f"Results: {output_root}")


if __name__ == "__main__":
    main()
