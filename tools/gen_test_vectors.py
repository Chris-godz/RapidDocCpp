#!/usr/bin/env python3
"""
Generate test vectors (fixtures) for C++ <-> Python cross-validation.

Uses the RapidDoc Python DXNN pipeline to process a test PDF and saves
intermediate results at each stage as .npy / .json files.

Usage:
    python gen_test_vectors.py --pdf <path.pdf> --output <dir> [--rapiddoc-root <path>]
    python gen_test_vectors.py --image <path.png> --output <dir>

Output structure:
    <output>/
    ├── layout/
    │   ├── input_image.png
    │   ├── preprocessed.npy            # (1, H, W, 3) uint8 NHWC
    │   ├── scale_factor.npy            # (1, 2) float32
    │   ├── dxnn_out_0.npy              # DX Engine output tensor 0
    │   ├── dxnn_out_1.npy              # DX Engine output tensor 1
    │   ├── onnx_boxes_raw.npy          # Raw ONNX output [N, 6]
    │   ├── onnx_raw_output.npy         # ONNX output before NMS
    │   ├── nms_result.json             # NMS'd boxes before large-image filter
    │   └── boxes.json                  # Final post-processed boxes
    ├── table/
    │   ├── input_image.png
    │   ├── preprocessed.npy            # (1, 768, 768, 3) uint8 NHWC
    │   ├── preprocess_info.json
    │   ├── seg_mask.npy                # Full mask from DX Engine (768, 768)
    │   ├── seg_mask_before_crop.npy    # Mask before padding removal
    │   ├── hpred.npy                   # Horizontal line mask
    │   ├── vpred.npy                   # Vertical line mask
    │   └── cells.json
    └── ocr/
        ├── input_image.png
        ├── det_boxes.json
        └── rec_results.json
"""

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np


RAPIDDOC_ROOT_DEFAULT = "/home/deepx/Desktop/RapidDoc"
DXNN_MODELS_DIR = "dxnn_models"
ONNX_MODELS_DIR = "onnx_models"


def ensure_rapiddoc_importable(rapiddoc_root: str):
    rapiddoc_root = Path(rapiddoc_root).resolve()
    if str(rapiddoc_root) not in sys.path:
        sys.path.insert(0, str(rapiddoc_root))


def find_model(rapiddoc_root: str, subdir: str, pattern: str) -> str:
    """Find a model file under rapiddoc_root/subdir matching pattern."""
    model_dir = Path(rapiddoc_root) / subdir
    if not model_dir.exists():
        return ""
    for f in model_dir.iterdir():
        if pattern in f.name:
            return str(f)
    return ""


def load_pdf_page(pdf_path: str, page_num: int = 0, dpi: int = 200) -> np.ndarray:
    """Render a PDF page to BGR image, matching Python's page_to_image."""
    from pypdfium2 import PdfDocument
    from rapid_doc.utils.pdf_reader import page_to_image

    pdf_doc = PdfDocument(pdf_path)
    page = pdf_doc[page_num]
    pil_image, scale = page_to_image(page, dpi=dpi)
    img_rgb = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    print(f"PDF page {page_num+1}/{len(pdf_doc)}, DPI={dpi}, scale={scale:.2f}, shape={img_bgr.shape}")
    page.close()
    pdf_doc.close()
    return img_bgr


# ── Layout ─────────────────────────────────────────────────────────────────

def gen_layout_vectors(image: np.ndarray, output_dir: str, rapiddoc_root: str,
                       layout_input_size: int = 640):
    """Generate layout module test vectors with intermediate dumps."""
    outdir = Path(output_dir) / "layout"
    outdir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outdir / "input_image.png"), image)

    from rapid_doc.model.layout.rapid_layout_self.model_handler.pp_doclayout.pre_process import PPPreProcess
    from rapid_doc.model.layout.rapid_layout_self.utils.typings import ModelType, EngineType

    if layout_input_size == 800:
        model_type = ModelType.PP_DOCLAYOUT_PLUS_L
    elif layout_input_size == 480:
        model_type = ModelType.PP_DOCLAYOUT_S
    else:
        model_type = ModelType.PP_DOCLAYOUT_L

    preprocessor = PPPreProcess(
        img_size=(layout_input_size, layout_input_size),
        model_type=model_type,
        engine_type=EngineType.DXENGINE,
    )

    preprocessed = preprocessor(image)
    np.save(str(outdir / "preprocessed.npy"), preprocessed)

    ori_h, ori_w = image.shape[:2]
    scale_h = float(layout_input_size) / ori_h
    scale_w = float(layout_input_size) / ori_w
    scale_factor = np.array([[scale_h, scale_w]], dtype=np.float32)
    np.save(str(outdir / "scale_factor.npy"), scale_factor)

    im_shape = np.array([[layout_input_size, layout_input_size]], dtype=np.float32)

    with open(str(outdir / "preprocess_info.json"), "w") as f:
        json.dump({
            "input_size": layout_input_size,
            "original_h": ori_h,
            "original_w": ori_w,
            "scale_h": scale_h,
            "scale_w": scale_w,
        }, f, indent=2)

    print(f"  Preprocessed: {preprocessed.shape}, dtype={preprocessed.dtype}")
    print(f"  Original: {ori_w}x{ori_h}, scale_factor=({scale_h:.4f}, {scale_w:.4f})")

    dxnn_path = find_model(rapiddoc_root, DXNN_MODELS_DIR, "pp_doclayout_l_part1.dxnn")
    onnx_path = find_model(rapiddoc_root, ONNX_MODELS_DIR, "pp_doclayout_l_part2.onnx")

    if not dxnn_path:
        print("  WARNING: Layout DXNN model not found — skipping inference")
        return
    if not onnx_path:
        print("  WARNING: Layout ONNX sub-model not found — skipping ONNX post-proc")

    try:
        from dx_engine import InferenceEngine
        import onnxruntime as ort

        engine = InferenceEngine(dxnn_path)
        print(f"  DX Engine loaded: {dxnn_path}")

        out = engine.run([preprocessed])
        np.save(str(outdir / "dxnn_out_0.npy"), out[0])
        np.save(str(outdir / "dxnn_out_1.npy"), out[1])
        print(f"  DXNN outputs: {[o.shape for o in out]}")

        if onnx_path:
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            ort_session = ort.InferenceSession(onnx_path, sess_options=sess_options,
                                                providers=['CPUExecutionProvider'])

            ort_feed = {
                "p2o.pd_op.concat.12.0": out[0],
                "p2o.pd_op.layer_norm.20.0": out[1],
                "im_shape": im_shape,
                "scale_factor": scale_factor,
            }
            ort_outputs = ort_session.run(None, ort_feed)
            np.save(str(outdir / "onnx_raw_output.npy"), ort_outputs[0])
            np.save(str(outdir / "onnx_boxes_raw.npy"), ort_outputs[0])
            print(f"  ONNX outputs: {[o.shape for o in ort_outputs]}")

            from rapid_doc.model.layout.rapid_layout_self.model_handler.pp_doclayout.post_process import PPPostProcess

            labels = [
                "paragraph_title", "image", "text", "number", "abstract", "content",
                "figure_title", "formula", "table", "table_title", "reference",
                "doc_title", "footnote", "header", "algorithm", "footer", "seal",
                "chart_title", "chart", "formula_number", "header_image",
                "footer_image", "aside_text",
            ]
            conf_thres = {
                0: 0.3, 1: 0.5, 2: 0.4, 3: 0.5, 4: 0.5, 5: 0.5,
                6: 0.5, 7: 0.3, 8: 0.5, 9: 0.5, 10: 0.5, 11: 0.5,
                12: 0.5, 13: 0.5, 14: 0.5, 15: 0.5, 16: 0.45, 17: 0.5,
                18: 0.5, 19: 0.5, 20: 0.5, 21: 0.5, 22: 0.5,
            }
            postprocessor = PPPostProcess(labels, conf_thres)
            boxes = postprocessor(ort_outputs[0], (ori_w, ori_h))

            with open(str(outdir / "boxes.json"), "w") as f:
                json.dump(boxes, f, indent=2, default=str)
            print(f"  Layout: {len(boxes)} boxes detected")

    except ImportError as e:
        print(f"  WARNING: Import error — {e}")
    except Exception as e:
        print(f"  ERROR: Layout vector generation failed: {e}")
        import traceback; traceback.print_exc()


# ── Table ──────────────────────────────────────────────────────────────────

def gen_table_vectors(image: np.ndarray, output_dir: str, rapiddoc_root: str):
    """Generate table module test vectors with intermediate dumps."""
    outdir = Path(output_dir) / "table"
    outdir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outdir / "input_image.png"), image)

    from rapid_doc.model.table.rapid_table_self.wired_table_rec.utils.utils import resize_with_padding

    padded, scale, pad_top, pad_left, orig_h, orig_w = resize_with_padding(image)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    preprocessed = rgb[None, :]  # (1, 768, 768, 3) uint8 NHWC

    np.save(str(outdir / "preprocessed.npy"), preprocessed)
    with open(str(outdir / "preprocess_info.json"), "w") as f:
        json.dump({
            "scale": float(scale),
            "pad_top": int(pad_top),
            "pad_left": int(pad_left),
            "original_h": int(orig_h),
            "original_w": int(orig_w),
        }, f, indent=2)
    print(f"  Preprocessed: {preprocessed.shape}, scale={scale:.4f}, pad=({pad_top},{pad_left})")

    dxnn_path = find_model(rapiddoc_root, DXNN_MODELS_DIR, "unet.dxnn")
    if not dxnn_path:
        print("  WARNING: Table DXNN model not found — skipping inference")
        return

    try:
        from dx_engine import InferenceEngine

        engine = InferenceEngine(dxnn_path)
        print(f"  DX Engine loaded: {dxnn_path}")

        result = engine.run([preprocessed[None, ...]])[0][0]
        seg_mask = result[0].astype(np.uint8)
        np.save(str(outdir / "seg_mask.npy"), seg_mask)
        np.save(str(outdir / "seg_mask_before_crop.npy"), seg_mask)
        print(f"  Seg mask: shape={seg_mask.shape}, unique={np.unique(seg_mask)}")

        hpred = copy.deepcopy(seg_mask)
        vpred = copy.deepcopy(seg_mask)
        whereh = np.where(hpred == 1)
        wherev = np.where(vpred == 2)
        hpred[wherev] = 0
        vpred[whereh] = 0

        if pad_top > 0 or pad_left > 0:
            h_end = int(orig_h * scale + pad_top)
            w_end = int(orig_w * scale + pad_left)
            hpred_cropped = hpred[pad_top:h_end, pad_left:w_end]
            vpred_cropped = vpred[pad_top:h_end, pad_left:w_end]
        else:
            hpred_cropped = hpred
            vpred_cropped = vpred

        hpred_resized = cv2.resize(hpred_cropped, (orig_w, orig_h))
        vpred_resized = cv2.resize(vpred_cropped, (orig_w, orig_h))

        np.save(str(outdir / "hpred.npy"), hpred_resized)
        np.save(str(outdir / "vpred.npy"), vpred_resized)

        h, w = hpred_cropped.shape
        hors_k = int(math.sqrt(w) * 1.2)
        vert_k = int(math.sqrt(h) * 1.2)

        with open(str(outdir / "morph_info.json"), "w") as f:
            json.dump({
                "hors_k": hors_k,
                "vert_k": vert_k,
                "cropped_h": h,
                "cropped_w": w,
            }, f, indent=2)

        print(f"  hpred/vpred saved, morph kernels: hors_k={hors_k}, vert_k={vert_k}")

        # Cell extraction via full Python postprocess pipeline
        try:
            from rapid_doc.model.table.rapid_table_self.wired_table_rec.table_structure_unet import TSRUnet
            tsr_config = {
                "engine_type": "dxengine",
                "model_path": dxnn_path,
            }
            tsr = TSRUnet(tsr_config)
            polygons, rotated_polygons = tsr(image)
            if polygons is not None and len(polygons) > 0:
                cells = []
                for i, poly in enumerate(polygons):
                    x_coords = poly[:, 0]
                    y_coords = poly[:, 1]
                    cells.append({
                        "bbox": [float(x_coords.min()), float(y_coords.min()),
                                 float(x_coords.max()), float(y_coords.max())],
                        "polygon": poly.tolist(),
                    })
                with open(str(outdir / "cells.json"), "w") as f:
                    json.dump(cells, f, indent=2)
                print(f"  Cells: {len(cells)} cells extracted and saved")
            else:
                print("  No cells detected by TSRUnet")
                with open(str(outdir / "cells.json"), "w") as f:
                    json.dump([], f)
        except Exception as cell_err:
            print(f"  WARNING: Cell extraction failed: {cell_err}")
            import traceback; traceback.print_exc()

    except ImportError as e:
        print(f"  WARNING: Import error — {e}")
    except Exception as e:
        print(f"  ERROR: Table vector generation failed: {e}")
        import traceback; traceback.print_exc()


# ── OCR ────────────────────────────────────────────────────────────────────

def gen_ocr_vectors(image: np.ndarray, output_dir: str, rapiddoc_root: str):
    """Generate OCR module test vectors."""
    outdir = Path(output_dir) / "ocr"
    outdir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outdir / "input_image.png"), image)

    det_model = find_model(rapiddoc_root, DXNN_MODELS_DIR, "ch_PP-OCRv5_server_det.dxnn")
    rec_model = find_model(rapiddoc_root, DXNN_MODELS_DIR, "ch_PP-OCRv5_rec_server_infer.dxnn")

    if not det_model:
        det_model = find_model(rapiddoc_root, DXNN_MODELS_DIR, "det_v5_640_640.dxnn")
    if not rec_model:
        rec_model = find_model(rapiddoc_root, DXNN_MODELS_DIR, "rec_v5_ratio_35.dxnn")

    if not det_model or not rec_model:
        print(f"  WARNING: OCR models not found (det={det_model}, rec={rec_model})")
        return

    try:
        from rapid_doc.model.ocr.dx_ocr import DxOcrModel

        char_dict_path = ""
        for candidate in [
            Path(rapiddoc_root) / "value_compare" / "recognition" / "character_dict_from_onnx.txt",
            Path(rapiddoc_root) / "rapid_doc" / "model" / "ocr" / "ppocr_keys_v1.txt",
        ]:
            if candidate.exists():
                char_dict_path = str(candidate)
                break

        if not char_dict_path:
            try:
                import rapidocr
                pkg_dir = Path(rapidocr.__file__).parent
                possible = list(pkg_dir.rglob("ppocr_keys_v1.txt"))
                if possible:
                    char_dict_path = str(possible[0])
            except ImportError:
                pass

        if not char_dict_path:
            print("  WARNING: Character dict not found")
            return

        ocr_config = {"char_dict_path": char_dict_path}

        ocr = DxOcrModel(
            det_model_path=det_model,
            rec_model_path=rec_model,
            ocr_config=ocr_config,
        )
        print(f"  OCR models loaded: det={det_model}, rec={rec_model}")

        try:
            dt_boxes, rec_res = ocr(image)
        except Exception as full_err:
            print(f"  WARNING: Full OCR pipeline failed ({full_err}), trying det-only...")
            dt_boxes = None
            rec_res = None

            det_result = ocr.text_detector(image)
            if det_result.boxes is not None:
                from rapid_doc.utils.ocr_utils import sorted_boxes
                dt_boxes = sorted_boxes(det_result.boxes)
                rec_res = None

        if dt_boxes is None:
            print("  OCR: No text detected")
            with open(str(outdir / "det_boxes.json"), "w") as f:
                json.dump([], f)
            with open(str(outdir / "rec_results.json"), "w") as f:
                json.dump([], f)
            return

        det_boxes = [b.tolist() if hasattr(b, "tolist") else b for b in dt_boxes]
        with open(str(outdir / "det_boxes.json"), "w") as f:
            json.dump(det_boxes, f, indent=2)

        if rec_res is not None:
            rec_results = [{"text": t, "score": float(s)} for t, s in rec_res]
        else:
            rec_results = []
        with open(str(outdir / "rec_results.json"), "w") as f:
            json.dump(rec_results, f, indent=2)
        print(f"  OCR: {len(det_boxes)} det boxes, {len(rec_results)} rec results")

    except ImportError as e:
        print(f"  WARNING: Import error — {e}")
    except Exception as e:
        print(f"  ERROR: OCR vector generation failed: {e}")
        import traceback; traceback.print_exc()


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate test vectors for C++/Python cross-validation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf", help="Path to test PDF file")
    group.add_argument("--image", help="Path to test image (PNG/JPG)")
    parser.add_argument("--page", type=int, default=0, help="PDF page number (0-based)")
    parser.add_argument("--output", default="test/fixtures", help="Output directory")
    parser.add_argument("--rapiddoc-root", default=RAPIDDOC_ROOT_DEFAULT,
                        help="Path to RapidDoc Python project root")
    parser.add_argument("--layout-input-size", type=int, default=640,
                        help="Layout model input size (640 for _l, 800 for _plus_l)")
    parser.add_argument("--modules", nargs="+", default=["layout", "table", "ocr"],
                        help="Modules to generate vectors for")
    args = parser.parse_args()

    ensure_rapiddoc_importable(args.rapiddoc_root)

    if args.pdf:
        print(f"Loading PDF: {args.pdf}, page {args.page}")
        image = load_pdf_page(args.pdf, page_num=args.page)
    else:
        image = cv2.imread(args.image)
        if image is None:
            print(f"ERROR: Cannot read image: {args.image}")
            sys.exit(1)

    print(f"Input image: {image.shape[1]}x{image.shape[0]}")
    print(f"Output directory: {args.output}")
    print(f"RapidDoc root: {args.rapiddoc_root}")
    print(f"Layout input size: {args.layout_input_size}")
    print()

    if "layout" in args.modules:
        print("[Layout] Generating test vectors...")
        gen_layout_vectors(image, args.output, args.rapiddoc_root, args.layout_input_size)
        print()

    if "table" in args.modules:
        print("[Table] Generating test vectors...")
        gen_table_vectors(image, args.output, args.rapiddoc_root)
        print()

    if "ocr" in args.modules:
        print("[OCR] Generating test vectors...")
        gen_ocr_vectors(image, args.output, args.rapiddoc_root)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
