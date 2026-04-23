#!/bin/bash
# RapidDocCpp Model Setup
#
# Downloads all required model files from sdk.deepx.ai and copies them
# to the correct directories. No symlinks, no complex logic.
#
# Models downloaded:
#   1. RapidDoc dxnn_models (layout, table) + onnx_models (layout NMS)
#   2. DXNN-OCR-cpp models (detection + recognition)
#
# Usage:  ./setup.sh [--force]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DOWNLOAD_DIR="$SCRIPT_DIR/.download_cache"
FORCE=0

for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        --help|-h) echo "Usage: $0 [--force]"; exit 0 ;;
    esac
done

BASE_URL="https://sdk.deepx.ai"
RAPIDDOC_DXNN="res/assets/dx_RapidDoc/dxnn_models.tar.gz"
RAPIDDOC_ONNX="res/assets/dx_RapidDoc/onnx_models.tar.gz"
OCR_SERVER="res/assets/dx_baidu_PPOCR/dxnn_optimized.tar.gz"

LAYOUT_DIR="$SCRIPT_DIR/engine/model_files/layout"
TABLE_DIR="$SCRIPT_DIR/engine/model_files/table"
OCR_DIR="$SCRIPT_DIR/3rd-party/DXNN-OCR-cpp/engine/model_files/server"

echo "=========================================="
echo " RapidDocCpp Model Setup"
echo "=========================================="

download_and_extract() {
    local url="$BASE_URL/$1"
    local name="$(basename "$1")"
    local dest="$DOWNLOAD_DIR/${name%.tar.gz}"

    if [ -d "$dest" ] && [ "$FORCE" -eq 0 ]; then
        echo "  [SKIP] $name already cached"
        return
    fi

    mkdir -p "$DOWNLOAD_DIR"
    echo "  Downloading $name ..."
    curl -fSL "$url" -o "$DOWNLOAD_DIR/$name"
    rm -rf "$dest"
    mkdir -p "$dest"
    tar xzf "$DOWNLOAD_DIR/$name" -C "$dest" --strip-components=1
    rm -f "$DOWNLOAD_DIR/$name"
    echo "  Extracted to $dest"
}

# --- Step 1: Download RapidDoc DXNN + ONNX models ---
echo ""
echo "[Step 1] RapidDoc models (layout + table) ..."
download_and_extract "$RAPIDDOC_DXNN"
download_and_extract "$RAPIDDOC_ONNX"

DXNN_CACHE="$DOWNLOAD_DIR/dxnn_models"
ONNX_CACHE="$DOWNLOAD_DIR/onnx_models"

mkdir -p "$LAYOUT_DIR" "$TABLE_DIR"

# Remove old symlinks/files and copy fresh
for f in pp_doclayout_l_part1.dxnn; do
    rm -f "$LAYOUT_DIR/$f"
    cp -v "$DXNN_CACHE/$f" "$LAYOUT_DIR/$f"
done
for f in pp_doclayout_l_part2.onnx; do
    rm -f "$LAYOUT_DIR/$f"
    cp -v "$ONNX_CACHE/$f" "$LAYOUT_DIR/$f"
done
rm -f "$TABLE_DIR/unet.dxnn"
cp -v "$DXNN_CACHE/unet.dxnn" "$TABLE_DIR/unet.dxnn"

# --- Step 1b: Download PaddleCls table classifier (paddle_cls.onnx) ---
# Mirrors Python rapid_doc/model/table/rapid_table_self/table_cls/main.py
# which downloads this same file from modelscope. If the download fails (no
# network / mirror unavailable), the C++ TableRecognizer falls back to the
# legacy lineRatio heuristic at runtime.
PADDLE_CLS_URL="https://www.modelscope.cn/models/RapidAI/RapidTable/resolve/master/table_cls/paddle_cls.onnx"
PADDLE_CLS_DEST="$ONNX_CACHE/paddle_cls.onnx"
if [ ! -f "$PADDLE_CLS_DEST" ] || [ "$FORCE" -eq 1 ]; then
    echo ""
    echo "[Step 1b] PaddleCls table classifier (paddle_cls.onnx) ..."
    mkdir -p "$ONNX_CACHE"
    if curl -fSL "$PADDLE_CLS_URL" -o "$PADDLE_CLS_DEST"; then
        echo "  Downloaded to $PADDLE_CLS_DEST"
    else
        echo "  [WARN] paddle_cls.onnx download failed; TableRecognizer will fall back to lineRatio heuristic"
        rm -f "$PADDLE_CLS_DEST"
    fi
else
    echo "  [SKIP] paddle_cls.onnx already cached"
fi

# --- Step 1c: Fetch SLANet+ wireless table structure model (slanet-plus.onnx) ---
# Mirrors Python rapid_doc/model/table/rapid_table_self/models/slanet-plus.onnx
# which is consumed by PPTableStructurer for wireless tables. We prefer the
# bundled RapidDoc workspace copy when present (no network required), then
# fall back to the modelscope mirror. If both fail, the wireless backend is
# disabled at runtime and the pipeline falls back to the legacy unsupported
# placeholder — matching pre-port behavior for wireless crops.
SLANET_DEST="$ONNX_CACHE/slanet-plus.onnx"
SLANET_LOCAL="${SLANET_LOCAL_SRC:-$HOME/Desktop/RapidDoc/rapid_doc/model/table/rapid_table_self/models/slanet-plus.onnx}"
SLANET_URL="https://www.modelscope.cn/models/RapidAI/RapidTable/resolve/v2.0.0/slanet-plus.onnx"
if [ ! -f "$SLANET_DEST" ] || [ "$FORCE" -eq 1 ]; then
    echo ""
    echo "[Step 1c] SLANet+ wireless table structure (slanet-plus.onnx) ..."
    mkdir -p "$ONNX_CACHE"
    if [ -f "$SLANET_LOCAL" ]; then
        cp -v "$SLANET_LOCAL" "$SLANET_DEST"
        echo "  Copied from $SLANET_LOCAL"
    elif curl -fSL "$SLANET_URL" -o "$SLANET_DEST"; then
        echo "  Downloaded to $SLANET_DEST"
    else
        echo "  [WARN] slanet-plus.onnx unavailable; wireless backend disabled"
        rm -f "$SLANET_DEST"
    fi
else
    echo "  [SKIP] slanet-plus.onnx already cached"
fi

# --- Step 2: Download OCR models ---
echo ""
echo "[Step 2] DXNN-OCR-cpp models (detection + recognition) ..."
download_and_extract "$OCR_SERVER"

OCR_CACHE="$DOWNLOAD_DIR/dxnn_optimized"
mkdir -p "$OCR_DIR"

# Detection models — rename to match DXNN-OCR-cpp expected names
for src_name in det_v5_640_640.dxnn det_v5_960_960.dxnn; do
    # e.g. det_v5_640_640.dxnn -> det_v5_640.dxnn
    dst_name="$(echo "$src_name" | sed 's/_[0-9]*\.dxnn$/.dxnn/')"
    if [ -f "$OCR_CACHE/$src_name" ]; then
        rm -f "$OCR_DIR/$dst_name"
        cp -v "$OCR_CACHE/$src_name" "$OCR_DIR/$dst_name"
    elif [ -f "$DXNN_CACHE/$src_name" ]; then
        rm -f "$OCR_DIR/$dst_name"
        cp -v "$DXNN_CACHE/$src_name" "$OCR_DIR/$dst_name"
    fi
done

# Recognition models
for ratio in 3 5 10 15 25 35; do
    fname="rec_v5_ratio_${ratio}.dxnn"
    src=""
    [ -f "$OCR_CACHE/$fname" ] && src="$OCR_CACHE/$fname"
    [ -f "$DXNN_CACHE/$fname" ] && src="$DXNN_CACHE/$fname"
    if [ -n "$src" ]; then
        rm -f "$OCR_DIR/$fname"
        cp -v "$src" "$OCR_DIR/$fname"
    fi
done

# Dictionary (already in repo, just verify)
DICT="$SCRIPT_DIR/3rd-party/DXNN-OCR-cpp/engine/model_files/ppocrv5_dict.txt"
if [ -f "$DICT" ]; then
    echo "  ppocrv5_dict.txt present"
else
    echo "  [WARN] ppocrv5_dict.txt missing"
fi

echo ""
echo "=========================================="
echo " Setup complete!"
echo "=========================================="
echo " Layout : $LAYOUT_DIR/"
echo " Table  : $TABLE_DIR/"
echo " OCR    : $OCR_DIR/"
echo ""
echo " Next: cmake -B build_Release -DBUILD_TESTS=ON -DBUILD_SERVER=OFF -DBUILD_OPENCV_FROM_SOURCE=OFF && cmake --build build_Release"
echo "=========================================="
