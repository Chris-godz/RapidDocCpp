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
