#!/bin/bash
# Setup script - Download model files for RapidDoc C++
# Downloads DXNN models and ONNX models from sdk.deepx.ai 
# Also downloads ONNX baseline models from ModelScope for CPU testing/fallback
#
# Usage: ./setup.sh [--force] [--onnx-only] [--dxnn-only] [--help]
#
# Model sources (same as Python RapidDoc project):
#   DXNN: https://sdk.deepx.ai/res/assets/dx_RapidDoc/dxnn_models.tar.gz
#   ONNX: https://sdk.deepx.ai/res/assets/dx_RapidDoc/onnx_models.tar.gz

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ========================================
# Color helpers (standalone, no dependency)
# ========================================
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ========================================
# Configuration
# ========================================

# DEEPX SDK download base
SDK_BASE_URL="https://sdk.deepx.ai/"

RAPIDDOC_DXNN_PATH="res/assets/dx_RapidDoc/dxnn_models.tar.gz"
RAPIDDOC_ONNX_PATH="res/assets/dx_RapidDoc/onnx_models.tar.gz"

# ModelScope ONNX baseline URLs (for individual downloads / CPU fallback)
# Layout models
MODELSCOPE_RAPIDDOC="https://www.modelscope.cn/models/RapidAI/RapidDoc/resolve/v1.0.0"
LAYOUT_PLUS_L_ONNX_URL="${MODELSCOPE_RAPIDDOC}/layout/PP-DocLayout_plus-L/pp_doclayout_plus_l.onnx"
LAYOUT_L_ONNX_URL="${MODELSCOPE_RAPIDDOC}/layout/PP-DocLayout-L/pp_doclayout_l.onnx"

# Table models
TABLE_SLANEXT_WIRED_ONNX_URL="${MODELSCOPE_RAPIDDOC}/table/SLANeXt_wired/slanext_wired.onnx"
TABLE_SLANEXT_WIRELESS_ONNX_URL="${MODELSCOPE_RAPIDDOC}/table/SLANeXt_wireless/slanext_wireless.onnx"
TABLE_RT_DETR_WIRED_ONNX_URL="${MODELSCOPE_RAPIDDOC}/table/RT-DETR-L_wired_table_cell_det/rt_detr_l_wired_table_cell_det.onnx"
TABLE_RT_DETR_WIRELESS_ONNX_URL="${MODELSCOPE_RAPIDDOC}/table/RT-DETR-L_wireless_table_cell_det/rt_detr_l_wireless_table_cell_det.onnx"

MODELSCOPE_RAPIDTABLE="https://www.modelscope.cn/models/RapidAI/RapidTable/resolve/v2.0.0"
TABLE_UNET_ONNX_URL="${MODELSCOPE_RAPIDTABLE}/../master/unet.onnx"
TABLE_SLANET_PLUS_ONNX_URL="${MODELSCOPE_RAPIDTABLE}/slanet-plus.onnx"
TABLE_CLS_ONNX_URL="https://www.modelscope.cn/models/RapidAI/RapidTable/resolve/master/table_cls/paddle_cls.onnx"

# Dictionary files (from PaddleOCR)
PADDLEOCR_DICT_BASE="https://ghproxy.net/https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/dict"
TABLE_SLANET_DICT_URL="${PADDLEOCR_DICT_BASE}/table_structure_dict.txt"
TABLE_SLANET_DICT_CH_URL="${PADDLEOCR_DICT_BASE}/table_structure_dict_ch.txt"
FORMULA_DICT_URL="${PADDLEOCR_DICT_BASE}/latex_symbol_dict.txt"

# Directories
MODELS_DIR="${SCRIPT_DIR}/engine/model_files"
LAYOUT_DIR="${MODELS_DIR}/layout"
TABLE_DIR="${MODELS_DIR}/table"
FORMULA_DIR="${MODELS_DIR}/formula"
DOWNLOAD_DIR="${SCRIPT_DIR}/engine/models"

# OCR submodule
OCR_SUBMODULE="${SCRIPT_DIR}/3rd-party/DXNN-OCR-cpp"

USE_FORCE=0
DXNN_ONLY=0
ONNX_ONLY=0

# ========================================
# Help
# ========================================
show_help() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo ""
    echo "Download model files required by RapidDoc C++"
    echo ""
    echo "Options:"
    echo "  --force       Overwrite existing model files"
    echo "  --dxnn-only   Download only DXNN models (requires DEEPX NPU)"
    echo "  --onnx-only   Download only ONNX baseline models (CPU testing)"
    echo "  --help        Show this help"
    echo ""
    echo "DXNN Models (from dxnn_models.tar.gz):"
    echo "  Layout:"
    echo "    pp_doclayout_l_part1.dxnn        # Layout backbone (NPU, PP-DocLayout-L)"
    echo "  Table:"
    echo "    unet.dxnn                        # Wired table UNet segmentation (NPU)"
    echo "  OCR (from DXNN-OCR-cpp submodule):"
    echo "    det_v5_640.dxnn, det_v5_960.dxnn # OCR detection multi-model"
    echo "    rec_v5_ratio_{3,5,10,15,25,35}.dxnn  # OCR recognition multi-model"
    echo ""
    echo "ONNX Models (from onnx_models.tar.gz):"
    echo "  Layout:"
    echo "    pp_doclayout_l_part2.onnx        # Layout NMS post-processing (CPU)"
    echo "    pp_doclayout_l.onnx              # Layout full ONNX baseline (CPU)"
    echo "    pp_doclayout_plus_l.onnx         # Layout Plus-L ONNX baseline (CPU)"
    echo "  Table:"
    echo "    unet.onnx                        # Table UNet ONNX baseline (CPU)"
    echo "    slanext_wired.onnx               # Wired table SLANeXt (CPU)"
    echo "    slanext_wireless.onnx            # Wireless table SLANeXt (CPU)"
    echo "    slanet-plus.onnx                 # Wireless table SLANet-Plus (CPU)"
    echo "    paddle_cls.onnx                  # Table type classifier (CPU)"
    echo ""
    echo "Model directory layout after download:"
    echo "  engine/model_files/"
    echo "    layout/"
    echo "      pp_doclayout_l_part1.dxnn      # Layout backbone (NPU) — 23 categories"
    echo "      pp_doclayout_l_part2.onnx      # Layout NMS post-processing (CPU)"
    echo "      pp_doclayout_l.onnx            # Layout ONNX baseline (CPU, testing)"
    echo "      pp_doclayout_plus_l.onnx       # Layout Plus-L ONNX (CPU, 20 categories)"
    echo "    table/"
    echo "      unet.dxnn                      # Wired table UNet (NPU)"
    echo "      unet.onnx                      # Table UNet ONNX baseline (CPU)"
    echo "      slanext_wired.onnx             # SLANeXt wired table (CPU)"
    echo "      slanext_wireless.onnx          # SLANeXt wireless table (CPU)"
    echo "      slanet-plus.onnx               # SLANet-Plus wireless table (CPU)"
    echo "      paddle_cls.onnx                # Table type classifier (CPU)"
    echo "      slanet_dict.txt                # SLANet dictionary (English)"
    echo "      slanet_dict_ch.txt            # SLANet dictionary (Chinese)"
    echo "    formula/"
    echo "      pp_formulanet_plus_m.onnx     # Formula recognition (NPU)"
    echo "      formula_dict.txt               # Formula dictionary"
    echo "  3rd-party/DXNN-OCR-cpp/engine/model_files/"
    echo "    server/                          # OCR server models (DXNN-OCR-cpp)"
    echo "    mobile/                          # OCR mobile models (DXNN-OCR-cpp)"
    echo ""
    echo "Layout categories (PP-DocLayout-L DXEngine, 23 classes — indices 0-22):"
    echo "  0: paragraph_title  1: image         2: text          3: number"
    echo "  4: abstract         5: content       6: figure_title  7: formula"
    echo "  8: table            9: table_title  10: reference    11: doc_title"
    echo " 12: footnote        13: header       14: algorithm    15: footer"
    echo " 16: seal            17: chart_title  18: chart        19: formula_number"
    echo " 20: header_image    21: footer_image 22: aside_text"
    echo ""
    echo "Layout categories (PP-DocLayout-Plus-L ONNX, 20 classes — indices 0-19):"
    echo "  0: paragraph_title  1: image         2: text          3: number"
    echo "  4: abstract         5: content       6: figure_title  7: formula"
    echo "  8: table            9: reference    10: doc_title    11: footnote"
    echo " 12: header          13: algorithm    14: footer       15: seal"
    echo " 16: chart           17: formula_number 18: aside_text 19: reference_content"
    exit 0
}

# ========================================
# Utility functions
# ========================================

# Download a single file with wget/curl
download_file() {
    local url="$1"
    local dest="$2"
    local desc="$3"

    if [ -f "$dest" ] && [ "$USE_FORCE" -eq 0 ]; then
        info "Already exists: $dest (skip, use --force to overwrite)"
        return 0
    fi

    mkdir -p "$(dirname "$dest")"
    info "Downloading: $desc"
    info "  URL:  $url"
    info "  Dest: $dest"

    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$dest" "$url" || {
            error "Download failed: $url"
            rm -f "$dest"
            return 1
        }
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$dest" "$url" || {
            error "Download failed: $url"
            rm -f "$dest"
            return 1
        }
    else
        error "Neither wget nor curl found. Please install one."
        return 1
    fi

    info "Downloaded: $(du -h "$dest" | cut -f1) → $dest"
}

# Download and extract tar.gz via DXNN-OCR-cpp get_resource.sh
download_sdk_resource() {
    local src_path="$1"
    local output_dir="$2"
    local desc="$3"

    if [ -d "$output_dir" ] && [ "$(ls -A "$output_dir" 2>/dev/null)" ] && [ "$USE_FORCE" -eq 0 ]; then
        info "Already exists: $output_dir (skip)"
        return 0
    fi

    local get_res_script="${OCR_SUBMODULE}/scripts/get_resource.sh"
    if [ ! -f "$get_res_script" ]; then
        warn "get_resource.sh not found at $get_res_script"
        warn "Falling back to direct wget..."
        local url="${SDK_BASE_URL}${src_path}"
        local archive="${DOWNLOAD_DIR}/$(basename "$src_path")"
        download_file "$url" "$archive" "$desc" || return 1
        mkdir -p "$output_dir"
        tar -xzf "$archive" -C "$output_dir" --strip-components=1 2>/dev/null || \
            tar -xzf "$archive" -C "$output_dir" 2>/dev/null
        return $?
    fi

    local force_arg=""
    [ "$USE_FORCE" -eq 1 ] && force_arg="--force"

    info "Downloading via get_resource.sh: $desc"
    bash "$get_res_script" \
        --src_path="$src_path" \
        --output="$output_dir" \
        $force_arg \
        --extract || {
        error "SDK resource download failed: $src_path"
        return 1
    }
}

# ========================================
# Parse arguments
# ========================================
for arg in "$@"; do
    case "$arg" in
        --force)     USE_FORCE=1 ;;
        --dxnn-only) DXNN_ONLY=1 ;;
        --onnx-only) ONNX_ONLY=1 ;;
        --help)      show_help ;;
        *)
            error "Unknown option: $arg"
            show_help
            ;;
    esac
done

# ========================================
# Main
# ========================================
echo "=========================================="
echo " RapidDoc C++ — Model Setup"
echo "=========================================="

# 0. Ensure git submodules are initialized
if [ ! -d "${OCR_SUBMODULE}/.git" ] && [ ! -f "${OCR_SUBMODULE}/.git" ]; then
    info "Initializing git submodules..."
    cd "$SCRIPT_DIR" && git submodule update --init --recursive
    info "✅ Submodules initialized"
else
    info "✅ Submodules already initialized"
fi

# Create model directories
mkdir -p "$LAYOUT_DIR" "$TABLE_DIR" "$FORMULA_DIR" "$DOWNLOAD_DIR"

# ========================================
# 1. DXNN Models (from sdk.deepx.ai)
# ========================================
if [ "$ONNX_ONLY" -eq 0 ]; then
    echo ""
    info "=== Step 1a: DXNN Models (sdk.deepx.ai/res/assets/dx_RapidDoc/dxnn_models.tar.gz) ==="

    download_sdk_resource \
        "$RAPIDDOC_DXNN_PATH" \
        "${DOWNLOAD_DIR}/dxnn_models" \
        "RapidDoc DXNN models (same as Python project)"

    # Copy individual model files to expected locations
    if [ -d "${DOWNLOAD_DIR}/dxnn_models" ]; then
        copy_model() {
            local filename="$1"
            local dest_dir="$2"
            local found_file=$(find "${DOWNLOAD_DIR}/dxnn_models" -name "$filename" | head -n 1)
            if [ -n "$found_file" ]; then
                cp "$found_file" "$dest_dir/"
                info "  Copied: $filename → $dest_dir/"
            else
                warn "  Missing model in archive: $filename"
            fi
        }

        # Layout DXNN model (PP-DocLayout-L split model)
        copy_model "pp_doclayout_l_part1.dxnn" "$LAYOUT_DIR"

        # Table DXNN model (UNet for wired table segmentation)
        copy_model "unet.dxnn" "$TABLE_DIR"

        # Formula DXNN models (if available in archive)
        copy_model "pp_formulanet_plus_l.dxnn" "$FORMULA_DIR"
        copy_model "pp_formulanet_plus_m.dxnn" "$FORMULA_DIR"

        info "✅ DXNN models copied to engine/model_files/"
    fi

    # OCR models via DXNN-OCR-cpp's own setup
    echo ""
    info "=== Step 1b: OCR DXNN Models ==="
    if [ -f "${OCR_SUBMODULE}/setup.sh" ]; then
        local_force=""
        [ "$USE_FORCE" -eq 1 ] && local_force="--force"
        info "Running DXNN-OCR-cpp/setup.sh ..."
        pushd "$OCR_SUBMODULE" > /dev/null
        bash setup.sh $local_force || warn "OCR model download had issues (may need sudo)"
        popd > /dev/null
        info "✅ OCR models downloaded (server + mobile)"
        info "   Server models: det_v5_{640,960}.dxnn, rec_v5_ratio_{3,5,10,15,25,35}.dxnn"
        info "   Mobile models: det_mobile_{640,960}.dxnn, rec_mobile_ratio_{3,5,10,15,25,35}.dxnn"
    else
        warn "DXNN-OCR-cpp/setup.sh not found — please download OCR models manually"
    fi
fi

# ========================================
# 2. ONNX Models 
# ========================================
if [ "$DXNN_ONLY" -eq 0 ]; then
    echo ""
    info "=== Step 2a: ONNX Models (sdk.deepx.ai/res/assets/dx_RapidDoc/onnx_models.tar.gz) ==="

    download_sdk_resource \
        "$RAPIDDOC_ONNX_PATH" \
        "${DOWNLOAD_DIR}/onnx_models" \
        "RapidDoc ONNX models (same as Python project)"

    # Copy ONNX models from archive to expected locations
    if [ -d "${DOWNLOAD_DIR}/onnx_models" ]; then
        copy_onnx_model() {
            local filename="$1"
            local dest_dir="$2"
            local found_file=$(find "${DOWNLOAD_DIR}/onnx_models" -name "$filename" | head -n 1)
            if [ -n "$found_file" ]; then
                cp "$found_file" "$dest_dir/"
                info "  Copied: $filename → $dest_dir/"
            else
                warn "  Not found in archive: $filename (will try ModelScope fallback)"
            fi
        }

        # Layout ONNX post-processing model (for DXNN split model pipeline)
        # Python project uses: onnx_models/pp_doclayout_l_part2.onnx
        copy_onnx_model "pp_doclayout_l_part2.onnx" "$LAYOUT_DIR"

        # Layout ONNX full model (CPU baseline)
        copy_onnx_model "pp_doclayout_l.onnx" "$LAYOUT_DIR"

        # Table ONNX models
        copy_onnx_model "unet.onnx" "$TABLE_DIR"
        copy_onnx_model "slanext_wired.onnx" "$TABLE_DIR"
        copy_onnx_model "slanext_wireless.onnx" "$TABLE_DIR"
        copy_onnx_model "slanet-plus.onnx" "$TABLE_DIR"

        # Formula ONNX models
        copy_onnx_model "pp_formulanet_plus_s.onnx" "$FORMULA_DIR"
        copy_onnx_model "pp_formulanet_plus_m.onnx" "$FORMULA_DIR"
        copy_onnx_model "pp_formulanet_plus_l.onnx" "$FORMULA_DIR"

        # OCR ONNX models (server)
        mkdir -p "${DOWNLOAD_DIR}/ocr_onnx"
        copy_onnx_model "ch_PP-OCRv5_server_det.onnx" "${DOWNLOAD_DIR}/ocr_onnx"
        copy_onnx_model "ch_PP-OCRv5_rec_server_infer.onnx" "${DOWNLOAD_DIR}/ocr_onnx"

        info "✅ ONNX models from archive copied to engine/model_files/"
    fi

    # Fallback: download individual ONNX models from ModelScope if not found in archive
    echo ""
    info "=== Step 2b: ONNX Baseline Models (ModelScope fallback) ==="

    # Layout Plus-L ONNX (20-category model, different from DXEngine's L model)
    download_file \
        "$LAYOUT_PLUS_L_ONNX_URL" \
        "${LAYOUT_DIR}/pp_doclayout_plus_l.onnx" \
        "Layout PP-DocLayout-Plus-L ONNX (20 categories)"

    # Layout L ONNX (23-category model, same categories as DXEngine backbone)
    download_file \
        "$LAYOUT_L_ONNX_URL" \
        "${LAYOUT_DIR}/pp_doclayout_l.onnx" \
        "Layout PP-DocLayout-L ONNX (23 categories)"

    # Table models
    download_file \
        "$TABLE_SLANEXT_WIRED_ONNX_URL" \
        "${TABLE_DIR}/slanext_wired.onnx" \
        "Table SLANeXt wired ONNX"

    download_file \
        "$TABLE_SLANEXT_WIRELESS_ONNX_URL" \
        "${TABLE_DIR}/slanext_wireless.onnx" \
        "Table SLANeXt wireless ONNX"

    download_file \
        "$TABLE_UNET_ONNX_URL" \
        "${TABLE_DIR}/unet.onnx" \
        "Table UNet ONNX"

    download_file \
        "$TABLE_SLANET_PLUS_ONNX_URL" \
        "${TABLE_DIR}/slanet-plus.onnx" \
        "Table SLANet-Plus ONNX (wireless)"

    download_file \
        "$TABLE_CLS_ONNX_URL" \
        "${TABLE_DIR}/paddle_cls.onnx" \
        "Table type classifier (wired/wireless)"

    # RT-DETR cell detection models (used with SLANeXt mode)
    download_file \
        "$TABLE_RT_DETR_WIRED_ONNX_URL" \
        "${TABLE_DIR}/rt_detr_l_wired_table_cell_det.onnx" \
        "RT-DETR wired table cell detection ONNX"

    download_file \
        "$TABLE_RT_DETR_WIRELESS_ONNX_URL" \
        "${TABLE_DIR}/rt_detr_l_wireless_table_cell_det.onnx" \
        "RT-DETR wireless table cell detection ONNX"

    # Dictionary files for table and formula recognition
    echo ""
    info "=== Dictionary files ==="
    download_file \
        "$TABLE_SLANET_DICT_URL" \
        "${TABLE_DIR}/slanet_dict.txt" \
        "SLANet dictionary (English)"

    download_file \
        "$TABLE_SLANET_DICT_CH_URL" \
        "${TABLE_DIR}/slanet_dict_ch.txt" \
        "SLANet dictionary (Chinese)"

    download_file \
        "$FORMULA_DICT_URL" \
        "${FORMULA_DIR}/formula_dict.txt" \
        "Formula recognition dictionary"

    info "✅ ONNX baseline models downloaded"
fi

# ========================================
# 3. Verify
# ========================================
echo ""
info "=== Model File Summary ==="
echo ""
info "Layout models:"
find "$LAYOUT_DIR" -type f \( -name "*.dxnn" -o -name "*.onnx" \) \
    -exec ls -lh {} \; 2>/dev/null | while read -r line; do echo "  $line"; done

echo ""
info "Table models:"
find "$TABLE_DIR" -type f \( -name "*.dxnn" -o -name "*.onnx" \) \
    -exec ls -lh {} \; 2>/dev/null | while read -r line; do echo "  $line"; done

echo ""
info "Formula models:"
find "$FORMULA_DIR" -type f \( -name "*.dxnn" -o -name "*.onnx" \) \
    -exec ls -lh {} \; 2>/dev/null | while read -r line; do echo "  $line"; done

echo ""
info "OCR models (DXNN-OCR-cpp):"
find "${OCR_SUBMODULE}/engine/model_files" -type f \( -name "*.dxnn" -o -name "*.txt" \) \
    -exec ls -lh {} \; 2>/dev/null | while read -r line; do echo "  $line"; done

echo ""
echo "=========================================="
info "Model setup complete!"
echo ""
info "Python RapidDoc consistency notes:"
info "  Layout DXNN:  pp_doclayout_l_part1.dxnn (backbone, 23 classes)"
info "  Layout ONNX:  pp_doclayout_l_part2.onnx (NMS post-processing)"
info "  Table DXNN:   unet.dxnn (wired table segmentation)"
info "  OCR DXNN:     server/{det_v5_*.dxnn, rec_v5_ratio_*.dxnn}"
info "  See --help for full category list"