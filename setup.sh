#!/bin/bash
# Setup script - Download model files for RapidDoc C++
# Usage: ./setup.sh [--dest=<path>]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model download URL (TODO: Update with actual model URLs)
BASE_URL="https://sdk.deepx.ai/"
LAYOUT_MODEL_PATH="res/assets/rapid_doc/layout_det.tar.gz"
TABLE_MODEL_PATH="res/assets/rapid_doc/table_unet.tar.gz"

OUTPUT_DIR="$SCRIPT_DIR/engine/models"
TARGET_DIR="$SCRIPT_DIR/engine/model_files"

# Function to display help
show_help() {
    echo "Usage: $(basename "$0") [OPTIONS]"
    echo "Options:"
    echo "  [--dest=<path>]            Destination path for model files (default: $OUTPUT_DIR)"
    echo "  [--help]                   Show this help message"
    exit 0
}

# Parse arguments
for i in "$@"; do
    case "$i" in
        --dest=*)
            OUTPUT_DIR="${i#*=}"
            ;;
        --help)
            show_help
            ;;
        *)
            echo "Unknown option: $i"
            show_help
            ;;
    esac
done

echo "=========================================="
echo "RapidDoc Model Setup"
echo "=========================================="

# Create directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TARGET_DIR"

# TODO: Download models from remote server
# For now, show instructions
echo "Model download not yet implemented."
echo ""
echo "Please manually place model files in: $TARGET_DIR"
echo ""
echo "Required models:"
echo "  - layout_det.dxnn      (Layout detection)"
echo "  - layout_nms.onnx      (NMS post-processing)"
echo "  - table_unet.dxnn      (Wired table recognition)"
echo ""
echo "OCR models should be placed in:"
echo "  3rd-party/DXNN-OCR-cpp/engine/model_files/"
echo ""
echo "=========================================="

# Initialize submodules if not done
if [ ! -d "3rd-party/DXNN-OCR-cpp/.git" ]; then
    echo "Initializing git submodules..."
    git submodule update --init --recursive
    echo "✅ Submodules initialized"
else
    echo "✅ Submodules already initialized"
fi

echo ""
echo "Next steps:"
echo "  1. Place model files in engine/model_files/"
echo "  2. Source DEEPX environment: source /path/to/dx_rt/set_env.sh"
echo "  3. Source project environment: source set_env.sh"
echo "  4. Build the project: ./build.sh"
echo "=========================================="
