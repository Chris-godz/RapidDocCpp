#!/usr/bin/env bash
#
# End-to-end cross-validation test pipeline.
#
# Steps:
#   1. Generate Python fixtures (if --gen-fixtures is passed or fixtures are missing)
#   2. CMake configure + build (Release + tests)
#   3. Run rapiddoc_tests (metric unit tests)
#   4. Run rapiddoc_cross_tests (C++ vs Python cross-validation)
#
# Prerequisites:
#   - RapidDoc Python project with models downloaded (setup.sh)
#   - DeepX NPU hardware + DXRT SDK installed
#   - System OpenCV, ONNX Runtime
#
# Usage:
#   ./tools/run_e2e_test.sh                     # build & test only
#   ./tools/run_e2e_test.sh --gen-fixtures       # regenerate Python fixtures first
#   ./tools/run_e2e_test.sh --gen-fixtures --pdf /path/to/test.pdf

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RAPIDDOC_ROOT="${RAPIDDOC_ROOT:-/home/deepx/Desktop/RapidDoc}"
BUILD_DIR="$PROJECT_ROOT/build_Release"
FIXTURE_DIR="$PROJECT_ROOT/test/fixtures"

GEN_FIXTURES=false
PDF_PATH=""
IMAGE_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gen-fixtures) GEN_FIXTURES=true; shift ;;
        --pdf)          PDF_PATH="$2"; GEN_FIXTURES=true; shift 2 ;;
        --image)        IMAGE_PATH="$2"; GEN_FIXTURES=true; shift 2 ;;
        --rapiddoc)     RAPIDDOC_ROOT="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--gen-fixtures] [--pdf <path>] [--image <path>] [--rapiddoc <path>]"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Auto-detect if fixtures are missing
if [[ ! -f "$FIXTURE_DIR/layout/preprocessed.npy" ]]; then
    echo "[INFO] Fixtures not found — enabling generation."
    GEN_FIXTURES=true
fi

echo "========================================"
echo " RapidDocCpp End-to-End Test"
echo "========================================"
echo " Project root : $PROJECT_ROOT"
echo " RapidDoc root: $RAPIDDOC_ROOT"
echo " Build dir    : $BUILD_DIR"
echo " Gen fixtures : $GEN_FIXTURES"
echo "========================================"
echo

# Step 1: Generate fixtures
if [[ "$GEN_FIXTURES" == "true" ]]; then
    echo "[Step 1] Generating Python fixtures..."
    export PYTHONPATH="$RAPIDDOC_ROOT:${PYTHONPATH:-}"

    GEN_ARGS="--output $FIXTURE_DIR --rapiddoc-root $RAPIDDOC_ROOT"
    if [[ -n "$PDF_PATH" ]]; then
        GEN_ARGS="--pdf $PDF_PATH $GEN_ARGS"
    elif [[ -n "$IMAGE_PATH" ]]; then
        GEN_ARGS="--image $IMAGE_PATH $GEN_ARGS"
    elif [[ -f "$FIXTURE_DIR/layout/input_image.png" ]]; then
        GEN_ARGS="--image $FIXTURE_DIR/layout/input_image.png $GEN_ARGS"
    else
        echo "[ERROR] No input file specified and no existing input_image.png found."
        echo "        Use --pdf or --image to provide test input."
        exit 1
    fi

    python3 "$PROJECT_ROOT/tools/gen_test_vectors.py" $GEN_ARGS
    echo
fi

# Step 2: CMake configure + build
echo "[Step 2] Building C++ project..."
cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTS=ON \
    -DBUILD_OPENCV_FROM_SOURCE=OFF \
    2>&1 | tail -5

cmake --build "$BUILD_DIR" -j"$(nproc)" 2>&1 | tail -10
echo

# Step 3: Run metric unit tests
echo "[Step 3] Running metric unit tests..."
"$BUILD_DIR/test/rapiddoc_tests"
echo

# Step 4: Run cross-validation tests
echo "[Step 4] Running cross-validation tests..."
"$BUILD_DIR/test/rapiddoc_cross_tests" --gtest_print_time=1
echo

echo "========================================"
echo " All tests completed successfully!"
echo "========================================"
