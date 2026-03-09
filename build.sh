#!/bin/bash
# Build script - Usage: ./build.sh [release|debug] [clean] [test]

set -e

BUILD_TYPE="Release"
CLEAN_BUILD=false
BUILD_TESTS=false

for arg in "$@"; do
    case $arg in
        clean) CLEAN_BUILD=true ;;
        debug) BUILD_TYPE="Debug" ;;
        release) BUILD_TYPE="Release" ;;
        test) BUILD_TESTS=true ;;
        *) echo "Usage: ./build.sh [release|debug] [clean] [test]"; exit 1 ;;
    esac
done

BUILD_DIR="build_${BUILD_TYPE}"

if [ "$CLEAN_BUILD" = true ]; then
    rm -rf ${BUILD_DIR}
fi

mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# Build options
CMAKE_EXTRA_ARGS=""
if [ "${BUILD_OPENCV_FROM_SOURCE}" = "OFF" ]; then
    CMAKE_EXTRA_ARGS="-DBUILD_OPENCV_FROM_SOURCE=OFF"
    echo "Using system OpenCV"
fi

if [ "$BUILD_TESTS" = true ]; then
    CMAKE_EXTRA_ARGS="${CMAKE_EXTRA_ARGS} -DRAPIDDOC_BUILD_TESTS=ON"
    echo "Building with unit tests enabled"
else
    CMAKE_EXTRA_ARGS="${CMAKE_EXTRA_ARGS} -DRAPIDDOC_BUILD_TESTS=OFF"
fi

INSTALL_PREFIX="$(pwd)"

if [ "$BUILD_TYPE" = "Debug" ]; then
    cmake .. -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} ${CMAKE_EXTRA_ARGS}
else
    cmake .. -DCMAKE_BUILD_TYPE=${BUILD_TYPE} -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} -DENABLE_DEBUG_INFO=ON ${CMAKE_EXTRA_ARGS}
fi

make -j$(nproc)

echo "✅ Build complete: ${BUILD_DIR}/bin"

# Run tests if enabled
if [ "$BUILD_TESTS" = true ]; then
    echo ""
    echo "========================================="
    echo "Running unit tests..."
    echo "========================================="
    if command -v ctest &> /dev/null; then
        ctest --output-on-failure --test-dir . -j$(nproc)
    else
        echo "Warning: ctest not found. Running test executables directly..."
        TESTS_FOUND=0
        for test_bin in bin/test_*; do
            if [ -x "$test_bin" ]; then
                echo "--- Running $test_bin ---"
                ./"$test_bin" --gtest_color=yes
                TESTS_FOUND=1
            fi
        done
        if [ "$TESTS_FOUND" -eq 0 ]; then
            echo "Warning: No test executables found in bin/"
        fi
    fi
fi
