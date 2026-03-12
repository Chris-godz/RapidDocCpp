# ========================================
# OpenCV Build Configuration
# ========================================
# Shared between RapidDocCpp and DXNN-OCR-cpp subproject.
# Build from source (submodule) or use system OpenCV.

option(BUILD_OPENCV_FROM_SOURCE "Build OpenCV from source (submodule) instead of using system OpenCV" ON)

if(NOT BUILD_OPENCV_FROM_SOURCE)
    message(STATUS "Using system-installed OpenCV")
    find_package(OpenCV REQUIRED)
    if(OpenCV_FOUND)
        message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    endif()
else()
    message(STATUS "Building OpenCV from source (submodule)")
    set(_rapiddoc_saved_build_tests ${BUILD_TESTS})

    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/3rd-party/opencv/CMakeLists.txt")
        message(FATAL_ERROR "OpenCV submodule not found. Run: git submodule update --init 3rd-party/opencv 3rd-party/opencv_contrib")
    endif()

    # Set all OpenCV options BEFORE add_subdirectory (OpenCV's BUILD_TESTS, not project's)
    set(BUILD_opencv_world OFF CACHE BOOL "" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_PERF_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_apps OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_python2 OFF CACHE BOOL "" FORCE)
    set(BUILD_opencv_python3 OFF CACHE BOOL "" FORCE)

    # Enable only necessary modules
    set(BUILD_LIST "core,imgproc,imgcodecs,highgui" CACHE STRING "" FORCE)

    # Contrib modules (for freetype)
    if(EXISTS "${CMAKE_SOURCE_DIR}/3rd-party/opencv_contrib/modules")
        message(STATUS "OpenCV contrib modules found - enabling freetype")
        set(OPENCV_EXTRA_MODULES_PATH "${CMAKE_SOURCE_DIR}/3rd-party/opencv_contrib/modules" CACHE PATH "" FORCE)
        set(BUILD_opencv_freetype ON CACHE BOOL "" FORCE)
        set(BUILD_LIST "core,imgproc,imgcodecs,highgui,freetype" CACHE STRING "" FORCE)
    endif()

    # Minimal options
    set(WITH_GTK OFF CACHE BOOL "" FORCE)
    set(WITH_QT OFF CACHE BOOL "" FORCE)
    set(WITH_OPENGL OFF CACHE BOOL "" FORCE)
    set(WITH_FFMPEG OFF CACHE BOOL "" FORCE)
    set(WITH_V4L OFF CACHE BOOL "" FORCE)
    set(WITH_1394 OFF CACHE BOOL "" FORCE)
    set(WITH_GSTREAMER OFF CACHE BOOL "" FORCE)

    # Image format support
    set(WITH_JPEG ON CACHE BOOL "" FORCE)
    set(WITH_PNG ON CACHE BOOL "" FORCE)
    set(WITH_TIFF ON CACHE BOOL "" FORCE)
    set(WITH_WEBP ON CACHE BOOL "" FORCE)

    # Performance
    set(ENABLE_FAST_MATH ON CACHE BOOL "" FORCE)
    set(CV_ENABLE_INTRINSICS ON CACHE BOOL "" FORCE)
    set(WITH_IPP OFF CACHE BOOL "" FORCE)
    set(WITH_TBB OFF CACHE BOOL "" FORCE)
    set(WITH_EIGEN OFF CACHE BOOL "" FORCE)

    # Do not use EXCLUDE_FROM_ALL so that opencv targets are built and linked by dependents (e.g. rapid_doc_cli)
    add_subdirectory(${CMAKE_SOURCE_DIR}/3rd-party/opencv ${CMAKE_BINARY_DIR}/opencv_build)

    set(_ocv_build "${CMAKE_BINARY_DIR}/opencv_build")
    # Ensure generated header exists (subproject build can leave it missing)
    file(MAKE_DIRECTORY "${_ocv_build}/opencv2")
    if(NOT EXISTS "${_ocv_build}/opencv2/opencv_modules.hpp")
        file(WRITE "${_ocv_build}/opencv2/opencv_modules.hpp"
            "/* fallback: built modules */\n#pragma once\n"
            "#define HAVE_OPENCV_CORE\n#define HAVE_OPENCV_IMGPROC\n"
            "#define HAVE_OPENCV_IMGCODECS\n#define HAVE_OPENCV_HIGHGUI\n#define HAVE_OPENCV_FREETYPE\n")
    endif()
    # Avoid parallel make race: pre-create object dirs so .d files can be written (make -j)
    file(MAKE_DIRECTORY "${_ocv_build}/modules/imgproc/CMakeFiles/opencv_imgproc.dir/src")
    file(MAKE_DIRECTORY "${_ocv_build}/modules/core/CMakeFiles/opencv_core.dir/src")
    file(MAKE_DIRECTORY "${_ocv_build}/3rdparty/libwebp")

    set(OpenCV_DIR "${_ocv_build}" CACHE PATH "OpenCV build dir" FORCE)

    set(OpenCV_INCLUDE_DIRS
        "${CMAKE_BINARY_DIR}"
        "${CMAKE_BINARY_DIR}/opencv_build"
        "${CMAKE_SOURCE_DIR}/3rd-party/opencv/include"
        "${CMAKE_SOURCE_DIR}/3rd-party/opencv/modules/core/include"
        "${CMAKE_SOURCE_DIR}/3rd-party/opencv/modules/imgproc/include"
        "${CMAKE_SOURCE_DIR}/3rd-party/opencv/modules/imgcodecs/include"
        "${CMAKE_SOURCE_DIR}/3rd-party/opencv/modules/highgui/include"
        "${CMAKE_SOURCE_DIR}/3rd-party/opencv_contrib/modules/freetype/include"
        CACHE INTERNAL "OpenCV include directories"
    )

    set(OpenCV_LIBS opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui opencv_freetype
        CACHE INTERNAL "OpenCV libraries")

    # Single interface target so executables (rapid_doc_cli, tests) get correct link order and deps
    add_library(rapiddoc_opencv_libs INTERFACE)
    target_link_libraries(rapiddoc_opencv_libs INTERFACE
        opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui opencv_freetype)

    # Restore project's BUILD_TESTS so test targets are still built when -DBUILD_TESTS=ON
    set(BUILD_TESTS ${_rapiddoc_saved_build_tests} CACHE BOOL "Build unit tests" FORCE)

    message(STATUS "✓ OpenCV will be built from source")
    message(STATUS "  Modules: ${BUILD_LIST}")
endif()
