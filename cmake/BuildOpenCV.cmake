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

    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/3rd-party/opencv/CMakeLists.txt")
        message(FATAL_ERROR "OpenCV submodule not found. Run: git submodule update --init 3rd-party/opencv 3rd-party/opencv_contrib")
    endif()

    # Set all OpenCV options BEFORE add_subdirectory
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

    add_subdirectory(${CMAKE_SOURCE_DIR}/3rd-party/opencv ${CMAKE_BINARY_DIR}/opencv_build EXCLUDE_FROM_ALL)

    set(OpenCV_DIR "${CMAKE_BINARY_DIR}/opencv_build" CACHE PATH "OpenCV build dir" FORCE)

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

    message(STATUS "✓ OpenCV will be built from source")
    message(STATUS "  Modules: ${BUILD_LIST}")
endif()
