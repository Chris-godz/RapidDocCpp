cmake_minimum_required(VERSION 3.14)
project(RapidDocCpp VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ========================================
# Build Options
# ========================================
option(BUILD_CLI "Build CLI application" ON)
option(BUILD_SERVER "Build HTTP API server" ON)
option(BUILD_TESTS "Build unit tests" OFF)
option(BUILD_OPENCV_FROM_SOURCE "Build OpenCV from source (submodule)" ON)

# Set default build type to Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type (default: Release)" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/release" CACHE PATH "Default install path" FORCE)
endif()

# ========================================
# Log Level Configuration
# ========================================
# Options: LOG_LEVEL_TRACE, LOG_LEVEL_DEBUG, LOG_LEVEL_INFO, LOG_LEVEL_WARN, LOG_LEVEL_ERROR, LOG_LEVEL_OFF
# Usage: cmake -DLOG_LEVEL=LOG_LEVEL_WARN ..
if(DEFINED LOG_LEVEL)
    message(STATUS "Log level set to: ${LOG_LEVEL}")
    add_compile_definitions(LOG_LEVEL=${LOG_LEVEL})
else()
    message(STATUS "Log level: Auto (DEBUG in Debug build, INFO in Release build)")
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# ========================================
# Find DXRT (DeepX Runtime)
# ========================================
include(dx_func)

# PROJECT_ROOT_DIR is set per-target (app, server, test) to avoid redefinition with DXNN-OCR-cpp
add_compile_options(-W -Wall -pthread -fPIC)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    add_compile_options(-fstrict-volatile-bitfields)
endif()

# ========================================
# Debug/Performance Profiling Configuration
# ========================================
if(CMAKE_BUILD_TYPE MATCHES "Debug")
    message(STATUS "Debug mode: Enabling debug symbols and profiling support")
    add_compile_options(-g3 -ggdb -rdynamic -fno-omit-frame-pointer)
    add_compile_options(-O1)
elseif(ENABLE_DEBUG_INFO)
    message(STATUS "Debug info enabled for profiling")
    add_compile_options(-g -rdynamic -fno-omit-frame-pointer)
endif()

# ========================================
# OpenCV Configuration (shared with DXNN-OCR-cpp)
# ========================================
include(cmake/BuildOpenCV.cmake)

# ========================================
# ONNX Runtime (for Layout post-processing)
# ========================================
include(cmake/FindONNXRuntime.cmake)

# ========================================
# 3rd-party Libraries
# ========================================
add_subdirectory(3rd-party/json)
add_subdirectory(3rd-party/spdlog)
# Pre-create object dirs to avoid parallel make race (opening dependency file ... No such file or directory)
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/3rd-party/spdlog/CMakeFiles/spdlog.dir/src")

# Suppress warnings from 3rd-party libraries
set(CMAKE_CXX_FLAGS_BACKUP "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

set(CLIPPER2_TESTS OFF CACHE BOOL "" FORCE)
set(CLIPPER2_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(3rd-party/clipper2/CPP)
# Pre-create object dirs for Clipper2 utils (parallel make race)
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/3rd-party/clipper2/CPP/CMakeFiles/Clipper2utils.dir/Utils")
file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/3rd-party/clipper2/CPP/CMakeFiles/Clipper2Zutils.dir/Utils")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BACKUP}")

# ========================================
# DXNN-OCR-cpp as Subproject
# ========================================
set(DXNN_OCR_AS_SUBPROJECT ON CACHE BOOL "" FORCE)
if(EXISTS "${CMAKE_SOURCE_DIR}/3rd-party/DXNN-OCR-cpp/CMakeLists.txt"
   AND EXISTS "${CMAKE_SOURCE_DIR}/3rd-party/DXNN-OCR-cpp/src/pipeline/ocr_pipeline.cpp")
    # DXNN uses parent's OpenCV (from source or system) via its BuildOpenCV when DXNN_OCR_AS_SUBPROJECT
    set(DXOCR_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(BUILD_OCR_SERVER OFF CACHE BOOL "" FORCE)
    set(BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
    add_subdirectory(3rd-party/DXNN-OCR-cpp)
    set(HAS_DXNN_OCR ON)
    # Pre-create object dirs to avoid parallel make race (opening dependency file ... No such file or directory)
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/3rd-party/DXNN-OCR-cpp/src/detection/CMakeFiles/ocr_detection.dir")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/3rd-party/DXNN-OCR-cpp/src/recognition/CMakeFiles/ocr_recognition.dir")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/3rd-party/DXNN-OCR-cpp/src/classification/CMakeFiles/ocr_classification.dir")
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/3rd-party/DXNN-OCR-cpp/src/preprocessing/CMakeFiles/ocr_preprocessing.dir")
else()
    message(WARNING "DXNN-OCR-cpp submodule not initialized — skipping OCR pipeline")
    set(HAS_DXNN_OCR OFF)
endif()

# ========================================
# Global Include Directories
# ========================================
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/3rd-party/json/include)
include_directories(${CMAKE_SOURCE_DIR}/3rd-party/spdlog/include)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/3rd-party/clipper2/CPP/Clipper2Lib/include)

if(HAS_DXNN_OCR)
    include_directories(${CMAKE_SOURCE_DIR}/3rd-party/DXNN-OCR-cpp/include)
endif()

# ========================================
# RapidDocCpp Libraries
# ========================================
add_subdirectory(src/common)
add_subdirectory(src/layout)
add_subdirectory(src/table)

if(EXISTS "${CMAKE_SOURCE_DIR}/src/pdf/CMakeLists.txt")
    add_subdirectory(src/pdf)
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/src/reading_order/CMakeLists.txt")
    add_subdirectory(src/reading_order)
endif()
if(EXISTS "${CMAKE_SOURCE_DIR}/src/output/CMakeLists.txt")
    add_subdirectory(src/output)
endif()
if(HAS_DXNN_OCR AND EXISTS "${CMAKE_SOURCE_DIR}/src/pipeline/CMakeLists.txt")
    add_subdirectory(src/pipeline)
endif()

# ========================================
# CLI Application
# ========================================
if(BUILD_CLI AND HAS_DXNN_OCR)
    message(STATUS "Building CLI application")
    add_subdirectory(app)
else()
    message(STATUS "Skipping CLI (requires DXNN-OCR-cpp)")
endif()

# ========================================
# Server (HTTP API) — Crow needs standalone ASIO
# ========================================
if(BUILD_SERVER AND HAS_DXNN_OCR)
    message(STATUS "Building HTTP API server")
    # 1) Prefer local 3rd-party/asio (submodule or manual clone) to avoid build-time network
    if(EXISTS "${CMAKE_SOURCE_DIR}/3rd-party/asio/asio/include/asio.hpp")
        set(ASIO_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/3rd-party/asio/asio/include)
        message(STATUS "Using ASIO from 3rd-party/asio")
    # 2) Fallback: system Asio (e.g. libasio-dev), avoids FetchContent when network fails
    else()
        find_path(ASIO_SYSTEM_INCLUDE asio.hpp
            PATHS /usr/include /usr/local/include
            PATH_SUFFIXES asio
            NO_DEFAULT_PATH)
        if(ASIO_SYSTEM_INCLUDE)
            set(ASIO_INCLUDE_DIR "${ASIO_SYSTEM_INCLUDE}")
            message(STATUS "Using ASIO from system: ${ASIO_INCLUDE_DIR}")
        endif()
    endif()
    # 3) Fetch from GitHub only if no local/system Asio (fails when TLS/network unavailable)
    if(NOT ASIO_INCLUDE_DIR)
        include(FetchContent)
        FetchContent_Declare(asio
            GIT_REPOSITORY https://github.com/chriskohlhoff/asio.git
            GIT_TAG        asio-1-30-0
        )
        FetchContent_Populate(asio)
        set(ASIO_INCLUDE_DIR ${asio_SOURCE_DIR}/asio/include)
    endif()
    add_subdirectory(src/server)
    file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/src/server/CMakeFiles/doc_server.dir")
else()
    message(STATUS "Skipping Server (requires DXNN-OCR-cpp)")
endif()

# ========================================
# Tests
# ========================================
if(BUILD_TESTS)
    message(STATUS "Building unit tests")
    enable_testing()
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(3rd-party/googletest EXCLUDE_FROM_ALL)
    add_subdirectory(test)
endif()

# ========================================
# Build Information
# ========================================
message(STATUS "========================================")
message(STATUS "RapidDoc C++ Project Configuration")
message(STATUS "========================================")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "System Processor: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "Source Directory: ${CMAKE_SOURCE_DIR}")
message(STATUS "DXRT Installed: ${DXRT_INSTALLED_DIR}")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "ONNX Runtime: ${ONNXRUNTIME_FOUND}")
message(STATUS "Build CLI: ${BUILD_CLI}")
message(STATUS "Build Server: ${BUILD_SERVER}")
message(STATUS "Build Tests: ${BUILD_TESTS}")
message(STATUS "========================================")
