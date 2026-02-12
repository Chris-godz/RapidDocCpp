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

add_definitions(-DPROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}")
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

# Suppress warnings from 3rd-party libraries
set(CMAKE_CXX_FLAGS_BACKUP "${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

set(CLIPPER2_TESTS OFF CACHE BOOL "" FORCE)
set(CLIPPER2_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(3rd-party/clipper2/CPP)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_BACKUP}")

# ========================================
# DXNN-OCR-cpp as Subproject
# ========================================
# Tell DXNN-OCR-cpp it's being built as a subproject
# so it skips its own 3rd-party builds and executables
set(DXNN_OCR_AS_SUBPROJECT ON CACHE BOOL "" FORCE)
add_subdirectory(3rd-party/DXNN-OCR-cpp)

# ========================================
# Global Include Directories
# ========================================
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/3rd-party/json/include)
include_directories(${CMAKE_SOURCE_DIR}/3rd-party/spdlog/include)
include_directories(SYSTEM ${CMAKE_SOURCE_DIR}/3rd-party/clipper2/CPP/Clipper2Lib/include)

# DXNN-OCR-cpp public headers
include_directories(${CMAKE_SOURCE_DIR}/3rd-party/DXNN-OCR-cpp/include)

# ========================================
# RapidDocCpp Libraries
# ========================================
add_subdirectory(src/common)
add_subdirectory(src/pdf)
add_subdirectory(src/layout)
add_subdirectory(src/table)
add_subdirectory(src/reading_order)
add_subdirectory(src/output)
add_subdirectory(src/pipeline)

# ========================================
# CLI Application
# ========================================
if(BUILD_CLI)
    message(STATUS "Building CLI application")
    add_subdirectory(app)
endif()

# ========================================
# Server (HTTP API)
# ========================================
if(BUILD_SERVER)
    message(STATUS "Building HTTP API server")
    add_subdirectory(src/server)
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
