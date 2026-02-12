# ========================================
# Poppler Build Configuration
# ========================================
# Compiles Poppler from git submodule source for PDF rendering.
# Reused from DXNN-OCR-cpp project.
#
# Usage:
#   include(${CMAKE_SOURCE_DIR}/cmake/FetchPoppler.cmake)
#   target_link_libraries(my_target poppler-cpp)
#
# Dependencies (Ubuntu/Debian):
#   sudo apt-get install -y libfontconfig1-dev libfreetype6-dev \
#       libjpeg-dev libpng-dev libtiff-dev libopenjp2-7-dev libboost-dev
#
# Submodule setup:
#   git submodule add https://gitlab.freedesktop.org/poppler/poppler.git 3rd-party/poppler
#   cd 3rd-party/poppler && git checkout poppler-24.02.0

set(POPPLER_VERSION "24.02.0" CACHE STRING "Poppler version (git tag)")

set(POPPLER_ROOT_DIR "${CMAKE_SOURCE_DIR}/3rd-party/poppler")
set(POPPLER_SOURCE_DIR "${POPPLER_ROOT_DIR}")
set(POPPLER_BUILD_DIR "${POPPLER_ROOT_DIR}/build")
set(POPPLER_INSTALL_DIR "${POPPLER_ROOT_DIR}/install")

message(STATUS "========================================")
message(STATUS "Poppler Configuration")
message(STATUS "========================================")
message(STATUS "  Version: ${POPPLER_VERSION}")
message(STATUS "  Source dir: ${POPPLER_SOURCE_DIR}")

# Check system dependencies
find_package(PkgConfig REQUIRED)
pkg_check_modules(FONTCONFIG REQUIRED fontconfig)
pkg_check_modules(FREETYPE REQUIRED freetype2)
pkg_check_modules(JPEG libjpeg)
pkg_check_modules(PNG libpng)

# Check if already built
set(POPPLER_LIBRARY_FILE "${POPPLER_INSTALL_DIR}/lib/libpoppler.a")
set(POPPLER_CPP_LIBRARY_FILE "${POPPLER_INSTALL_DIR}/lib/libpoppler-cpp.a")
set(POPPLER_HEADER_FILE "${POPPLER_INSTALL_DIR}/include/poppler/cpp/poppler-document.h")
set(POPPLER_NEED_BUILD TRUE)

if(EXISTS "${POPPLER_LIBRARY_FILE}" AND EXISTS "${POPPLER_CPP_LIBRARY_FILE}" AND EXISTS "${POPPLER_HEADER_FILE}")
    set(POPPLER_VERSION_FILE "${POPPLER_INSTALL_DIR}/VERSION")
    if(EXISTS "${POPPLER_VERSION_FILE}")
        file(READ "${POPPLER_VERSION_FILE}" POPPLER_EXISTING_VERSION)
        string(STRIP "${POPPLER_EXISTING_VERSION}" POPPLER_EXISTING_VERSION)
        if("${POPPLER_EXISTING_VERSION}" STREQUAL "${POPPLER_VERSION}")
            set(POPPLER_NEED_BUILD FALSE)
            message(STATUS "  Poppler ${POPPLER_VERSION} already built, skipping")
        endif()
    endif()
endif()

# Build from submodule
if(POPPLER_NEED_BUILD)
    message(STATUS "Building Poppler from git submodule...")

    if(NOT EXISTS "${POPPLER_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "Poppler source not found at: ${POPPLER_SOURCE_DIR}\n"
            "Please initialize the git submodule:\n"
            "  git submodule update --init 3rd-party/poppler\n"
            "  cd 3rd-party/poppler && git checkout poppler-${POPPLER_VERSION}"
        )
    endif()

    file(MAKE_DIRECTORY "${POPPLER_BUILD_DIR}")
    file(MAKE_DIRECTORY "${POPPLER_INSTALL_DIR}")

    execute_process(
        COMMAND ${CMAKE_COMMAND}
            -S "${POPPLER_SOURCE_DIR}"
            -B "${POPPLER_BUILD_DIR}"
            -DCMAKE_BUILD_TYPE=Release
            -DCMAKE_INSTALL_PREFIX=${POPPLER_INSTALL_DIR}
            -DCMAKE_POSITION_INDEPENDENT_CODE=ON
            -DENABLE_CPP=ON
            -DENABLE_UTILS=OFF
            -DENABLE_GLIB=OFF
            -DENABLE_GOBJECT_INTROSPECTION=OFF
            -DENABLE_QT5=OFF
            -DENABLE_QT6=OFF
            -DENABLE_LIBCURL=OFF
            -DENABLE_BOOST=OFF
            -DBUILD_SHARED_LIBS=OFF
            -DENABLE_SPLASH=ON
            -DSPLASH_CMYK=OFF
            -DENABLE_LIBOPENJPEG=none
            -DENABLE_LIBTIFF=OFF
            -DENABLE_NSS3=OFF
            -DENABLE_GPGME=OFF
            -DENABLE_LCMS=OFF
            -DENABLE_ZLIB_UNCOMPRESS=OFF
            -DBUILD_GTK_TESTS=OFF
            -DBUILD_QT5_TESTS=OFF
            -DBUILD_QT6_TESTS=OFF
            -DBUILD_CPP_TESTS=OFF
            -DBUILD_MANUAL_TESTS=OFF
            -DTESTDATADIR=""
        WORKING_DIRECTORY "${POPPLER_BUILD_DIR}"
        RESULT_VARIABLE CMAKE_RESULT
        ERROR_VARIABLE CMAKE_ERROR
    )

    if(NOT CMAKE_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Failed to configure Poppler:\n${CMAKE_ERROR}\n"
            "sudo apt-get install -y libfontconfig1-dev libfreetype6-dev libjpeg-dev libpng-dev"
        )
    endif()

    include(ProcessorCount)
    ProcessorCount(NPROC)
    if(NPROC EQUAL 0)
        set(NPROC 4)
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} --build "${POPPLER_BUILD_DIR}" --parallel ${NPROC}
        RESULT_VARIABLE BUILD_RESULT
        ERROR_VARIABLE BUILD_ERROR
    )

    if(NOT BUILD_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to build Poppler:\n${BUILD_ERROR}")
    endif()

    execute_process(
        COMMAND ${CMAKE_COMMAND} --install "${POPPLER_BUILD_DIR}"
        RESULT_VARIABLE INSTALL_RESULT
    )

    if(NOT INSTALL_RESULT EQUAL 0)
        message(FATAL_ERROR "Failed to install Poppler")
    endif()

    file(WRITE "${POPPLER_INSTALL_DIR}/VERSION" "${POPPLER_VERSION}")
    message(STATUS "Poppler built and installed successfully")
endif()

# ========================================
# Set Poppler paths and create targets
# ========================================
set(POPPLER_ROOT "${POPPLER_INSTALL_DIR}")

find_path(Poppler_INCLUDE_DIR
    NAMES "poppler/cpp/poppler-document.h"
    PATHS "${POPPLER_ROOT}"
    PATH_SUFFIXES "include"
    NO_DEFAULT_PATH
)

find_library(Poppler_LIBRARY
    NAMES "poppler"
    PATHS "${POPPLER_ROOT}"
    PATH_SUFFIXES "lib" "lib64"
    NO_DEFAULT_PATH
)

find_library(Poppler_CPP_LIBRARY
    NAMES "poppler-cpp"
    PATHS "${POPPLER_ROOT}"
    PATH_SUFFIXES "lib" "lib64"
    NO_DEFAULT_PATH
)

if(EXISTS "${POPPLER_ROOT}/VERSION")
    file(READ "${POPPLER_ROOT}/VERSION" Poppler_VERSION)
    string(STRIP "${Poppler_VERSION}" Poppler_VERSION)
else()
    set(Poppler_VERSION "${POPPLER_VERSION}")
endif()

if(Poppler_LIBRARY AND Poppler_CPP_LIBRARY AND Poppler_INCLUDE_DIR)
    add_library(poppler STATIC IMPORTED GLOBAL)
    set_target_properties(poppler PROPERTIES
        IMPORTED_LOCATION "${Poppler_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${Poppler_INCLUDE_DIR}"
    )

    add_library(poppler-cpp STATIC IMPORTED GLOBAL)
    set_target_properties(poppler-cpp PROPERTIES
        IMPORTED_LOCATION "${Poppler_CPP_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${Poppler_INCLUDE_DIR};${Poppler_INCLUDE_DIR}/poppler"
        INTERFACE_LINK_LIBRARIES "poppler;${FONTCONFIG_LIBRARIES};${FREETYPE_LIBRARIES};pthread"
    )

    set(Poppler_FOUND TRUE)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Poppler
    REQUIRED_VARS Poppler_LIBRARY Poppler_CPP_LIBRARY Poppler_INCLUDE_DIR
    VERSION_VAR Poppler_VERSION
)

if(Poppler_FOUND)
    message(STATUS "✓ Poppler configured: ${Poppler_VERSION}")
    set(POPPLER_INCLUDE_DIRS "${Poppler_INCLUDE_DIR}" CACHE INTERNAL "Poppler include directories")
    set(POPPLER_LIBRARIES "${Poppler_CPP_LIBRARY};${Poppler_LIBRARY}" CACHE INTERNAL "Poppler libraries")
else()
    message(FATAL_ERROR "Failed to configure Poppler.")
endif()
