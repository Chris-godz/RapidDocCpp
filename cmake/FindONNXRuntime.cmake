# ========================================
# Find ONNX Runtime
# ========================================
# Used for Layout post-processing (NMS sub-model).
# Searches system paths and common install locations.
#
# Sets:
#   ONNXRUNTIME_FOUND       - TRUE if found
#   ONNXRUNTIME_INCLUDE_DIRS - Include directories
#   ONNXRUNTIME_LIBRARIES   - Libraries to link

set(ONNXRUNTIME_ROOT "" CACHE PATH "ONNX Runtime installation directory")

# Search paths
set(_ONNXRT_SEARCH_PATHS
    ${ONNXRUNTIME_ROOT}
    /usr/local
    /usr
    /opt/onnxruntime
)

# Find header
find_path(ONNXRUNTIME_INCLUDE_DIR
    NAMES onnxruntime_cxx_api.h
    PATHS ${_ONNXRT_SEARCH_PATHS}
    PATH_SUFFIXES include include/onnxruntime include/onnxruntime/core/session
)

# Find library
find_library(ONNXRUNTIME_LIBRARY
    NAMES onnxruntime
    PATHS ${_ONNXRT_SEARCH_PATHS}
    PATH_SUFFIXES lib lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNXRuntime
    DEFAULT_MSG
    ONNXRUNTIME_LIBRARY
    ONNXRUNTIME_INCLUDE_DIR
)

if(ONNXRUNTIME_FOUND)
    set(ONNXRUNTIME_INCLUDE_DIRS ${ONNXRUNTIME_INCLUDE_DIR})
    set(ONNXRUNTIME_LIBRARIES ${ONNXRUNTIME_LIBRARY})

    if(NOT TARGET onnxruntime::onnxruntime)
        add_library(onnxruntime::onnxruntime SHARED IMPORTED)
        set_target_properties(onnxruntime::onnxruntime PROPERTIES
            IMPORTED_LOCATION "${ONNXRUNTIME_LIBRARY}"
            INTERFACE_INCLUDE_DIRECTORIES "${ONNXRUNTIME_INCLUDE_DIR}"
        )
    endif()

    message(STATUS "✓ ONNX Runtime found")
    message(STATUS "  Include: ${ONNXRUNTIME_INCLUDE_DIR}")
    message(STATUS "  Library: ${ONNXRUNTIME_LIBRARY}")
else()
    message(WARNING
        "ONNX Runtime not found. Layout post-processing will use stub implementation.\n"
        "To install: sudo apt-get install libonnxruntime-dev\n"
        "Or set -DONNXRUNTIME_ROOT=/path/to/onnxruntime"
    )
endif()

mark_as_advanced(ONNXRUNTIME_INCLUDE_DIR ONNXRUNTIME_LIBRARY)
