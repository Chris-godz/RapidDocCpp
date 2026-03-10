# ========================================
# Test Utilities
# ========================================
add_library(test_utils STATIC
    metrics.cpp
    npy_loader.cpp
)

target_include_directories(test_utils PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(test_utils PUBLIC
    doc_common
)

# ========================================
# Unit Tests — Metrics (self-test)
# ========================================
add_executable(rapiddoc_tests
    test_metrics.cpp
)

target_include_directories(rapiddoc_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(rapiddoc_tests PRIVATE
    gtest_main
    test_utils
)

add_test(NAME MetricsTests COMMAND rapiddoc_tests)

# ========================================
# Cross-Validation Tests (C++ vs Python fixtures)
# ========================================
add_executable(rapiddoc_cross_tests
    test_layout_preprocess.cpp
    test_layout_postprocess.cpp
    test_layout_nms_standalone.cpp
    test_table_preprocess.cpp
    test_table_inference.cpp
    test_table_postprocess.cpp
    test_ocr_integration.cpp
    test_live_inference.cpp
)

target_include_directories(rapiddoc_cross_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/3rd-party/json/include
)

target_compile_definitions(rapiddoc_cross_tests PRIVATE
    PROJECT_ROOT_DIR="${PROJECT_SOURCE_DIR}"
)

target_link_libraries(rapiddoc_cross_tests PRIVATE
    gtest_main
    test_utils
    doc_layout
    doc_table
    ${OpenCV_LIBS}
)

# Link ONNX Runtime if available (used by layout postprocess test)
if(ONNXRUNTIME_FOUND)
    target_include_directories(rapiddoc_cross_tests PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
    target_link_libraries(rapiddoc_cross_tests PRIVATE ${ONNXRUNTIME_LIBRARIES})
    target_compile_definitions(rapiddoc_cross_tests PRIVATE HAS_ONNXRUNTIME=1)
endif()

add_test(NAME CrossValidationTests COMMAND rapiddoc_cross_tests)

# ========================================
# End-to-End Pipeline Tests
# ========================================
if(HAS_DXNN_OCR)
    add_executable(rapiddoc_e2e_tests
        test_e2e_pipeline.cpp
    )

    target_include_directories(rapiddoc_e2e_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/3rd-party/json/include
        ${PROJECT_SOURCE_DIR}/3rd-party/DXNN-OCR-cpp/include
    )

    target_compile_definitions(rapiddoc_e2e_tests PRIVATE
        PROJECT_ROOT_DIR="${PROJECT_SOURCE_DIR}"
    )

    target_link_libraries(rapiddoc_e2e_tests PRIVATE
        gtest_main
        doc_pipeline
        ${OpenCV_LIBS}
    )

    add_test(NAME E2ETests COMMAND rapiddoc_e2e_tests)

    # E2E Runner tool (processes all PDFs, not a test)
    add_executable(run_cpp_e2e
        ${PROJECT_SOURCE_DIR}/tools/run_cpp_e2e.cpp
    )

    target_include_directories(run_cpp_e2e PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/3rd-party/json/include
        ${PROJECT_SOURCE_DIR}/3rd-party/DXNN-OCR-cpp/include
    )

    target_compile_definitions(run_cpp_e2e PRIVATE
        PROJECT_ROOT_DIR="${PROJECT_SOURCE_DIR}"
    )

    target_link_libraries(run_cpp_e2e PRIVATE
        doc_pipeline
        ${OpenCV_LIBS}
    )
endif()
