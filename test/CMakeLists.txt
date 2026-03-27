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
    test_perf_utils.cpp
    test_detail_report.cpp
)

target_include_directories(rapiddoc_tests PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(rapiddoc_tests PRIVATE
    gtest_main
    test_utils
    doc_output
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

if(TARGET rapiddoc_opencv_libs)
    set(_test_ocv_libs rapiddoc_opencv_libs)
else()
    set(_test_ocv_libs ${OpenCV_LIBS})
endif()
target_link_libraries(rapiddoc_cross_tests PRIVATE
    gtest_main
    test_utils
    doc_layout
    doc_table
    ${_test_ocv_libs}
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
    add_executable(rapiddoc_contract_tests
        test_phase1_contracts.cpp
    )

    target_include_directories(rapiddoc_contract_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/3rd-party/json/include
        ${PROJECT_SOURCE_DIR}/3rd-party/DXNN-OCR-cpp/include
    )

    target_compile_definitions(rapiddoc_contract_tests PRIVATE
        PROJECT_ROOT_DIR="${PROJECT_SOURCE_DIR}"
    )

    if(TARGET rapiddoc_opencv_libs)
        set(_contract_ocv_libs rapiddoc_opencv_libs)
    else()
        set(_contract_ocv_libs ${OpenCV_LIBS})
    endif()
    target_link_libraries(rapiddoc_contract_tests PRIVATE
        gtest_main
        doc_pipeline
        ${_contract_ocv_libs}
    )

    add_test(NAME Phase1ContractTests COMMAND rapiddoc_contract_tests)

    if(TARGET doc_server)
        add_executable(rapid_doc_test_server
            http_test_server_main.cpp
        )

        target_include_directories(rapid_doc_test_server PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/3rd-party/json/include
            ${PROJECT_SOURCE_DIR}/3rd-party/DXNN-OCR-cpp/include
        )

        target_compile_definitions(rapid_doc_test_server PRIVATE
            PROJECT_ROOT_DIR="${PROJECT_SOURCE_DIR}"
        )

        target_link_libraries(rapid_doc_test_server PRIVATE
            doc_server
            doc_pipeline
            spdlog::spdlog
        )

        set_target_properties(rapid_doc_test_server PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        )

        add_executable(rapid_doc_npu_test_server
            http_npu_test_server_main.cpp
        )

        target_include_directories(rapid_doc_npu_test_server PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/3rd-party/json/include
            ${PROJECT_SOURCE_DIR}/3rd-party/DXNN-OCR-cpp/include
        )

        target_compile_definitions(rapid_doc_npu_test_server PRIVATE
            PROJECT_ROOT_DIR="${PROJECT_SOURCE_DIR}"
        )

        target_link_libraries(rapid_doc_npu_test_server PRIVATE
            doc_server
            doc_pipeline
            spdlog::spdlog
        )

        set_target_properties(rapid_doc_npu_test_server PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
        )

        add_executable(rapiddoc_integration_tests
            test_phase1_integration.cpp
            test_phase1_http_integration.cpp
            test_topology_lb_integration.cpp
        )

        target_include_directories(rapiddoc_integration_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/3rd-party/json/include
            ${PROJECT_SOURCE_DIR}/3rd-party/DXNN-OCR-cpp/include
        )

        target_compile_definitions(rapiddoc_integration_tests PRIVATE
            PROJECT_ROOT_DIR="${PROJECT_SOURCE_DIR}"
            RAPIDDOC_BUILD_DIR="${PROJECT_BINARY_DIR}"
        )

        if(TARGET rapiddoc_opencv_libs)
            set(_integration_ocv_libs rapiddoc_opencv_libs)
        else()
            set(_integration_ocv_libs ${OpenCV_LIBS})
        endif()
        target_link_libraries(rapiddoc_integration_tests PRIVATE
            gtest_main
            doc_pipeline
            doc_server
            ${_integration_ocv_libs}
        )
        add_dependencies(rapiddoc_integration_tests rapid_doc_test_server)
        add_dependencies(rapiddoc_integration_tests rapid_doc_topology_lb)

        add_test(NAME Phase1IntegrationTests COMMAND rapiddoc_integration_tests)

        add_executable(rapiddoc_npu_concurrency_tests
            test_phase2_npu_concurrency.cpp
        )

        target_include_directories(rapiddoc_npu_concurrency_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}
            ${PROJECT_SOURCE_DIR}/include
            ${PROJECT_SOURCE_DIR}/3rd-party/json/include
            ${PROJECT_SOURCE_DIR}/3rd-party/DXNN-OCR-cpp/include
        )

        target_compile_definitions(rapiddoc_npu_concurrency_tests PRIVATE
            PROJECT_ROOT_DIR="${PROJECT_SOURCE_DIR}"
            RAPIDDOC_BUILD_DIR="${PROJECT_BINARY_DIR}"
        )

        if(TARGET rapiddoc_opencv_libs)
            set(_npu_integration_ocv_libs rapiddoc_opencv_libs)
        else()
            set(_npu_integration_ocv_libs ${OpenCV_LIBS})
        endif()
        target_link_libraries(rapiddoc_npu_concurrency_tests PRIVATE
            gtest_main
            doc_pipeline
            doc_server
            ${_npu_integration_ocv_libs}
        )
        add_dependencies(rapiddoc_npu_concurrency_tests rapid_doc_npu_test_server)

        add_test(NAME NpuConcurrencyIntegrationTests COMMAND rapiddoc_npu_concurrency_tests)
    endif()

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

    if(TARGET rapiddoc_opencv_libs)
        set(_e2e_ocv_libs rapiddoc_opencv_libs)
    else()
        set(_e2e_ocv_libs ${OpenCV_LIBS})
    endif()
    target_link_libraries(rapiddoc_e2e_tests PRIVATE
        gtest_main
        doc_pipeline
        ${_e2e_ocv_libs}
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

    if(TARGET rapiddoc_opencv_libs)
        set(_run_e2e_ocv_libs rapiddoc_opencv_libs)
    else()
        set(_run_e2e_ocv_libs ${OpenCV_LIBS})
    endif()
    target_link_libraries(run_cpp_e2e PRIVATE
        doc_pipeline
        ${_run_e2e_ocv_libs}
    )
endif()
