# ========================================
# RapidDocCpp — Unit Tests
# ========================================
# Build with: cmake -DRAPIDDOC_BUILD_TESTS=ON ..
# Run with:   ctest --test-dir build_Release -V

include(GoogleTest)

# Common include dirs for all tests
set(TEST_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/test
    ${OpenCV_INCLUDE_DIRS}
)

# ========================================
# Test 1: Tensor Comparison Utilities
# ========================================
add_executable(test_tensor_compare
    test_tensor_compare.cpp
)

target_include_directories(test_tensor_compare PRIVATE
    ${TEST_INCLUDE_DIRS}
)

target_link_libraries(test_tensor_compare
    gtest
    gtest_main
    ${OpenCV_LIBS}
)

gtest_discover_tests(test_tensor_compare)

# ========================================
# Test 2: PDF Renderer
# ========================================
add_executable(test_pdf_renderer
    test_pdf_renderer.cpp
)

target_include_directories(test_pdf_renderer PRIVATE
    ${TEST_INCLUDE_DIRS}
    ${POPPLER_INCLUDE_DIRS}
)

target_link_libraries(test_pdf_renderer
    gtest
    gtest_main
    doc_pdf
    poppler-cpp
    poppler
    ${OpenCV_LIBS}
    ${FONTCONFIG_LIBRARIES}
    ${FREETYPE_LIBRARIES}
    z
    png
    jpeg
    pthread
)

gtest_discover_tests(test_pdf_renderer)

# ========================================
# Test 3: Layout Detector
# ========================================
add_executable(test_layout_detector
    test_layout_detector.cpp
)

target_include_directories(test_layout_detector PRIVATE
    ${TEST_INCLUDE_DIRS}
)

target_link_libraries(test_layout_detector
    gtest
    gtest_main
    doc_layout
    doc_common
    ${OpenCV_LIBS}
)

gtest_discover_tests(test_layout_detector)

# ========================================
# Test 4: Table Recognizer
# ========================================
add_executable(test_table_recognizer
    test_table_recognizer.cpp
)

target_include_directories(test_table_recognizer PRIVATE
    ${TEST_INCLUDE_DIRS}
)

target_link_libraries(test_table_recognizer
    gtest
    gtest_main
    doc_table
    doc_common
    ${OpenCV_LIBS}
)

gtest_discover_tests(test_table_recognizer)

# ========================================
# Test 5: Reading Order (XY-Cut++)
# ========================================
add_executable(test_reading_order
    test_reading_order.cpp
)

target_include_directories(test_reading_order PRIVATE
    ${TEST_INCLUDE_DIRS}
)

target_link_libraries(test_reading_order
    gtest
    gtest_main
    doc_reading_order
    doc_common
    ${OpenCV_LIBS}
)

gtest_discover_tests(test_reading_order)

# ========================================
# Test 6: DocPipeline
# ========================================
add_executable(test_doc_pipeline
    test_doc_pipeline.cpp
)

target_include_directories(test_doc_pipeline PRIVATE
    ${TEST_INCLUDE_DIRS}
    ${POPPLER_INCLUDE_DIRS}
)

target_link_libraries(test_doc_pipeline
    gtest
    gtest_main
    doc_pipeline
    doc_output
    doc_common
    doc_pdf
    poppler-cpp
    poppler
    ${OpenCV_LIBS}
    ${FONTCONFIG_LIBRARIES}
    ${FREETYPE_LIBRARIES}
    z
    png
    jpeg
    pthread
)

gtest_discover_tests(test_doc_pipeline)

# ========================================
# Test 7: Output (Markdown + ContentList)
# ========================================
add_executable(test_output
    test_output.cpp
)

target_include_directories(test_output PRIVATE
    ${TEST_INCLUDE_DIRS}
)

target_link_libraries(test_output
    gtest
    gtest_main
    doc_output
    doc_common
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
)

gtest_discover_tests(test_output)

# ========================================
# Test 8: Server
# ========================================
add_executable(test_server
    test_server.cpp
)

target_include_directories(test_server PRIVATE
    ${TEST_INCLUDE_DIRS}
)

target_link_libraries(test_server
    gtest
    gtest_main
    doc_server
    doc_pipeline
    doc_output
    doc_common
    ${OpenCV_LIBS}
    nlohmann_json::nlohmann_json
    pthread
)

gtest_discover_tests(test_server)
