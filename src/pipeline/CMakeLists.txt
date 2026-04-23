add_library(doc_pipeline STATIC
    doc_pipeline.cpp
)

target_compile_definitions(doc_pipeline PRIVATE
    PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
)

target_include_directories(doc_pipeline PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/3rd-party/DXNN-OCR-cpp/include
    ${OpenCV_INCLUDE_DIRS}
)

# PUBLIC so executables (rapid_doc_cli, tests) that link doc_pipeline get OpenCV/ocr_pipeline transitively and resolve cv::cvtColor etc.
target_link_libraries(doc_pipeline PUBLIC
    doc_common
    doc_pdf
    doc_layout
    doc_formula
    doc_table
    doc_reading_order
    doc_output
    ocr_pipeline
    ${OpenCV_LIBS}
    spdlog::spdlog
    stdc++fs
)
