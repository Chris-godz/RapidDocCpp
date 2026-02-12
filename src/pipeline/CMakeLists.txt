add_library(doc_pipeline STATIC
    doc_pipeline.cpp
)

target_include_directories(doc_pipeline PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/3rd-party/DXNN-OCR-cpp/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(doc_pipeline
    doc_common
    doc_pdf
    doc_layout
    doc_table
    doc_reading_order
    doc_output
    # DXNN-OCR-cpp libraries
    ocr_pipeline
    ${OpenCV_LIBS}
    spdlog::spdlog
    stdc++fs
)
