add_library(doc_pdf STATIC
    pdf_renderer.cpp
)

target_include_directories(doc_pdf PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${POPPLER_INCLUDE_DIRS}
)

target_link_libraries(doc_pdf
    doc_common
    poppler-cpp
    ${OpenCV_LIBS}
    spdlog::spdlog
    stdc++fs
    z
)
