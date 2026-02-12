add_library(doc_pdf STATIC
    pdf_renderer.cpp
)

target_include_directories(doc_pdf PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Note: Poppler is linked when BUILD_SERVER is ON.
# For standalone PDF rendering, include FetchPoppler.cmake here too.
# Currently stubbed — Poppler linking will be added when implementation is complete.

target_link_libraries(doc_pdf
    doc_common
    ${OpenCV_LIBS}
    spdlog::spdlog
    stdc++fs
)
