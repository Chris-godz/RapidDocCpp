add_library(doc_reading_order STATIC
    xycut.cpp
)

target_include_directories(doc_reading_order PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(doc_reading_order
    doc_common
    spdlog::spdlog
)
