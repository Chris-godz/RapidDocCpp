add_library(doc_table STATIC
    table_recognizer.cpp
)

target_include_directories(doc_table PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(doc_table
    doc_common
    ${OpenCV_LIBS}
    spdlog::spdlog
)
