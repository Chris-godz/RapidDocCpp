add_library(doc_output STATIC
    markdown_writer.cpp
    content_list.cpp
)

target_include_directories(doc_output PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/3rd-party/json/include
)

target_link_libraries(doc_output PUBLIC
    doc_common
)
