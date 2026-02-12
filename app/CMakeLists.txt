# CLI Application

add_executable(rapid_doc_cli main.cpp)

target_include_directories(rapid_doc_cli PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(rapid_doc_cli
    doc_pipeline
    doc_common
    ${OpenCV_LIBS}
    spdlog::spdlog
    stdc++fs
)

set_target_properties(rapid_doc_cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

install(TARGETS rapid_doc_cli DESTINATION bin)
