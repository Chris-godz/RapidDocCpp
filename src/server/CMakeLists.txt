# HTTP Server

add_library(doc_server STATIC
    server.cpp
)

target_include_directories(doc_server PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/3rd-party/crow/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(doc_server
    doc_pipeline
    doc_common
    nlohmann_json::nlohmann_json
    pthread
)

# Server executable
add_executable(rapid_doc_server server_main.cpp)

target_include_directories(rapid_doc_server PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/3rd-party/crow/include
)

target_link_libraries(rapid_doc_server
    doc_server
    doc_pipeline
    spdlog::spdlog
)

set_target_properties(rapid_doc_server PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

install(TARGETS rapid_doc_server DESTINATION bin)
