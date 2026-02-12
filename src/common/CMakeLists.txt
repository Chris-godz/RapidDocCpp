add_library(doc_common STATIC
    types.cpp
    config.cpp
)

target_include_directories(doc_common PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(doc_common
    ${OpenCV_LIBS}
    spdlog::spdlog
    stdc++fs
)
