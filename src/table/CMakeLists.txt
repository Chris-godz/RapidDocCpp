add_library(doc_table STATIC
    table_recognizer.cpp
)

find_package(dxrt REQUIRED HINTS ${DXRT_INSTALLED_DIR})

target_include_directories(doc_table PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

if(ONNXRUNTIME_FOUND)
    target_link_libraries(doc_table PUBLIC ${ONNXRUNTIME_LIBRARIES})
    target_include_directories(doc_table PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
    target_compile_definitions(doc_table PUBLIC HAS_ONNXRUNTIME=1)
endif()

target_link_libraries(doc_table PUBLIC
    doc_common
    dxrt
    pthread
    ${OpenCV_LIBS}
    spdlog::spdlog
)

target_compile_definitions(doc_table PRIVATE HAS_DXRT=1)
