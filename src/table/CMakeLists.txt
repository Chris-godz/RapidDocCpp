add_library(doc_table STATIC
    table_recognizer.cpp
    table_classifier.cpp
    table_wireless_recognizer.cpp
)

target_include_directories(doc_table PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/3rd-party/DXNN-OCR-cpp/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(doc_table
    doc_common
    ${OpenCV_LIBS}
    spdlog::spdlog
    dxrt
    pthread
)

if(ONNXRUNTIME_FOUND)
    target_include_directories(doc_table PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
    target_link_libraries(doc_table ${ONNXRUNTIME_LIBRARIES})
    target_compile_definitions(doc_table PRIVATE HAS_ONNXRUNTIME=1)
endif()
