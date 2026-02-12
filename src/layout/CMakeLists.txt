add_library(doc_layout STATIC
    layout_detector.cpp
)

target_include_directories(doc_layout PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

# Link ONNX Runtime if available (for post-processing sub-model)
if(ONNXRUNTIME_FOUND)
    target_include_directories(doc_layout PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
    target_link_libraries(doc_layout ${ONNXRUNTIME_LIBRARIES})
    target_compile_definitions(doc_layout PRIVATE HAS_ONNXRUNTIME=1)
endif()

target_link_libraries(doc_layout
    doc_common
    ${OpenCV_LIBS}
    spdlog::spdlog
)
