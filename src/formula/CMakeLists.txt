add_library(doc_formula STATIC
    formula_recognizer.cpp
)

target_include_directories(doc_formula PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

if(ONNXRUNTIME_FOUND)
    target_link_libraries(doc_formula PUBLIC ${ONNXRUNTIME_LIBRARIES})
    target_include_directories(doc_formula PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
    target_compile_definitions(doc_formula PUBLIC HAS_ONNXRUNTIME=1)
endif()

target_link_libraries(doc_formula PUBLIC
    doc_common
    ${OpenCV_LIBS}
    spdlog::spdlog
)
