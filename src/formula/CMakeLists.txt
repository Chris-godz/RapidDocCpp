add_library(doc_formula STATIC
    formula_recognizer.cpp
)

target_include_directories(doc_formula PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/3rd-party/json/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(doc_formula
    doc_common
    ${OpenCV_LIBS}
    spdlog::spdlog
)

if(ONNXRUNTIME_FOUND)
    target_include_directories(doc_formula PRIVATE ${ONNXRUNTIME_INCLUDE_DIRS})
    target_link_libraries(doc_formula ${ONNXRUNTIME_LIBRARIES})
    target_compile_definitions(doc_formula PRIVATE HAS_ONNXRUNTIME=1)
endif()

