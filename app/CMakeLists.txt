# CLI Application

add_executable(rapid_doc_cli main.cpp)

target_include_directories(rapid_doc_cli PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_compile_definitions(rapid_doc_cli PRIVATE
    PROJECT_ROOT_DIR="${CMAKE_SOURCE_DIR}"
)

# OpenCV last so unresolved symbols from static libs (e.g. ocr_common → cv::cvtColor) are resolved.
if(TARGET rapiddoc_opencv_libs)
    set(_rapiddoc_ocv_libs rapiddoc_opencv_libs)
    add_dependencies(rapid_doc_cli opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui opencv_freetype)
else()
    set(_rapiddoc_ocv_libs ${OpenCV_LIBS})
endif()
target_link_libraries(rapid_doc_cli
    doc_pipeline
    doc_common
    spdlog::spdlog
    stdc++fs
    ${_rapiddoc_ocv_libs}
)

set_target_properties(rapid_doc_cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

install(TARGETS rapid_doc_cli DESTINATION bin)
