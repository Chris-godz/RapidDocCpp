add_library(doc_pdf STATIC
    pdf_renderer.cpp
)

# Poppler installed from DXNN-OCR-cpp submodule build
set(POPPLER_INSTALL_DIR "${CMAKE_SOURCE_DIR}/3rd-party/DXNN-OCR-cpp/3rd-party/poppler/install")

find_library(POPPLER_LIB poppler PATHS "${POPPLER_INSTALL_DIR}/lib" NO_DEFAULT_PATH)
find_library(POPPLER_CPP_LIB poppler-cpp PATHS "${POPPLER_INSTALL_DIR}/lib" NO_DEFAULT_PATH)

if(NOT POPPLER_LIB OR NOT POPPLER_CPP_LIB)
    message(WARNING "Poppler not found at ${POPPLER_INSTALL_DIR} — PDF rendering disabled")
    target_compile_definitions(doc_pdf PRIVATE NO_POPPLER=1)
else()
    message(STATUS "Poppler found: ${POPPLER_CPP_LIB}")
endif()

target_include_directories(doc_pdf PUBLIC
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

if(POPPLER_CPP_LIB)
    target_include_directories(doc_pdf PRIVATE
        "${POPPLER_INSTALL_DIR}/include"
        "${POPPLER_INSTALL_DIR}/include/poppler"
    )
endif()

target_link_libraries(doc_pdf
    doc_common
    ${OpenCV_LIBS}
    spdlog::spdlog
    stdc++fs
)

if(POPPLER_CPP_LIB AND POPPLER_LIB)
    target_link_libraries(doc_pdf ${POPPLER_CPP_LIB} ${POPPLER_LIB}
        fontconfig freetype png jpeg tiff z pthread)
endif()
