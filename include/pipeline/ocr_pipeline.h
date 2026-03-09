/**
 * @file ocr_pipeline.h
 * @brief OCR Pipeline forward declarations for DXNN-OCR-cpp integration
 * 
 * This header provides forward declarations of the DXNN-OCR-cpp
 * public API types used within the RapidDocCpp pipeline.
 * 
 * The actual implementation lives in the submodule:
 *   3rd-party/DXNN-OCR-cpp
 * 
 * Only forward declarations are provided here to avoid propagating
 * heavy DXRT dependencies into all translation units.  The full
 * DXNN-OCR-cpp headers are included only in doc_pipeline.cpp.
 * 
 * Key types:
 *   ocr::OCRPipeline         – async detection + recognition pipeline
 *   ocr::OCRPipelineConfig   – pipeline configuration
 *   ocr::OCRTaskConfig       – per-request task parameters
 *   ocr::PipelineOCRResult   – single text-box result (box, text, confidence)
 */

#pragma once

namespace ocr {
    class OCRPipeline;
    struct OCRPipelineConfig;
    struct OCRTaskConfig;
    struct PipelineOCRResult;
    struct DetectorConfig;
    struct ClassifierConfig;
    struct DocumentPreprocessingConfig;
}

namespace DeepXOCR {
    struct RecognizerConfig;
}
