#pragma once

/**
 * @file config.h
 * @brief Pipeline configuration for RapidDoc C++
 * 
 * Controls which stages are enabled, model paths, and runtime parameters.
 * Stages that DEEPX NPU does not support are disabled by default.
 */

#include <string>

namespace rapid_doc {

/**
 * @brief Model paths configuration
 */
struct ModelPaths {
    // Layout detection (DEEPX NPU + ONNX post-processing)
    std::string layoutDxnnModel;       // .dxnn model for DX NPU inference
    std::string layoutOnnxSubModel;    // .onnx sub-model for NMS post-processing

    // Table recognition (DEEPX NPU — wired tables only)
    std::string tableUnetDxnnModel;    // .dxnn model for UNET table segmentation

    // Table classification & wireless table (ONNX Runtime)
    std::string tableClsOnnxModel;     // .onnx model for table classification
    std::string tableSlanetOnnxModel;  // .onnx model for SLANet wireless table
    std::string tableSlanetDictPath;   // Dictionary for SLANet token decoding

    // Formula recognition (ONNX Runtime)
    std::string formulaOnnxModel;      // .onnx model for formula recognition
    std::string formulaDictPath;       // Dictionary for formula token decoding

    // OCR models are managed by DXNN-OCR-cpp subproject
    // Set via OCRPipelineConfig — see ocr_pipeline.h
    std::string ocrModelDir;           // Base directory for OCR .dxnn models

    // Character dictionary (used by OCR recognition)
    std::string ocrDictPath;           // ppocrv5_dict.txt
};

/**
 * @brief Pipeline stage enable/disable switches
 * 
 * NOTE: Stages marked [NPU_UNSUPPORTED] are disabled by default.
 * They have interface stubs for future enablement when NPU support is added.
 */
struct PipelineStages {
    bool enablePdfRender = true;        // PDF → page images
    bool enableLayout = true;           // Layout detection (PP-DocLayout on NPU)
    bool enableOcr = true;              // OCR detection + recognition (DXNN-OCR-cpp)
    bool enableWiredTable = true;       // Wired table recognition (UNET on NPU)
    bool enableReadingOrder = true;     // XY-Cut reading order sort
    bool enableMarkdownOutput = true;   // Generate Markdown output

    // [NPU_UNSUPPORTED] — disabled by default, stubs only
    bool enableFormula = true;         // Formula/equation recognition
    bool enableWirelessTable = true;   // Wireless table recognition (SLANet)
    bool enableTableClassify = true;   // Table type classification
};

/**
 * @brief Runtime parameters
 */
struct RuntimeConfig {
    int pdfDpi = 200;                   // PDF rendering DPI
    int maxPages = 0;                   // Max pages to process (0 = all)
    int maxConcurrentPages = 4;         // Parallel PDF rendering limit
    
    // Layout detection
    // PP-DocLayout-L uses 640x640 input (PP-DocLayout-Plus-L uses 800x800)
    // DXNN split model is PP-DocLayout-L, consistent with Python RapidDoc DXEngine path
    float layoutConfThreshold = 0.4f;   // Layout detection confidence threshold
    int layoutInputSize = 640;          // Layout model input size (PP-DocLayout-L)

    // Table recognition
    float tableConfThreshold = 0.5f;    // Table detection confidence threshold

    // Table cell OCR backfill
    bool enableTableCellOcr = true;     // Run OCR on table cells and fill <td> text

    // Output
    std::string outputDir = "./output"; // Output directory
    bool saveImages = true;             // Save extracted images
    bool saveVisualization = false;     // Save layout visualization
};

/**
 * @brief Complete pipeline configuration
 */
struct PipelineConfig {
    ModelPaths models;
    PipelineStages stages;
    RuntimeConfig runtime;

    /**
     * @brief Create default configuration with standard model paths
     * @param projectRoot Project root directory (for relative paths)
     * @return Default configuration
     */
    static PipelineConfig Default(const std::string& projectRoot = ".");

    /**
     * @brief Validate configuration (check model files exist, etc.)
     * @return Empty string if valid, error message otherwise
     */
    std::string validate() const;

    /**
     * @brief Print configuration summary to log
     */
    void show() const;
};

} // namespace rapid_doc
