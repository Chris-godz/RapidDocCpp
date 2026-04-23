#pragma once

/**
 * @file config.h
 * @brief Pipeline configuration for RapidDoc C++
 * 
 * Controls which stages are enabled, model paths, and runtime parameters.
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

    // OCR models are managed by DXNN-OCR-cpp subproject
    // Set via OCRPipelineConfig — see ocr_pipeline.h
    std::string ocrModelDir;           // Base directory for OCR .dxnn models

    // Character dictionary (used by OCR recognition)
    std::string ocrDictPath;           // ppocrv5_dict.txt

    // Formula recognition (ONNX Runtime)
    std::string formulaOnnxModel;      // pp_formulanet_plus_m.onnx

    // Table classifier (ONNX Runtime) — wired/wireless router (PaddleCls).
    std::string tableClsOnnxModel;     // paddle_cls.onnx

    // Wireless table structure model (ONNX Runtime) — SLANet+.
    std::string slanetPlusOnnxModel;   // slanet-plus.onnx
};

/**
 * @brief Pipeline stage enable/disable switches
 *
 * NOTE: Wireless table backend remains unsupported; only the table-type
 * classifier (PaddleCls, wired/wireless) is available.
 */
struct PipelineStages {
    bool enablePdfRender = true;        // PDF → page images
    bool enableLayout = true;           // Layout detection (PP-DocLayout on NPU)
    bool enableOcr = true;              // OCR detection + recognition (DXNN-OCR-cpp)
    bool enableWiredTable = true;       // Wired table recognition (UNET on NPU)
    bool enableReadingOrder = true;     // XY-Cut reading order sort
    bool enableMarkdownOutput = true;   // Generate Markdown output

    bool enableFormula = true;          // Formula/equation LaTeX recognition
    bool enableWirelessTable = true;    // Wireless table recognition (SLANet+)
    // Table wired/wireless routing via PaddleCls ONNX (paddle_cls.onnx).
    // When true (default), classifier verdict is the primary router and the
    // geometric lineRatio heuristic only serves as a fallback when the
    // classifier is unavailable. When false, the legacy heuristic is used
    // exclusively.
    bool enableTableClassify = true;
};

/**
 * @brief Runtime parameters
 */
struct RuntimeConfig {
    int pdfDpi = 200;                   // PDF rendering DPI
    int maxPages = 0;                   // Max pages to process (0 = all)
    int startPageId = 0;                // Inclusive start page (0-based)
    int endPageId = -1;                 // Inclusive end page (-1 = all)
    int maxConcurrentPages = 4;         // Parallel PDF rendering limit
    int deviceId = -1;                  // DXRT device affinity (-1 = runtime default)
    
    // Layout detection
    float layoutConfThreshold = 0.5f;   // Layout detection confidence threshold
    int layoutInputSize = 640;          // Layout model input size (pp_doclayout_l)

    // Table recognition
    float tableConfThreshold = 0.5f;    // Table detection confidence threshold

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
