#pragma once

/**
 * @file config.h
 * @brief Pipeline configuration for RapidDoc C++
 * 
 * Controls which stages are enabled, model paths, and runtime parameters.
 */

#include <string>
#include <cstddef>

namespace rapid_doc {

enum class PipelineMode {
    Serial,
    PagePipelineMvp,
};

const char* pipelineModeToString(PipelineMode mode);
bool parsePipelineMode(const std::string& raw, PipelineMode& mode);

enum class OcrOuterMode {
    ImmediatePerTask,
    ShadowWindowedCollect,
};

const char* ocrOuterModeToString(OcrOuterMode mode);
bool parseOcrOuterMode(const std::string& raw, OcrOuterMode& mode);

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
};

/**
 * @brief Pipeline stage enable/disable switches
 * 
 * NOTE: Wireless table and table classification remain unsupported.
 * Formula regions are exposed as image fallbacks rather than LaTeX output.
 */
struct PipelineStages {
    bool enablePdfRender = true;        // PDF → page images
    bool enableLayout = true;           // Layout detection (PP-DocLayout on NPU)
    bool enableOcr = true;              // OCR detection + recognition (DXNN-OCR-cpp)
    bool enableWiredTable = true;       // Wired table recognition (UNET on NPU)
    bool enableReadingOrder = true;     // XY-Cut reading order sort
    bool enableMarkdownOutput = true;   // Generate Markdown output

    bool enableFormula = true;          // Formula/equation image fallback
    bool enableWirelessTable = false;   // Wireless table recognition (SLANet)
    bool enableTableClassify = false;   // Table type classification
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
    PipelineMode pipelineMode = PipelineMode::Serial;
    OcrOuterMode ocrOuterMode = OcrOuterMode::ImmediatePerTask;
    size_t ocrShadowWindow = 8;
    size_t stageQueueRendered = 2;
    size_t stageQueuePlanned = 2;
    size_t stageQueueOcrTable = 4;
    size_t stageQueueFinalize = 4;
    
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
