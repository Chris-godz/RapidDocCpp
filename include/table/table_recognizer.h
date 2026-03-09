#pragma once

/**
 * @file table_recognizer.h
 * @brief Table recognition using DEEPX NPU (UNET model — wired tables only)
 *
 * Architecture (matching Python dx_infer_session.py):
 *   - Single DX Engine: dxrt::InferenceEngine loads .dxnn UNET model
 *   - No ONNX sub-model needed (unlike Layout)
 *   - Post-processing in C++:
 *     1. Separate H/V line masks from UNET class map {0=bg,1=hline,2=vline}
 *     2. Morphological close + connected components → line segments
 *     3. Line adjustment + extension → draw lines on blank image
 *     4. Connected components on inverted line image → cell polygons
 *     5. Table structure recovery (row grouping, column benchmarking, merge cells)
 *     6. Generate HTML <table> from logic points
 *
 * IMPORTANT: Only wired tables (tables with visible borders) are supported.
 * Wireless table recognition (SLANet/SLANeXt) is NOT supported on DEEPX NPU.
 */

#include "common/types.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

namespace rapid_doc {

// ============================================================================
// Configuration
// ============================================================================

struct TableRecognizerConfig {
    std::string unetDxnnModelPath;       ///< .dxnn UNET model path
    std::string tableClsOnnxModelPath;   ///< .onnx model path for table classification (wired/wireless)
    std::string tableSlanetOnnxModelPath;///< .onnx model path for SLANet wireless table
    std::string tableSlanetDictPath;     ///< dictionary for SLANet tokens
    int inputSize = 768;                 ///< Model input size (DXEngine uses 768)
    float lineScaleFactor = 1.2f;        ///< Morphological kernel scale: sqrt(dim) * factor
    float cellContainThresh = 0.6f;      ///< Containment threshold for OCR-to-cell matching
    bool useAsync = false;               ///< Enable async inference
    int deviceId = 0;                    ///< DX device ID
    bool enableTableClassify = false;    ///< Enable ONNX table classification
    bool enableWirelessTable = false;    ///< Enable ONNX SLANet wireless table
};

// ============================================================================
// Internal types (used across pipeline stages, not exposed in types.h)
// ============================================================================

struct LineSegment {
    cv::Point2f start;
    cv::Point2f end;
    float angle = 0.0f;     ///< degrees (0°=horizontal, 90°=vertical)
    float length = 0.0f;
};

struct LogicCell {
    int rowStart = 0;
    int rowEnd = 1;          ///< exclusive end (rowEnd - rowStart = rowSpan)
    int colStart = 0;
    int colEnd = 1;          ///< exclusive end (colEnd - colStart = colSpan)
    float x0 = 0, y0 = 0, x1 = 0, y1 = 0;  ///< Bounding box
    std::string text;        ///< OCR text content

    float width() const { return x1 - x0; }
    float height() const { return y1 - y0; }
    cv::Rect toRect() const {
        return cv::Rect(static_cast<int>(x0), static_cast<int>(y0),
                         static_cast<int>(width()), static_cast<int>(height()));
    }
};

// ============================================================================
// TableRecognizer
// ============================================================================

class TableRecognizer {
public:
    explicit TableRecognizer(const TableRecognizerConfig& config);
    ~TableRecognizer();

    bool initialize();

    /**
     * @brief Recognize a wired table from a cropped BGR image.
     *
     * Pipeline: preprocess → UNET inference → postprocess →
     *   line extraction → cell extraction → structure recovery → HTML
     *
     * Falls back to morphology-only when DX Engine is not loaded.
     * Returns TableResult with .html, .cells, .supported fields.
     */
    TableResult recognize(const cv::Mat& tableImage);

    /**
     * @brief Heuristic wired/wireless classification (Canny + morphology).
     * Fallback for the Table Cls model which cannot run on DEEPX NPU.
     */
    static TableType estimateTableType(const cv::Mat& tableImage);

    /**
     * @brief Classify whether table is wired or wireless using ONNX model.
     * Falls back to estimateTableType if ONNX model is not available or disabled.
     */
    TableType classifyTableType(const cv::Mat& tableImage);

    /**
     * @brief Recognize a wireless table using SLANet ONNX model.
     * Returns TableResult with .html and .cells.
     */
    TableResult recognizeWireless(const cv::Mat& tableImage);

    bool isInitialized() const { return initialized_; }

private:
    // --- Pipeline stages ---
    cv::Mat preprocess(const cv::Mat& image,
                       float& scaleX, float& scaleY,
                       int& padLeft, int& padTop);

    cv::Mat runInference(const cv::Mat& preprocessed);

    void postprocessMasks(const cv::Mat& predMask,
                          const cv::Size& originalSize,
                          float scaleX, float scaleY,
                          int padLeft, int padTop,
                          cv::Mat& hMask, cv::Mat& vMask);

    std::vector<LineSegment> extractLineSegments(const cv::Mat& lineMask,
                                                  bool isHorizontal);

    void adjustLines(std::vector<LineSegment>& hLines,
                     std::vector<LineSegment>& vLines,
                     const cv::Size& imageSize);

    std::vector<TableCell> extractCells(const std::vector<LineSegment>& hLines,
                                         const std::vector<LineSegment>& vLines,
                                         const cv::Size& imageSize);

    std::vector<LogicCell> recoverLogicStructure(const std::vector<TableCell>& cells,
                                                  const cv::Size& imageSize);

    std::string generateHtml(const std::vector<LogicCell>& logicCells);

    // --- Utilities ---
    static float pointDistance(const cv::Point2f& a, const cv::Point2f& b);
    static bool isBoxContained(const cv::Rect& inner, const cv::Rect& outer,
                               float threshold);
    static cv::Point2f lineIntersection(const LineSegment& l1, const LineSegment& l2,
                                         bool& found);

    // --- Members ---
    TableRecognizerConfig config_;
    bool initialized_ = false;
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rapid_doc