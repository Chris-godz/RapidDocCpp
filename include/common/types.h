#pragma once

/**
 * @file types.h
 * @brief Core data types for RapidDoc C++ pipeline
 * 
 * Defines all shared data structures used across pipeline stages:
 * Layout detection, OCR, table recognition, reading order, and output generation.
 */

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace rapid_doc {

// ========================================
// Layout Detection Types
// ========================================

/**
 * @brief PP-DocLayout category enumeration
 * 
 * Maps to PP-DocLayout-plus-L model's 20 categories.
 * Categories marked [NPU_UNSUPPORTED] indicate pipeline stages
 * that cannot be processed by DEEPX NPU in the current version.
 */
enum class LayoutCategory : int {
    TEXT = 0,
    TITLE = 1,
    FIGURE = 2,
    FIGURE_CAPTION = 3,
    TABLE = 4,
    TABLE_CAPTION = 5,
    TABLE_FOOTNOTE = 6,
    HEADER = 7,
    FOOTER = 8,
    REFERENCE = 9,
    EQUATION = 10,           // [NPU_UNSUPPORTED] Formula recognition
    INTERLINE_EQUATION = 11, // [NPU_UNSUPPORTED] Formula recognition
    STAMP = 12,
    CODE = 13,
    TOC = 14,                // Table of Contents
    ABSTRACT = 15,
    CONTENT = 16,
    LIST = 17,
    INDEX = 18,
    SEPARATOR = 19,
    UNKNOWN = -1,
};

/**
 * @brief Convert LayoutCategory to string name
 */
const char* layoutCategoryToString(LayoutCategory cat);

/**
 * @brief Check if a layout category is supported by DEEPX NPU pipeline
 */
bool isCategorySupported(LayoutCategory cat);

/**
 * @brief Single detected region in a page
 */
struct LayoutBox {
    float x0, y0, x1, y1;      // Bounding box coordinates (top-left, bottom-right)
    LayoutCategory category;     // Detected element type
    float confidence;            // Detection confidence [0, 1]
    int index;                   // Original detection order
    int clsId = -1;             // Original model class ID
    std::string label;          // Original model label (e.g. "doc_title", "paragraph_title")

    // Convenience methods
    float width() const { return x1 - x0; }
    float height() const { return y1 - y0; }
    float area() const { return width() * height(); }
    cv::Rect toRect() const {
        return cv::Rect(
            static_cast<int>(x0), static_cast<int>(y0),
            static_cast<int>(width()), static_cast<int>(height())
        );
    }
    cv::Point2f center() const { return cv::Point2f((x0 + x1) / 2, (y0 + y1) / 2); }
};

/**
 * @brief Layout detection result for a single page
 */
struct LayoutResult {
    std::vector<LayoutBox> boxes;
    double inferenceTimeMs = 0.0;

    // Filter helpers
    std::vector<LayoutBox> getBoxesByCategory(LayoutCategory cat) const;
    std::vector<LayoutBox> getTextBoxes() const;
    std::vector<LayoutBox> getTableBoxes() const;
    std::vector<LayoutBox> getEquationBoxes() const;
    std::vector<LayoutBox> getSupportedBoxes() const;
    std::vector<LayoutBox> getUnsupportedBoxes() const;
};

// ========================================
// Table Recognition Types
// ========================================

/**
 * @brief Table type classification
 */
enum class TableType {
    WIRED,       // Has visible borders — supported by UNET on NPU
    WIRELESS,    // No visible borders — NOT supported (requires SLANet)
    UNKNOWN,
};

/**
 * @brief Single table cell
 */
struct TableCell {
    int row, col;             // Cell position (logical grid)
    int rowSpan, colSpan;     // Spans (from TableRecover)
    float x0, y0, x1, y1;    // Cell bounding box (axis-aligned)
    float poly[8];            // 4-point polygon (x1,y1,x2,y2,x3,y3,x4,y4)
    std::string content;      // OCR text content
};

/**
 * @brief Table recognition result
 */
struct TableResult {
    TableType type = TableType::UNKNOWN;
    std::string html;                     // HTML representation
    std::vector<TableCell> cells;
    bool supported = false;               // Whether NPU could process this
    double inferenceTimeMs = 0.0;
};

// ========================================
// Page-Level Types
// ========================================

/**
 * @brief Rendered page image from PDF
 */
struct PageImage {
    cv::Mat image;          // Rendered page as BGR Mat
    int pageIndex;          // 0-based page number
    int dpi;                // Rendering DPI
    double scaleFactor;     // Scale relative to PDF coordinates
    int pdfWidth;           // Original PDF page width (points)
    int pdfHeight;          // Original PDF page height (points)
    double renderTimeMs = 0.0; // Pure page render + color conversion time
};

/**
 * @brief Single content element in the final output
 */
struct ContentElement {
    enum class Type {
        TEXT,
        TITLE,
        IMAGE,
        TABLE,
        EQUATION,         // Placeholder — NPU unsupported
        CODE,
        LIST,
        HEADER,
        FOOTER,
        REFERENCE,
        UNKNOWN,
    };

    Type type = Type::UNKNOWN;
    std::string text;                     // Text content or LaTeX
    std::string imagePath;                // Path if image element
    std::string html;                     // HTML if table element
    LayoutBox layoutBox;                  // Original bounding box
    int pageIndex = 0;
    int readingOrder = 0;                 // Assigned by XY-Cut
    float confidence = 0.0f;
    bool skipped = false;                 // True if NPU couldn't process this

    // Normalized bbox (0-1000 scale, matching Python ContentList format)
    struct NormalizedBBox {
        int x0, y0, x1, y1;
    };
    NormalizedBBox getNormalizedBBox(int pageWidth, int pageHeight) const;
};

/**
 * @brief Stage timings for a single processed page
 */
struct PageStageStats {
    double layoutTimeMs = 0.0;
    double ocrTimeMs = 0.0;
    double tableTimeMs = 0.0;
    double figureTimeMs = 0.0;
    double formulaTimeMs = 0.0;
    double unsupportedTimeMs = 0.0;
    double readingOrderTimeMs = 0.0;
    // Non-overlapping observability slices for Phase 2 lock-splitting prep.
    double npuSerialTimeMs = 0.0;
    double cpuOnlyTimeMs = 0.0;
    double npuLockWaitTimeMs = 0.0;
    double npuLockHoldTimeMs = 0.0;
    double npuServiceTimeMs = 0.0;
    double npuSlotWaitTimeMs = 0.0;
    double layoutNpuServiceTimeMs = 0.0;
    double layoutNpuSlotWaitTimeMs = 0.0;
    double ocrOuterSlotHoldTimeMs = 0.0;
    double ocrSubmoduleWindowTimeMs = 0.0;
    double ocrSlotWaitTimeMs = 0.0;
    double ocrCollectWaitTimeMs = 0.0;
    double ocrInflightPeak = 0.0;
    double ocrBufferedOutOfOrderCount = 0.0;
    double tableNpuServiceTimeMs = 0.0;
    double tableNpuSlotWaitTimeMs = 0.0;
    double tableOcrServiceTimeMs = 0.0;
    double tableOcrSlotWaitTimeMs = 0.0;
    double cpuPreTimeMs = 0.0;
    double cpuPostTimeMs = 0.0;
    double finalizeCpuTimeMs = 0.0;
    double tableFinalizeTimeMs = 0.0;
    double ocrCollectOrMergeTimeMs = 0.0;
    double layoutQueueWaitTimeMs = 0.0;
    double planQueueWaitTimeMs = 0.0;
    double ocrTableQueueWaitTimeMs = 0.0;
    double finalizeQueueWaitTimeMs = 0.0;
    double renderQueuePushBlockTimeMs = 0.0;
    double layoutQueuePushBlockTimeMs = 0.0;
    double planQueuePushBlockTimeMs = 0.0;
    double ocrTableQueuePushBlockTimeMs = 0.0;
    double queueBackpressureTimeMs = 0.0;
    double pipelineOverlapFactor = 0.0;
    std::string pipelineMode;
    // Phase 1 attribution counters (additive fields).
    double textBoxesRawCount = 0.0;
    double textBoxesAfterDedupCount = 0.0;
    double tableBoxesRawCount = 0.0;
    double tableBoxesAfterDedupCount = 0.0;
    double ocrSubmitCount = 0.0;
    double ocrSubmitAreaSum = 0.0;
    double ocrSubmitAreaMean = 0.0;
    double ocrSubmitAreaP50 = 0.0;
    double ocrSubmitAreaP95 = 0.0;
    double ocrSubmitSmallCount = 0.0;
    double ocrSubmitMediumCount = 0.0;
    double ocrSubmitLargeCount = 0.0;
    double ocrSubmitTextCount = 0.0;
    double ocrSubmitTitleCount = 0.0;
    double ocrSubmitCodeCount = 0.0;
    double ocrSubmitListCount = 0.0;
    double ocrDedupSkippedCount = 0.0;
    double tableNpuSubmitCount = 0.0;
    double tableDedupSkippedCount = 0.0;
    double ocrTimeoutCount = 0.0;
    double ocrBufferedResultHitCount = 0.0;
};

/**
 * @brief Aggregated stage timings for a processed document
 */
struct DocumentStageStats : public PageStageStats {
    double pdfRenderTimeMs = 0.0;
    double outputGenTimeMs = 0.0;
};

/**
 * @brief Complete result for a single page
 */
struct PageResult {
    int pageIndex = 0;
    int pageWidth = 0;
    int pageHeight = 0;
    LayoutResult layoutResult;
    std::vector<ContentElement> elements;  // Sorted by reading order
    std::vector<TableResult> tableResults;
    PageStageStats stats;
    double totalTimeMs = 0.0;
};

/**
 * @brief Complete document processing result
 */
struct DocumentResult {
    std::vector<PageResult> pages;
    std::string markdown;                   // Generated Markdown
    std::string contentListJson;            // Structured JSON output
    double totalTimeMs = 0.0;
    int totalPages = 0;
    int processedPages = 0;
    int skippedElements = 0;                // Elements skipped due to NPU limitations
    DocumentStageStats stats;
};

} // namespace rapid_doc
