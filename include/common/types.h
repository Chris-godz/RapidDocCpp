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
 * Maps to PP-DocLayout-L model's 23 categories (DXEngine path).
 * The DXNN split model (pp_doclayout_l_part1.dxnn) outputs these class IDs.
 * 
 * Categories marked [NPU_UNSUPPORTED] indicate pipeline stages
 * that cannot be processed by DEEPX NPU in the current version.
 *
 * For PP-DocLayout-Plus-L (ONNX, 20 categories), use LayoutCategoryPlusL.
 */
enum class LayoutCategory : int {
    // PP-DocLayout-L DXEngine output categories (indices 0-22)
    PARAGRAPH_TITLE = 0,     // paragraph_title → Title
    IMAGE = 1,               // image → ImageBody
    TEXT = 2,                 // text → Text
    NUMBER = 3,              // number → Abandon
    ABSTRACT = 4,            // abstract → Text
    CONTENT = 5,             // content → Text
    FIGURE_TITLE = 6,        // figure_title → ImageCaption
    FORMULA = 7,             // formula → InterlineEquation [NPU_UNSUPPORTED]
    TABLE = 8,               // table → TableBody
    TABLE_TITLE = 9,         // table_title → TableCaption
    REFERENCE = 10,          // reference → Text
    DOC_TITLE = 11,          // doc_title → Title
    FOOTNOTE = 12,           // footnote → Abandon
    HEADER = 13,             // header → Abandon
    ALGORITHM = 14,          // algorithm → Text
    FOOTER = 15,             // footer → Abandon
    SEAL = 16,               // seal → Abandon
    CHART_TITLE = 17,        // chart_title → ImageCaption
    CHART = 18,              // chart → ImageBody
    FORMULA_NUMBER = 19,     // formula_number → InterlineEquationNumber
    HEADER_IMAGE = 20,       // header_image → Abandon
    FOOTER_IMAGE = 21,       // footer_image → Abandon
    ASIDE_TEXT = 22,         // aside_text → Text
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
    int row, col;             // Cell position
    int rowSpan, colSpan;     // Spans
    float x0, y0, x1, y1;    // Cell bounding box
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

    // Normalized bbox (0-1000 scale)
    struct NormalizedBBox {
        int x0, y0, x1, y1;
    };
    NormalizedBBox getNormalizedBBox(int pageWidth, int pageHeight) const;
};

/**
 * @brief Complete result for a single page
 */
struct PageResult {
    int pageIndex = 0;
    LayoutResult layoutResult;
    std::vector<ContentElement> elements;  // Sorted by reading order
    std::vector<TableResult> tableResults;
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

    struct Stats {
        double pdfRenderTimeMs = 0.0;
        double layoutTimeMs = 0.0;
        double ocrTimeMs = 0.0;
        double tableTimeMs = 0.0;
        double readingOrderTimeMs = 0.0;
        double outputGenTimeMs = 0.0;
    } stats;
};

} // namespace rapid_doc
