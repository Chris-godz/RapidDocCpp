#pragma once

/**
 * @file doc_pipeline.h
 * @brief Main document processing pipeline
 * 
 * Orchestrates the full document analysis flow:
 *   1. PDF Rendering (Poppler)
 *   2. Layout Detection (DEEPX NPU + ONNX RT post-processing)
 *   3. OCR (DXNN-OCR-cpp — text detection + recognition)
 *   4. Table Recognition (DEEPX NPU UNET — wired tables only)
 *   5. Reading Order (XY-Cut++ algorithm)
 *   6. Output Generation (Markdown + JSON content list)
 * 
 * Unsupported stages (Formula, Wireless Table, Table Classify) are
 * gracefully skipped with placeholders in the output.
 * 
 * The pipeline uses DXNN-OCR-cpp's OCRPipeline via Git submodule.
 */

#include "common/types.h"
#include "common/config.h"
#include "pdf/pdf_renderer.h"
#include "layout/layout_detector.h"
#include "table/table_recognizer.h"
#include "reading_order/xycut.h"
#include "output/markdown_writer.h"
#include "output/content_list.h"
#include <string>
#include <memory>
#include <functional>
#include <vector>

// Forward declaration for DXNN-OCR-cpp types
namespace ocr {
    class OCRPipeline;
    struct OCRPipelineConfig;
    struct PipelineOCRResult;
}

namespace rapid_doc {

/**
 * @brief Progress callback for pipeline stages
 */
using ProgressCallback = std::function<void(const std::string& stage, int current, int total)>;

/**
 * @brief Main document processing pipeline
 */
class DocPipeline {
public:
    explicit DocPipeline(const PipelineConfig& config);
    ~DocPipeline();

    /**
     * @brief Initialize all pipeline components
     * @return true if all enabled components initialized successfully
     */
    bool initialize();

    /**
     * @brief Process a PDF file
     * @param pdfPath Path to input PDF file
     * @return Complete document result
     */
    DocumentResult processPdf(const std::string& pdfPath);

    /**
     * @brief Process a PDF from memory
     * @param data Raw PDF bytes
     * @param size Data size
     * @return Complete document result
     */
    DocumentResult processPdfFromMemory(const uint8_t* data, size_t size);

    /**
     * @brief Process a single page image (no PDF rendering)
     * @param image Page image (BGR)
     * @param pageIndex Page number (for output metadata)
     * @return Page result
     */
    PageResult processImage(const cv::Mat& image, int pageIndex = 0);

    /**
     * @brief Set progress callback
     */
    void setProgressCallback(ProgressCallback callback) { progressCallback_ = callback; }

    /**
     * @brief Check if pipeline is initialized
     */
    bool isInitialized() const { return initialized_; }

    /**
     * @brief Get current configuration
     */
    const PipelineConfig& config() const { return config_; }

private:
    /**
     * @brief Process a single page through the pipeline
     */
    PageResult processPage(const PageImage& pageImage);

    /**
     * @brief Run OCR on text regions detected by layout
     * @param image Full page image
     * @param textBoxes Layout boxes classified as text
     * @return ContentElements with OCR text
     */
    std::vector<ContentElement> runOcrOnRegions(
        const cv::Mat& image,
        const std::vector<LayoutBox>& textBoxes
    );

    /**
     * @brief Run table recognition on detected table regions
     * @param image Full page image
     * @param tableBoxes Layout boxes classified as table
     * @return ContentElements with table HTML
     */
    std::vector<ContentElement> runTableRecognition(
        const cv::Mat& image,
        const std::vector<LayoutBox>& tableBoxes
    );

    /**
     * @brief Handle unsupported elements (formula, etc.)
     * Creates placeholder ContentElements
     */
    std::vector<ContentElement> handleUnsupportedElements(
        const std::vector<LayoutBox>& unsupportedBoxes
    );

    /**
     * @brief Save extracted images (figures) from layout
     */
    void saveExtractedImages(
        const cv::Mat& image,
        const std::vector<LayoutBox>& figureBoxes,
        int pageIndex,
        std::vector<ContentElement>& elements
    );

    /**
     * @brief Report progress
     */
    void reportProgress(const std::string& stage, int current, int total);

    PipelineConfig config_;
    bool initialized_ = false;

    // Pipeline components
    std::unique_ptr<PdfRenderer> pdfRenderer_;
    std::unique_ptr<LayoutDetector> layoutDetector_;
    std::unique_ptr<TableRecognizer> tableRecognizer_;
    std::unique_ptr<ocr::OCRPipeline> ocrPipeline_;

    // Output writers
    MarkdownWriter markdownWriter_;
    ContentListWriter contentListWriter_;

    // Callbacks
    ProgressCallback progressCallback_;
};

} // namespace rapid_doc
