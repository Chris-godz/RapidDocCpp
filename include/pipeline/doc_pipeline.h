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
#include "pipeline/ocr_pipeline.h"
#include <string>
#include <memory>
#include <functional>
#include <vector>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <unordered_set>

namespace rapid_doc {

class DocPipelineTestAccess;

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
     * @brief Process a single image as a single-page document.
     * @param image Input image (BGR)
     * @param pageIndex Page number metadata
     * @return Complete single-page document result
     */
    DocumentResult processImageDocument(const cv::Mat& image, int pageIndex = 0);

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
    void setOutputDir(const std::string& dir) { config_.runtime.outputDir = dir; }
    void setStageOptions(const PipelineStages& stages) { config_.stages = stages; }
    void setSaveImages(bool save) { config_.runtime.saveImages = save; }
    void setSaveVisualization(bool save) { config_.runtime.saveVisualization = save; }
    void setPageRange(int startPageId, int endPageId) {
        config_.runtime.startPageId = startPageId;
        config_.runtime.endPageId = endPageId;
    }
    void setMaxPages(int maxPages) { config_.runtime.maxPages = maxPages; }

private:
    friend class DocPipelineTestAccess;

    /**
     * @brief Process a single page through the pipeline
     */
    PageResult processPage(const PageImage& pageImage);

    using OcrSubmitHook = std::function<bool(const cv::Mat&, int64_t)>;
    using OcrFetchHook = std::function<bool(
        std::vector<ocr::PipelineOCRResult>&, int64_t&, bool&)>;
    using TableRecognizeHook = std::function<TableResult(const cv::Mat&)>;
    using TableHtmlHook = std::function<std::string(const std::vector<TableCell>&)>;

    /**
     * @brief Run OCR on text regions detected by layout
     * @param image Full page image
     * @param textBoxes Layout boxes classified as text
     * @return ContentElements with OCR text
     */
    std::vector<ContentElement> runOcrOnRegions(
        const cv::Mat& image,
        const std::vector<LayoutBox>& textBoxes,
        int pageIndex
    );

    /**
     * @brief Run table recognition on detected table regions
     * @param image Full page image
     * @param tableBoxes Layout boxes classified as table
     * @return ContentElements with table HTML
     */
    std::vector<ContentElement> runTableRecognition(
        const cv::Mat& image,
        const std::vector<LayoutBox>& tableBoxes,
        int pageIndex
    );

    /**
     * @brief Handle unsupported elements (formula, etc.)
     * Creates placeholder ContentElements
     */
    std::vector<ContentElement> handleUnsupportedElements(
        const std::vector<LayoutBox>& unsupportedBoxes,
        int pageIndex
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

    void saveFormulaImages(
        const cv::Mat& image,
        const std::vector<LayoutBox>& equationBoxes,
        int pageIndex,
        std::vector<ContentElement>& elements
    );

    void saveLayoutVisualization(
        const cv::Mat& image,
        const LayoutResult& layoutResult,
        int pageIndex
    );

    /**
     * @brief Run OCR on a single image crop and return combined text.
     * @param crop BGR image to OCR
     * @param taskId Unique task ID for the async OCR queue
     * @return Recognized text (empty on failure)
     */
    std::string ocrOnCrop(const cv::Mat& crop, int64_t taskId);
    bool submitOcrTask(const cv::Mat& crop, int64_t taskId);
    bool fetchOcrResult(
        std::vector<ocr::PipelineOCRResult>& results,
        int64_t& resultId,
        bool& success);
    bool waitForOcrResult(
        int64_t taskId,
        std::vector<ocr::PipelineOCRResult>& results,
        bool& success);
    int64_t allocateOcrTaskId();

    TableResult recognizeTable(const cv::Mat& tableCrop);
    std::string generateTableHtml(const std::vector<TableCell>& cells);
    ContentElement makeTableFallbackElement(
        const LayoutBox& box,
        int pageIndex,
        const std::string& reason) const;
    static std::string tableFallbackMessage(const std::string& reason);

    void reportProgress(const std::string& stage, int current, int total);

    PipelineConfig config_;
    bool initialized_ = false;

    // Pipeline components
    std::unique_ptr<PdfRenderer> pdfRenderer_;
    std::unique_ptr<LayoutDetector> layoutDetector_;
    std::unique_ptr<TableRecognizer> tableRecognizer_;
    std::unique_ptr<ocr::OCRPipeline> ocrPipeline_;

    OcrSubmitHook ocrSubmitHook_;
    OcrFetchHook ocrFetchHook_;
    TableRecognizeHook tableRecognizeHook_;
    TableHtmlHook tableHtmlHook_;

    struct BufferedOcrResult {
        std::vector<ocr::PipelineOCRResult> results;
        bool success = false;
    };
    std::unordered_map<int64_t, BufferedOcrResult> bufferedOcrResults_;
    std::unordered_set<int64_t> timedOutOcrTaskIds_;
    std::chrono::milliseconds ocrWaitTimeout_{30000};
    std::atomic<int64_t> nextOcrTaskId_{1};

    // Output writers
    MarkdownWriter markdownWriter_;
    ContentListWriter contentListWriter_;

    // Callbacks
    ProgressCallback progressCallback_;
};

} // namespace rapid_doc
