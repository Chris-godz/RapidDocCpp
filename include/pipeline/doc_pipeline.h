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
#include <optional>
#include <mutex>
#include <unordered_map>
#include <unordered_set>

namespace rapid_doc {

namespace detail {

struct OcrWorkItem {
    LayoutBox box;
    ContentElement::Type type = ContentElement::Type::TEXT;
    int pageIndex = 0;
    float confidence = 0.0f;
    cv::Mat crop;
    bool skipped = false;
    std::vector<size_t> originalIndices;
};

struct OcrFetchResult {
    int64_t taskId = -1;
    bool submitted = false;
    bool fetched = false;
    bool success = false;
    bool bufferedHit = false;
    double slotWaitMs = 0.0;
    double outerSlotHoldMs = 0.0;
    double submoduleWindowMs = 0.0;
    double collectWaitMs = 0.0;
    size_t bufferedOutOfOrderCount = 0;
    std::optional<std::chrono::steady_clock::time_point> submitTime;
    std::vector<ocr::PipelineOCRResult> results;
};

struct TableWorkItem {
    LayoutBox box;
    int pageIndex = 0;
    cv::Mat crop;
    bool invalidRoi = false;
    std::vector<size_t> originalIndices;
};

struct TableNpuResult {
    LayoutBox box;
    int pageIndex = 0;
    bool hasTableResult = false;
    bool hasFallback = false;
    std::string fallbackReason;
    TableResult tableResult;
    TableRecognizer::NpuStageResult npuStage;
    std::vector<ocr::PipelineOCRResult> ocrBoxes;
};

} // namespace detail

class DocPipelineTestAccess;
class DocServer;

/**
 * @brief Progress callback for pipeline stages
 */
using ProgressCallback = std::function<void(const std::string& stage, int current, int total)>;

/**
 * @brief Per-request runtime/stage overrides for one pipeline execution.
 *
 * These overrides are merged onto the pipeline's base config and do not mutate
 * shared pipeline configuration state.
 */
struct PipelineRunOverrides {
    std::optional<std::string> outputDir;
    std::optional<bool> saveImages;
    std::optional<bool> saveVisualization;
    std::optional<int> startPageId;
    std::optional<int> endPageId;
    std::optional<int> maxPages;
    std::optional<PipelineMode> pipelineMode;
    std::optional<OcrOuterMode> ocrOuterMode;
    std::optional<size_t> ocrShadowWindow;
    std::optional<bool> enableFormula;
    std::optional<bool> enableWiredTable;
    std::optional<bool> enableMarkdownOutput;
};

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
    DocumentResult processPdfWithOverrides(
        const std::string& pdfPath,
        const PipelineRunOverrides& overrides);

    /**
     * @brief Process a PDF from memory
     * @param data Raw PDF bytes
     * @param size Data size
     * @return Complete document result
     */
    DocumentResult processPdfFromMemory(const uint8_t* data, size_t size);
    DocumentResult processPdfFromMemoryWithOverrides(
        const uint8_t* data,
        size_t size,
        const PipelineRunOverrides& overrides);

    /**
     * @brief Process a single image as a single-page document.
     * @param image Input image (BGR)
     * @param pageIndex Page number metadata
     * @return Complete single-page document result
     */
    DocumentResult processImageDocument(const cv::Mat& image, int pageIndex = 0);
    DocumentResult processImageDocumentWithOverrides(
        const cv::Mat& image,
        int pageIndex,
        const PipelineRunOverrides& overrides);

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
    friend class DocServer;

    struct ExecutionContext {
        PipelineStages stages;
        RuntimeConfig runtime;
    };

    struct PageContext {
        std::chrono::steady_clock::time_point wallStartTime;
        PageImage pageImage;
        PageResult pageResult;
        std::vector<LayoutBox> textBoxes;
        std::vector<LayoutBox> tableBoxes;
        std::vector<LayoutBox> figureBoxes;
        std::vector<LayoutBox> equationBoxes;
        std::vector<LayoutBox> unsupportedBoxes;

        std::vector<detail::OcrWorkItem> ocrWorkItems;
        std::vector<size_t> ocrWorkItemForOriginal;
        std::vector<size_t> ocrCanonicalWorkItem;
        std::vector<detail::OcrFetchResult> ocrFetchResults;
        std::vector<ContentElement> textElements;

        std::vector<detail::TableWorkItem> tableWorkItems;
        std::vector<size_t> tableWorkItemForOriginal;
        std::vector<size_t> tableCanonicalWorkItem;
        std::vector<detail::TableNpuResult> tableNpuResults;
        std::vector<ContentElement> tableElements;

        double npuLockWaitTotalMs = 0.0;
        double npuLockHoldTotalMs = 0.0;
        double npuSerialTotalMs = 0.0;
        double cpuOnlyTotalMs = 0.0;
        double layoutNpuServiceTotalMs = 0.0;
        double layoutNpuSlotWaitTotalMs = 0.0;
        double ocrTableNpuServiceTotalMs = 0.0;
        double ocrTableNpuSlotWaitTotalMs = 0.0;
        double ocrOuterSlotHoldTotalMs = 0.0;
        double ocrSubmoduleWindowTotalMs = 0.0;
        double ocrSlotWaitTotalMs = 0.0;
        double ocrCollectWaitTotalMs = 0.0;
        size_t ocrInflightPeak = 0;
        size_t ocrBufferedOutOfOrderCount = 0;
        double tableNpuServiceTotalMs = 0.0;
        double tableNpuSlotWaitTotalMs = 0.0;
        double tableOcrServiceTotalMs = 0.0;
        double tableOcrSlotWaitTotalMs = 0.0;
        double ocrTableCpuPreTotalMs = 0.0;
        double ocrTableCpuPostTotalMs = 0.0;
        double finalizeCpuTotalMs = 0.0;
        double tableFinalizeTotalMs = 0.0;
        double ocrCollectOrMergeTotalMs = 0.0;
        double ocrSubmitAreaSum = 0.0;
        size_t textBoxesRawCount = 0;
        size_t textBoxesAfterDedupCount = 0;
        size_t tableBoxesRawCount = 0;
        size_t tableBoxesAfterDedupCount = 0;
        size_t ocrSubmitCount = 0;
        size_t ocrSubmitSmallCount = 0;
        size_t ocrSubmitMediumCount = 0;
        size_t ocrSubmitLargeCount = 0;
        size_t ocrSubmitTextCount = 0;
        size_t ocrSubmitTitleCount = 0;
        size_t ocrSubmitCodeCount = 0;
        size_t ocrSubmitListCount = 0;
        size_t ocrDedupSkippedCount = 0;
        size_t tableNpuSubmitCount = 0;
        size_t tableDedupSkippedCount = 0;
        size_t ocrTimeoutCount = 0;
        size_t ocrBufferedResultHitCount = 0;
        std::vector<double> ocrSubmitAreas;
    };

    /**
     * @brief Process a single page through the pipeline
     */
    PageResult processPage(const PageImage& pageImage);
    PageResult processPage(const PageImage& pageImage, const ExecutionContext& ctx);
    void runLayoutStage(PageContext& pageCtx, const ExecutionContext& ctx);
    void runPlanStage(PageContext& pageCtx, const ExecutionContext& ctx);
    void runOcrTableStage(PageContext& pageCtx, const ExecutionContext& ctx);
    void runFinalizeStage(PageContext& pageCtx, const ExecutionContext& ctx);
    void runTableFinalizeCpuSubstage(PageContext& pageCtx, const ExecutionContext& ctx);
    void runDocumentFinalizeSubstage(PageContext& pageCtx, const ExecutionContext& ctx);

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
    std::vector<ContentElement> runTableRecognition(
        const cv::Mat& image,
        const std::vector<LayoutBox>& tableBoxes,
        int pageIndex,
        const ExecutionContext& ctx
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
    void saveExtractedImages(
        const cv::Mat& image,
        const std::vector<LayoutBox>& figureBoxes,
        int pageIndex,
        std::vector<ContentElement>& elements,
        const ExecutionContext& ctx
    );

    void saveFormulaImages(
        const cv::Mat& image,
        const std::vector<LayoutBox>& equationBoxes,
        int pageIndex,
        std::vector<ContentElement>& elements
    );
    void saveFormulaImages(
        const cv::Mat& image,
        const std::vector<LayoutBox>& equationBoxes,
        int pageIndex,
        std::vector<ContentElement>& elements,
        const ExecutionContext& ctx
    );

    void saveLayoutVisualization(
        const cv::Mat& image,
        const LayoutResult& layoutResult,
        int pageIndex
    );
    void saveLayoutVisualization(
        const cv::Mat& image,
        const LayoutResult& layoutResult,
        int pageIndex,
        const ExecutionContext& ctx
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
        bool& success,
        bool* bufferedHit = nullptr,
        size_t* bufferedOutOfOrderCount = nullptr);
    int64_t allocateOcrTaskId();

    TableResult recognizeTable(const cv::Mat& tableCrop);
    TableRecognizer::NpuStageResult recognizeTableNpuStage(const cv::Mat& tableCrop);
    TableResult finalizeTableRecognizePostprocess(
        const cv::Mat& tableCrop,
        const TableRecognizer::NpuStageResult& npuStage);
    std::string generateTableHtml(const std::vector<TableCell>& cells);
    ContentElement makeTableFallbackElement(
        const LayoutBox& box,
        int pageIndex,
        const std::string& reason) const;
    static std::string tableFallbackMessage(const std::string& reason);

    ExecutionContext makeExecutionContext(const PipelineRunOverrides* overrides) const;
    void resetOcrTransientStateForRun();
    std::mutex& npuSerialMutex();
    bool shouldUsePdfStreaming(const ExecutionContext& ctx) const;
    DocumentResult processRenderedPages(
        std::vector<PageImage> pageImages,
        const ExecutionContext& ctx);
    DocumentResult processPdfSerial(
        PdfRenderer& renderer,
        const std::function<bool(PdfRenderer&, const PdfRenderer::PageVisitor&)>& renderFn,
        const ExecutionContext& ctx);
    DocumentResult processPdfPagePipelineMvp(
        PdfRenderer& renderer,
        const std::function<bool(PdfRenderer&, const PdfRenderer::PageVisitor&)>& renderFn,
        const ExecutionContext& ctx);
    DocumentResult processPdfStreaming(
        PdfRenderer& renderer,
        const std::function<bool(PdfRenderer&, const PdfRenderer::PageVisitor&)>& renderFn,
        const ExecutionContext& ctx);
    DocumentResult processPdfInternal(const std::string& pdfPath, const ExecutionContext& ctx);
    DocumentResult processPdfFromMemoryInternal(
        const uint8_t* data, size_t size, const ExecutionContext& ctx);
    DocumentResult processImageDocumentInternal(
        const cv::Mat& image, int pageIndex, const ExecutionContext& ctx);

    void reportProgress(const std::string& stage, int current, int total);

    PipelineConfig config_;
    bool initialized_ = false;

    // Pipeline components
    std::unique_ptr<PdfRenderer> pdfRenderer_;
    std::unique_ptr<LayoutDetector> layoutDetector_;
    std::unique_ptr<TableRecognizer> tableRecognizer_;
    std::unique_ptr<ocr::OCRPipeline> ocrPipeline_;
    std::mutex npuSerialMutex_;
    std::mutex* externalNpuSerialMutex_ = nullptr;

    OcrSubmitHook ocrSubmitHook_;
    OcrFetchHook ocrFetchHook_;
    TableRecognizeHook tableRecognizeHook_;
    TableHtmlHook tableHtmlHook_;

    struct BufferedOcrResult {
        std::vector<ocr::PipelineOCRResult> results;
        bool success = false;
    };
    mutable std::mutex ocrStateMutex_;
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
