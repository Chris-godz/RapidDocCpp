/**
 * @file doc_pipeline.cpp
 * @brief Main document processing pipeline implementation
 * 
 * Orchestrates PDF rendering → Layout → OCR → Table → Reading Order → Output
 * Uses DXNN-OCR-cpp via submodule for OCR functionality.
 */

#include "pipeline/doc_pipeline.h"
#include "common/logger.h"
#include "common/perf_utils.h"
#include <filesystem>
#include <chrono>
#include <thread>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <condition_variable>
#include <deque>

namespace fs = std::filesystem;

namespace rapid_doc {

namespace {

void finalizeDocumentStats(DocumentResult& result) {
    const double pdfRenderTimeMs = result.stats.pdfRenderTimeMs;
    const double outputGenTimeMs = result.stats.outputGenTimeMs;
    result.stats = accumulateDocumentStageStats(result.pages);
    result.stats.pdfRenderTimeMs = pdfRenderTimeMs;
    result.stats.outputGenTimeMs = outputGenTimeMs;
}

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
    bool submitted = false;
    bool fetched = false;
    bool success = false;
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

struct ExactRoiKey {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;

    bool operator==(const ExactRoiKey& other) const {
        return x == other.x &&
               y == other.y &&
               width == other.width &&
               height == other.height;
    }
};

struct ExactRoiKeyHasher {
    size_t operator()(const ExactRoiKey& key) const {
        size_t seed = static_cast<size_t>(key.x);
        seed = (seed * 1315423911u) ^ static_cast<size_t>(key.y);
        seed = (seed * 1315423911u) ^ static_cast<size_t>(key.width);
        seed = (seed * 1315423911u) ^ static_cast<size_t>(key.height);
        return seed;
    }
};

struct ExactRoiGroup {
    cv::Rect roi;
    size_t canonicalIndex = 0;
    std::vector<size_t> originalIndices;
};

bool prefersCanonicalBox(const LayoutBox& candidate, const LayoutBox& current) {
    if (candidate.confidence != current.confidence) {
        return candidate.confidence > current.confidence;
    }
    return candidate.index < current.index;
}

std::vector<ExactRoiGroup> groupExactRoiDuplicates(
    const std::vector<LayoutBox>& boxes,
    const cv::Size& imageSize)
{
    std::vector<ExactRoiGroup> groups;
    std::unordered_map<ExactRoiKey, size_t, ExactRoiKeyHasher> groupByKey;
    const cv::Rect bounds(0, 0, imageSize.width, imageSize.height);

    for (size_t i = 0; i < boxes.size(); ++i) {
        const cv::Rect roi = boxes[i].toRect() & bounds;
        if (roi.width <= 0 || roi.height <= 0) {
            ExactRoiGroup group;
            group.roi = roi;
            group.canonicalIndex = i;
            group.originalIndices.push_back(i);
            groups.push_back(std::move(group));
            continue;
        }

        const ExactRoiKey key{roi.x, roi.y, roi.width, roi.height};
        const auto [it, inserted] = groupByKey.emplace(key, groups.size());
        if (inserted) {
            ExactRoiGroup group;
            group.roi = roi;
            group.canonicalIndex = i;
            group.originalIndices.push_back(i);
            groups.push_back(std::move(group));
            continue;
        }

        auto& group = groups[it->second];
        group.originalIndices.push_back(i);
        if (prefersCanonicalBox(boxes[i], boxes[group.canonicalIndex])) {
            group.canonicalIndex = i;
        }
    }

    return groups;
}

std::vector<OcrWorkItem> buildOcrWorkItems(
    const cv::Mat& image,
    const std::vector<LayoutBox>& textBoxes,
    int pageIndex)
{
    std::vector<OcrWorkItem> items;
    const auto groups = groupExactRoiDuplicates(textBoxes, image.size());
    items.reserve(groups.size());

    for (const auto& group : groups) {
        const auto& box = textBoxes[group.canonicalIndex];
        OcrWorkItem item;
        item.box = box;
        item.type = (box.category == LayoutCategory::TITLE)
                        ? ContentElement::Type::TITLE
                        : ContentElement::Type::TEXT;
        item.pageIndex = pageIndex;
        item.confidence = box.confidence;
        item.originalIndices = group.originalIndices;

        if (group.roi.width <= 0 || group.roi.height <= 0) {
            item.skipped = true;
        } else {
            item.crop = image(group.roi).clone();
        }
        items.push_back(std::move(item));
    }
    return items;
}

std::string combineOcrTextLines(const std::vector<ocr::PipelineOCRResult>& ocrResults) {
    std::string combined;
    for (const auto& r : ocrResults) {
        if (!combined.empty()) {
            combined += "\n";
        }
        combined += r.text;
    }
    return combined;
}

void matchTableOcrToCells(
    TableResult& tableResult,
    const std::vector<ocr::PipelineOCRResult>& ocrBoxes)
{
    if (tableResult.cells.empty() || ocrBoxes.empty()) {
        return;
    }

    for (const auto& ocrRes : ocrBoxes) {
        if (ocrRes.text.empty()) {
            continue;
        }

        cv::Rect ocrRect = ocrRes.getBoundingRect();
        int bestCell = -1;
        float bestOverlap = 0.0f;

        for (size_t ci = 0; ci < tableResult.cells.size(); ++ci) {
            auto& c = tableResult.cells[ci];
            cv::Rect cellRect(static_cast<int>(c.x0), static_cast<int>(c.y0),
                              static_cast<int>(c.x1 - c.x0),
                              static_cast<int>(c.y1 - c.y0));
            cv::Rect inter = ocrRect & cellRect;
            if (inter.width <= 0 || inter.height <= 0) {
                continue;
            }
            float interArea = static_cast<float>(inter.area());
            float ocrArea = static_cast<float>(std::max(1, ocrRect.area()));
            float overlap = interArea / ocrArea;
            if (overlap > bestOverlap) {
                bestOverlap = overlap;
                bestCell = static_cast<int>(ci);
            }
        }

        if (bestCell >= 0 && bestOverlap > 0.3f) {
            auto& cell = tableResult.cells[bestCell];
            if (!cell.content.empty()) {
                cell.content += "\n";
            }
            cell.content += ocrRes.text;
        }
    }
}

} // namespace

DocPipeline::DocPipeline(const PipelineConfig& config)
    : config_(config)
{
}

DocPipeline::~DocPipeline() {
    if (ocrPipeline_) {
        ocrPipeline_->stop();
    }
}

bool DocPipeline::initialize() {
    LOG_INFO("Initializing RapidDoc pipeline...");
    config_.show();

    // Validate configuration
    std::string err = config_.validate();
    if (!err.empty()) {
        LOG_ERROR("Configuration validation failed: {}", err);
        return false;
    }

    // Initialize PDF renderer
    if (config_.stages.enablePdfRender) {
        PdfRenderConfig pdfCfg;
        pdfCfg.dpi = config_.runtime.pdfDpi;
        pdfCfg.maxPages = config_.runtime.maxPages;
        pdfCfg.startPageId = config_.runtime.startPageId;
        pdfCfg.endPageId = config_.runtime.endPageId;
        pdfCfg.maxConcurrentRenders = config_.runtime.maxConcurrentPages;
        pdfRenderer_ = std::make_unique<PdfRenderer>(pdfCfg);
        LOG_INFO("PDF renderer initialized");
    }

    // Initialize Layout detector
    if (config_.stages.enableLayout) {
        LayoutDetectorConfig layoutCfg;
        layoutCfg.dxnnModelPath = config_.models.layoutDxnnModel;
        layoutCfg.onnxSubModelPath = config_.models.layoutOnnxSubModel;
        layoutCfg.inputSize = config_.runtime.layoutInputSize;
        layoutCfg.confThreshold = config_.runtime.layoutConfThreshold;
        layoutCfg.deviceId = config_.runtime.deviceId;
        layoutDetector_ = std::make_unique<LayoutDetector>(layoutCfg);
        if (!layoutDetector_->initialize()) {
            LOG_ERROR("Failed to initialize layout detector");
            return false;
        }
        LOG_INFO("Layout detector initialized");
    }

    // Initialize Table recognizer (wired tables only)
    if (config_.stages.enableWiredTable) {
        TableRecognizerConfig tableCfg;
        tableCfg.unetDxnnModelPath = config_.models.tableUnetDxnnModel;
        tableCfg.threshold = config_.runtime.tableConfThreshold;
        tableCfg.deviceId = config_.runtime.deviceId;
        tableRecognizer_ = std::make_unique<TableRecognizer>(tableCfg);
        if (!tableRecognizer_->initialize()) {
            LOG_ERROR("Failed to initialize table recognizer");
            return false;
        }
        LOG_INFO("Table recognizer initialized (wired tables only)");
    }

    // Initialize OCR pipeline (from DXNN-OCR-cpp)
    if (config_.stages.enableOcr) {
        ocr::OCRPipelineConfig ocrCfg;
        ocrCfg.deviceId = config_.runtime.deviceId;

        // Detection model paths — use only 640 model to conserve NPU memory
        ocrCfg.detectorConfig.model640Path = config_.models.ocrModelDir + "/det_v5_640.dxnn";
        ocrCfg.detectorConfig.model960Path = "";
        ocrCfg.detectorConfig.sizeThreshold = 99999;

        // Recognition model paths
        std::string mdir = config_.models.ocrModelDir;
        ocrCfg.recognizerConfig.modelPaths = {
            {3,  mdir + "/rec_v5_ratio_3.dxnn"},
            {5,  mdir + "/rec_v5_ratio_5.dxnn"},
            {10, mdir + "/rec_v5_ratio_10.dxnn"},
            {15, mdir + "/rec_v5_ratio_15.dxnn"},
            {25, mdir + "/rec_v5_ratio_25.dxnn"},
            {35, mdir + "/rec_v5_ratio_35.dxnn"},
        };
        ocrCfg.recognizerConfig.dictPath = config_.models.ocrDictPath;

        // Disable heavy document-level preprocessing for per-region OCR
        ocrCfg.useDocPreprocessing = false;
        ocrCfg.useClassification = false;
        ocrCfg.enableVisualization = false;

        ocrPipeline_ = std::make_unique<ocr::OCRPipeline>(ocrCfg);
        if (!ocrPipeline_->initialize()) {
            LOG_ERROR("Failed to initialize OCR pipeline");
            return false;
        }
        ocrPipeline_->start();
        LOG_INFO("OCR pipeline initialized (DXNN-OCR-cpp)");
    }

    // Create output directory
    if (!fs::exists(config_.runtime.outputDir)) {
        fs::create_directories(config_.runtime.outputDir);
    }

    initialized_ = true;
    LOG_INFO("RapidDoc pipeline initialized successfully");
    return true;
}

DocPipeline::ExecutionContext DocPipeline::makeExecutionContext(
    const PipelineRunOverrides* overrides) const
{
    ExecutionContext ctx{config_.stages, config_.runtime};
    if (overrides == nullptr) {
        return ctx;
    }

    if (overrides->outputDir.has_value()) ctx.runtime.outputDir = *overrides->outputDir;
    if (overrides->saveImages.has_value()) ctx.runtime.saveImages = *overrides->saveImages;
    if (overrides->saveVisualization.has_value()) ctx.runtime.saveVisualization = *overrides->saveVisualization;
    if (overrides->startPageId.has_value()) ctx.runtime.startPageId = *overrides->startPageId;
    if (overrides->endPageId.has_value()) ctx.runtime.endPageId = *overrides->endPageId;
    if (overrides->maxPages.has_value()) ctx.runtime.maxPages = *overrides->maxPages;
    if (overrides->enableFormula.has_value()) ctx.stages.enableFormula = *overrides->enableFormula;
    if (overrides->enableWiredTable.has_value()) ctx.stages.enableWiredTable = *overrides->enableWiredTable;
    if (overrides->enableMarkdownOutput.has_value()) {
        ctx.stages.enableMarkdownOutput = *overrides->enableMarkdownOutput;
    }
    return ctx;
}

void DocPipeline::resetOcrTransientStateForRun() {
    std::lock_guard<std::mutex> lock(ocrStateMutex_);
    bufferedOcrResults_.clear();
    timedOutOcrTaskIds_.clear();
}

std::mutex& DocPipeline::npuSerialMutex() {
    if (externalNpuSerialMutex_ != nullptr) {
        return *externalNpuSerialMutex_;
    }
    return npuSerialMutex_;
}

DocumentResult DocPipeline::processPdf(const std::string& pdfPath) {
    return processPdfInternal(pdfPath, makeExecutionContext(nullptr));
}

DocumentResult DocPipeline::processPdfWithOverrides(
    const std::string& pdfPath,
    const PipelineRunOverrides& overrides)
{
    return processPdfInternal(pdfPath, makeExecutionContext(&overrides));
}

bool DocPipeline::shouldUsePdfStreaming(const ExecutionContext& ctx) const {
    int requestedPages = 0;
    if (ctx.runtime.maxPages > 0) {
        requestedPages = ctx.runtime.maxPages;
    } else if (ctx.runtime.endPageId >= ctx.runtime.startPageId &&
               ctx.runtime.endPageId >= 0) {
        requestedPages = ctx.runtime.endPageId - ctx.runtime.startPageId + 1;
    }

    return requestedPages <= 0 || requestedPages > 2;
}

DocumentResult DocPipeline::processRenderedPages(
    std::vector<PageImage> pageImages,
    const ExecutionContext& ctx)
{
    DocumentResult result;
    result.totalPages = static_cast<int>(pageImages.size());

    for (size_t i = 0; i < pageImages.size(); ++i) {
        reportProgress(
            "Processing",
            static_cast<int>(i + 1),
            static_cast<int>(pageImages.size()));
        result.stats.pdfRenderTimeMs += pageImages[i].renderTimeMs;
        PageResult pageResult = processPage(pageImages[i], ctx);
        result.pages.push_back(std::move(pageResult));
        result.processedPages++;
    }

    return result;
}

DocumentResult DocPipeline::processPdfStreaming(
    PdfRenderer& renderer,
    const std::function<bool(PdfRenderer&, const PdfRenderer::PageVisitor&)>& renderFn,
    const ExecutionContext& ctx)
{
    DocumentResult result;

    struct StreamState {
        std::mutex mutex;
        std::condition_variable ready;
        std::condition_variable space;
        std::deque<PageImage> queue;
        bool producerDone = false;
        bool cancelled = false;
        std::exception_ptr producerError;
    } stream;

    const size_t maxQueuedPages = static_cast<size_t>(std::max(2, ctx.runtime.maxConcurrentPages));
    std::thread producer([&]() {
        try {
            renderFn(renderer, [&](PageImage&& page) {
                std::unique_lock<std::mutex> lock(stream.mutex);
                stream.space.wait(lock, [&]() {
                    return stream.cancelled || stream.queue.size() < maxQueuedPages;
                });
                if (stream.cancelled) {
                    return false;
                }
                stream.queue.push_back(std::move(page));
                lock.unlock();
                stream.ready.notify_one();
                return true;
            });
        } catch (...) {
            std::lock_guard<std::mutex> lock(stream.mutex);
            stream.producerError = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(stream.mutex);
            stream.producerDone = true;
        }
        stream.ready.notify_all();
    });

    try {
        while (true) {
            PageImage pageImage;
            {
                std::unique_lock<std::mutex> lock(stream.mutex);
                stream.ready.wait(lock, [&]() {
                    return !stream.queue.empty() || stream.producerDone;
                });

                if (stream.queue.empty()) {
                    break;
                }

                pageImage = std::move(stream.queue.front());
                stream.queue.pop_front();
            }
            stream.space.notify_one();

            result.stats.pdfRenderTimeMs += pageImage.renderTimeMs;
            PageResult pageResult = processPage(pageImage, ctx);
            result.pages.push_back(std::move(pageResult));
            result.processedPages++;
            result.totalPages++;
        }
    } catch (...) {
        {
            std::lock_guard<std::mutex> lock(stream.mutex);
            stream.cancelled = true;
        }
        stream.space.notify_all();
        producer.join();
        throw;
    }

    producer.join();
    if (stream.producerError) {
        std::rethrow_exception(stream.producerError);
    }

    return result;
}

DocumentResult DocPipeline::processPdfInternal(
    const std::string& pdfPath,
    const ExecutionContext& ctx)
{
    LOG_INFO("Processing PDF: {}", pdfPath);
    auto startTime = std::chrono::steady_clock::now();

    DocumentResult result;

    if (!initialized_) {
        LOG_ERROR("Pipeline not initialized");
        return result;
    }

    reportProgress("PDF Render", 0, 1);
    if (ctx.stages.enablePdfRender) {
        PdfRenderConfig pdfCfg;
        pdfCfg.dpi = ctx.runtime.pdfDpi;
        pdfCfg.maxPages = ctx.runtime.maxPages;
        pdfCfg.startPageId = ctx.runtime.startPageId;
        pdfCfg.endPageId = ctx.runtime.endPageId;
        pdfCfg.maxConcurrentRenders = ctx.runtime.maxConcurrentPages;
        PdfRenderer renderer(pdfCfg);
        if (shouldUsePdfStreaming(ctx)) {
            result = processPdfStreaming(
                renderer,
                [&pdfPath](PdfRenderer& activeRenderer, const PdfRenderer::PageVisitor& visitor) {
                    return activeRenderer.renderFileStreaming(pdfPath, visitor);
                },
                ctx);
        } else {
            result = processRenderedPages(renderer.renderFile(pdfPath), ctx);
        }
    }

    if (result.pages.empty()) {
        LOG_WARN("No pages rendered from PDF");
        return result;
    }

    LOG_INFO("Rendered {} pages from PDF", result.pages.size());

    finalizeDocumentStats(result);

    reportProgress("Output", 0, 1);
    auto outputStart = std::chrono::steady_clock::now();
    if (ctx.stages.enableMarkdownOutput) {
        result.markdown = markdownWriter_.generate(result);
    }
    result.contentListJson = contentListWriter_.generate(result);
    auto outputEnd = std::chrono::steady_clock::now();
    result.stats.outputGenTimeMs =
        std::chrono::duration<double, std::milli>(outputEnd - outputStart).count();

    auto endTime = std::chrono::steady_clock::now();
    result.totalTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    for (const auto& page : result.pages) {
        for (const auto& elem : page.elements) {
            if (elem.skipped) result.skippedElements++;
        }
    }

    LOG_INFO("Document processing complete: {} pages, {} skipped elements, {:.1f}ms",
             result.processedPages, result.skippedElements, result.totalTimeMs);

    return result;
}

DocumentResult DocPipeline::processPdfFromMemory(const uint8_t* data, size_t size) {
    return processPdfFromMemoryInternal(data, size, makeExecutionContext(nullptr));
}

DocumentResult DocPipeline::processPdfFromMemoryWithOverrides(
    const uint8_t* data,
    size_t size,
    const PipelineRunOverrides& overrides)
{
    return processPdfFromMemoryInternal(data, size, makeExecutionContext(&overrides));
}

DocumentResult DocPipeline::processPdfFromMemoryInternal(
    const uint8_t* data,
    size_t size,
    const ExecutionContext& ctx)
{
    LOG_INFO("Processing PDF from memory: {} bytes", size);

    DocumentResult result;
    if (!initialized_) {
        LOG_ERROR("Pipeline not initialized");
        return result;
    }

    auto startTime = std::chrono::steady_clock::now();

    if (ctx.stages.enablePdfRender) {
        PdfRenderConfig pdfCfg;
        pdfCfg.dpi = ctx.runtime.pdfDpi;
        pdfCfg.maxPages = ctx.runtime.maxPages;
        pdfCfg.startPageId = ctx.runtime.startPageId;
        pdfCfg.endPageId = ctx.runtime.endPageId;
        pdfCfg.maxConcurrentRenders = ctx.runtime.maxConcurrentPages;
        PdfRenderer renderer(pdfCfg);
        if (shouldUsePdfStreaming(ctx)) {
            result = processPdfStreaming(
                renderer,
                [data, size](PdfRenderer& activeRenderer, const PdfRenderer::PageVisitor& visitor) {
                    return activeRenderer.renderFromMemoryStreaming(data, size, visitor);
                },
                ctx);
        } else {
            result = processRenderedPages(renderer.renderFromMemory(data, size), ctx);
        }
    }

    finalizeDocumentStats(result);

    auto outputStart = std::chrono::steady_clock::now();
    if (ctx.stages.enableMarkdownOutput) {
        result.markdown = markdownWriter_.generate(result);
    }
    result.contentListJson = contentListWriter_.generate(result);
    auto outputEnd = std::chrono::steady_clock::now();
    result.stats.outputGenTimeMs =
        std::chrono::duration<double, std::milli>(outputEnd - outputStart).count();

    for (const auto& page : result.pages) {
        for (const auto& elem : page.elements) {
            if (elem.skipped) result.skippedElements++;
        }
    }

    auto endTime = std::chrono::steady_clock::now();
    result.totalTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    return result;
}

DocumentResult DocPipeline::processImageDocument(const cv::Mat& image, int pageIndex) {
    return processImageDocumentInternal(image, pageIndex, makeExecutionContext(nullptr));
}

DocumentResult DocPipeline::processImageDocumentWithOverrides(
    const cv::Mat& image,
    int pageIndex,
    const PipelineRunOverrides& overrides)
{
    return processImageDocumentInternal(image, pageIndex, makeExecutionContext(&overrides));
}

DocumentResult DocPipeline::processImageDocumentInternal(
    const cv::Mat& image,
    int pageIndex,
    const ExecutionContext& ctx)
{
    auto startTime = std::chrono::steady_clock::now();

    DocumentResult result;
    if (!initialized_) {
        LOG_ERROR("Pipeline not initialized");
        return result;
    }

    PageImage pageImage;
    pageImage.image = image;
    pageImage.pageIndex = pageIndex;
    pageImage.dpi = ctx.runtime.pdfDpi;
    pageImage.scaleFactor = 1.0;
    pageImage.pdfWidth = image.cols;
    pageImage.pdfHeight = image.rows;
    PageResult pageResult = processPage(pageImage, ctx);

    result.pages.push_back(std::move(pageResult));
    result.totalPages = 1;
    result.processedPages = 1;

    finalizeDocumentStats(result);

    auto outputStart = std::chrono::steady_clock::now();
    if (ctx.stages.enableMarkdownOutput) {
        result.markdown = markdownWriter_.generate(result);
    }
    result.contentListJson = contentListWriter_.generate(result);
    auto outputEnd = std::chrono::steady_clock::now();
    result.stats.outputGenTimeMs =
        std::chrono::duration<double, std::milli>(outputEnd - outputStart).count();

    for (const auto& elem : result.pages.front().elements) {
        if (elem.skipped) result.skippedElements++;
    }

    auto endTime = std::chrono::steady_clock::now();
    result.totalTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();
    return result;
}

PageResult DocPipeline::processImage(const cv::Mat& image, int pageIndex) {
    LOG_INFO("Processing image: {}x{}, page {}", image.cols, image.rows, pageIndex);

    PageImage pageImage;
    pageImage.image = image;
    pageImage.pageIndex = pageIndex;
    pageImage.dpi = config_.runtime.pdfDpi;
    pageImage.scaleFactor = 1.0;
    pageImage.pdfWidth = image.cols;
    pageImage.pdfHeight = image.rows;

    return processPage(pageImage);
}

PageResult DocPipeline::processPage(const PageImage& pageImage) {
    return processPage(pageImage, makeExecutionContext(nullptr));
}

PageResult DocPipeline::processPage(const PageImage& pageImage, const ExecutionContext& ctx) {
    auto startTime = std::chrono::steady_clock::now();
    PageResult result;
    result.pageIndex = pageImage.pageIndex;
    result.pageWidth = pageImage.image.cols;
    result.pageHeight = pageImage.image.rows;

    const cv::Mat& image = pageImage.image;
    int pageWidth = image.cols;
    int pageHeight = image.rows;

    std::vector<LayoutBox> textBoxes;
    std::vector<LayoutBox> tableBoxes;
    std::vector<LayoutBox> figureBoxes;
    std::vector<LayoutBox> equationBoxes;
    std::vector<LayoutBox> unsupportedBoxes;

    double npuLockWaitTotalMs = 0.0;
    double npuLockHoldTotalMs = 0.0;
    double npuSerialTotalMs = 0.0;
    double cpuOnlyTotalMs = 0.0;
    size_t textBoxesRawCount = 0;
    size_t textBoxesAfterDedupCount = 0;
    size_t tableBoxesRawCount = 0;
    size_t tableBoxesAfterDedupCount = 0;
    size_t ocrSubmitCount = 0;
    size_t ocrDedupSkippedCount = 0;
    size_t tableNpuSubmitCount = 0;
    size_t tableDedupSkippedCount = 0;
    size_t ocrTimeoutCount = 0;
    size_t ocrBufferedResultHitCount = 0;

    auto runNpuSerialized = [&](auto&& fn) -> double {
        auto lockWaitStart = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> npuLock(npuSerialMutex());
        auto lockAcquired = std::chrono::steady_clock::now();
        npuLockWaitTotalMs +=
            std::chrono::duration<double, std::milli>(lockAcquired - lockWaitStart).count();

        auto serialStart = lockAcquired;
        fn();
        auto serialEnd = std::chrono::steady_clock::now();

        const double serialMs =
            std::chrono::duration<double, std::milli>(serialEnd - serialStart).count();
        npuSerialTotalMs += serialMs;
        npuLockHoldTotalMs +=
            std::chrono::duration<double, std::milli>(serialEnd - lockAcquired).count();
        return serialMs;
    };

    // Step 1: Layout detection (NPU, serialized)
    if (layoutDetector_ && ctx.stages.enableLayout) {
        runNpuSerialized([&]() {
            auto layoutStart = std::chrono::steady_clock::now();
            result.layoutResult = layoutDetector_->detect(image);
            auto layoutEnd = std::chrono::steady_clock::now();
            result.layoutResult.inferenceTimeMs =
                std::chrono::duration<double, std::milli>(layoutEnd - layoutStart).count();
            result.stats.layoutTimeMs = result.layoutResult.inferenceTimeMs;
        });

        LOG_DEBUG("Page {}: detected {} layout boxes",
                  pageImage.pageIndex, result.layoutResult.boxes.size());
    }

    // Derive layout buckets from structure (CPU-only, outside NPU lock).
    {
        auto bucketStart = std::chrono::steady_clock::now();
        textBoxes = result.layoutResult.getTextBoxes();
        tableBoxes = result.layoutResult.getTableBoxes();
        figureBoxes = result.layoutResult.getBoxesByCategory(LayoutCategory::FIGURE);
        equationBoxes = result.layoutResult.getEquationBoxes();
        unsupportedBoxes = result.layoutResult.getUnsupportedBoxes();
        textBoxesRawCount = textBoxes.size();
        tableBoxesRawCount = tableBoxes.size();
        auto bucketEnd = std::chrono::steady_clock::now();
        cpuOnlyTotalMs +=
            std::chrono::duration<double, std::milli>(bucketEnd - bucketStart).count();
    }

    // OCR on text regions: move ROI crop, element assembly, and text concatenation to CPU-only.
    if (ctx.stages.enableOcr) {
        std::vector<OcrWorkItem> ocrWorkItems;
        std::vector<size_t> ocrWorkItemForOriginal;
        std::vector<size_t> ocrCanonicalWorkItem;
        {
            auto prepStart = std::chrono::steady_clock::now();
            ocrWorkItems = buildOcrWorkItems(image, textBoxes, pageImage.pageIndex);
            textBoxesAfterDedupCount = ocrWorkItems.size();
            if (textBoxesRawCount > textBoxesAfterDedupCount) {
                ocrDedupSkippedCount += (textBoxesRawCount - textBoxesAfterDedupCount);
            }
            ocrWorkItemForOriginal.assign(textBoxes.size(), 0);
            ocrCanonicalWorkItem.assign(ocrWorkItems.size(), 0);
            std::unordered_map<ExactRoiKey, size_t, ExactRoiKeyHasher> ocrCanonicalByKey;
            const cv::Rect bounds(0, 0, image.cols, image.rows);
            for (size_t workIndex = 0; workIndex < ocrWorkItems.size(); ++workIndex) {
                ocrCanonicalWorkItem[workIndex] = workIndex;
                const cv::Rect roi = ocrWorkItems[workIndex].box.toRect() & bounds;
                if (roi.width > 0 && roi.height > 0) {
                    const ExactRoiKey key{roi.x, roi.y, roi.width, roi.height};
                    const auto [it, inserted] = ocrCanonicalByKey.emplace(key, workIndex);
                    if (!inserted) {
                        ocrCanonicalWorkItem[workIndex] = it->second;
                        ++ocrDedupSkippedCount;
                    }
                }
                for (size_t originalIndex : ocrWorkItems[workIndex].originalIndices) {
                    ocrWorkItemForOriginal[originalIndex] = workIndex;
                }
            }
            auto prepEnd = std::chrono::steady_clock::now();
            cpuOnlyTotalMs +=
                std::chrono::duration<double, std::milli>(prepEnd - prepStart).count();
        }

        std::vector<OcrFetchResult> fetchResults(ocrWorkItems.size());
        result.stats.ocrTimeMs = runNpuSerialized([&]() {
            for (size_t i = 0; i < ocrWorkItems.size(); ++i) {
                const auto& item = ocrWorkItems[i];
                const size_t canonicalWorkIndex = ocrCanonicalWorkItem[i];
                if (canonicalWorkIndex != i) {
                    fetchResults[i] = fetchResults[canonicalWorkIndex];
                    continue;
                }
                if (item.skipped || item.crop.empty()) {
                    continue;
                }

                const int64_t taskId = allocateOcrTaskId();
                auto& fetch = fetchResults[i];
                fetch.submitted = false;
                fetch.fetched = false;
                fetch.success = false;
                fetch.results.clear();

                fetch.submitted = submitOcrTask(item.crop, taskId);
                if (!fetch.submitted) {
                    continue;
                }
                ++ocrSubmitCount;
                bool bufferedHit = false;
                fetch.fetched =
                    waitForOcrResult(taskId, fetch.results, fetch.success, &bufferedHit);
                if (bufferedHit) {
                    ++ocrBufferedResultHitCount;
                }
                if (!fetch.fetched) {
                    ++ocrTimeoutCount;
                    LOG_WARN("OCR timeout for task {}", taskId);
                }
            }
        });

        {
            auto assembleStart = std::chrono::steady_clock::now();
            for (size_t originalIndex = 0; originalIndex < textBoxes.size(); ++originalIndex) {
                const auto& box = textBoxes[originalIndex];
                const size_t workIndex = ocrWorkItemForOriginal[originalIndex];
                const auto& item = ocrWorkItems[workIndex];
                ContentElement elem;
                elem.type = (box.category == LayoutCategory::TITLE)
                                ? ContentElement::Type::TITLE
                                : ContentElement::Type::TEXT;
                elem.layoutBox = box;
                elem.confidence = box.confidence;
                elem.pageIndex = pageImage.pageIndex;

                if (item.skipped) {
                    elem.skipped = true;
                    result.elements.push_back(std::move(elem));
                    continue;
                }

                const auto& fetch = fetchResults[workIndex];
                if (fetch.fetched && fetch.success && !fetch.results.empty()) {
                    elem.text = combineOcrTextLines(fetch.results);
                }
                result.elements.push_back(std::move(elem));
            }
            auto assembleEnd = std::chrono::steady_clock::now();
            cpuOnlyTotalMs +=
                std::chrono::duration<double, std::milli>(assembleEnd - assembleStart).count();
        }
    }

    // Table recognition remains NPU-serialized.
    if (ctx.stages.enableWiredTable) {
        std::vector<TableWorkItem> tableWorkItems;
        std::vector<size_t> tableWorkItemForOriginal;
        std::vector<size_t> tableCanonicalWorkItem;
        {
            auto prepStart = std::chrono::steady_clock::now();
            const auto groups = groupExactRoiDuplicates(tableBoxes, image.size());
            tableWorkItems.reserve(groups.size());
            tableWorkItemForOriginal.assign(tableBoxes.size(), 0);
            tableCanonicalWorkItem.assign(groups.size(), 0);
            std::unordered_map<ExactRoiKey, size_t, ExactRoiKeyHasher> tableCanonicalByKey;
            const cv::Rect bounds(0, 0, image.cols, image.rows);
            for (const auto& group : groups) {
                const auto& box = tableBoxes[group.canonicalIndex];
                TableWorkItem item;
                item.box = box;
                item.pageIndex = pageImage.pageIndex;
                item.originalIndices = group.originalIndices;
                if (group.roi.width <= 0 || group.roi.height <= 0) {
                    item.invalidRoi = true;
                } else {
                    item.crop = image(group.roi).clone();
                }
                const size_t workIndex = tableWorkItems.size();
                tableCanonicalWorkItem[workIndex] = workIndex;
                const cv::Rect roi = box.toRect() & bounds;
                if (roi.width > 0 && roi.height > 0) {
                    const ExactRoiKey key{roi.x, roi.y, roi.width, roi.height};
                    const auto [it, inserted] = tableCanonicalByKey.emplace(key, workIndex);
                    if (!inserted) {
                        tableCanonicalWorkItem[workIndex] = it->second;
                        ++tableDedupSkippedCount;
                    }
                }
                for (size_t originalIndex : group.originalIndices) {
                    tableWorkItemForOriginal[originalIndex] = workIndex;
                }
                tableWorkItems.push_back(std::move(item));
            }
            tableBoxesAfterDedupCount = tableWorkItems.size();
            if (tableBoxesRawCount > tableBoxesAfterDedupCount) {
                tableDedupSkippedCount += (tableBoxesRawCount - tableBoxesAfterDedupCount);
            }
            auto prepEnd = std::chrono::steady_clock::now();
            cpuOnlyTotalMs +=
                std::chrono::duration<double, std::milli>(prepEnd - prepStart).count();
        }

        std::vector<TableNpuResult> tableNpuResults;
        const double tableNpuStageMs = runNpuSerialized([&]() {
            tableNpuResults.reserve(tableWorkItems.size());
            for (size_t i = 0; i < tableWorkItems.size(); ++i) {
                const auto& item = tableWorkItems[i];
                TableNpuResult npuResult;
                npuResult.box = item.box;
                npuResult.pageIndex = item.pageIndex;
                const size_t canonicalWorkIndex = tableCanonicalWorkItem[i];
                if (canonicalWorkIndex != i) {
                    npuResult = tableNpuResults[canonicalWorkIndex];
                    npuResult.box = item.box;
                    npuResult.pageIndex = item.pageIndex;
                    tableNpuResults.push_back(std::move(npuResult));
                    continue;
                }

                if (item.invalidRoi || item.crop.empty()) {
                    npuResult.hasFallback = true;
                    npuResult.fallbackReason = "invalid_table_bbox";
                    tableNpuResults.push_back(std::move(npuResult));
                    continue;
                }
                ++tableNpuSubmitCount;

                if (tableRecognizeHook_) {
                    npuResult.tableResult = recognizeTable(item.crop);
                    npuResult.hasTableResult = true;
                    if (!npuResult.tableResult.supported) {
                        npuResult.hasFallback = true;
                        npuResult.fallbackReason =
                            (npuResult.tableResult.type == TableType::WIRELESS)
                                ? "wireless_table"
                                : "table_model_unavailable";
                        tableNpuResults.push_back(std::move(npuResult));
                        continue;
                    }

                    if (npuResult.tableResult.cells.empty()) {
                        npuResult.hasFallback = true;
                        npuResult.fallbackReason = "no_cell_table";
                        tableNpuResults.push_back(std::move(npuResult));
                        continue;
                    }
                } else {
                    npuResult.npuStage = recognizeTableNpuStage(item.crop);
                    if (!npuResult.npuStage.supported) {
                        npuResult.hasFallback = true;
                        npuResult.fallbackReason =
                            (npuResult.npuStage.type == TableType::WIRELESS)
                                ? "wireless_table"
                                : "table_model_unavailable";
                        tableNpuResults.push_back(std::move(npuResult));
                        continue;
                    }
                }

                tableNpuResults.push_back(std::move(npuResult));
            }
        });

        {
            auto postprocessStart = std::chrono::steady_clock::now();
            for (size_t i = 0; i < tableNpuResults.size(); ++i) {
                auto& npuResult = tableNpuResults[i];
                if (npuResult.hasFallback || npuResult.hasTableResult) {
                    continue;
                }

                npuResult.tableResult =
                    finalizeTableRecognizePostprocess(tableWorkItems[i].crop, npuResult.npuStage);
                npuResult.hasTableResult = true;

                if (!npuResult.tableResult.supported) {
                    npuResult.hasFallback = true;
                    npuResult.fallbackReason =
                        (npuResult.tableResult.type == TableType::WIRELESS)
                            ? "wireless_table"
                            : "table_model_unavailable";
                    continue;
                }

                if (npuResult.tableResult.cells.empty()) {
                    npuResult.hasFallback = true;
                    npuResult.fallbackReason = "no_cell_table";
                    continue;
                }
            }
            auto postprocessEnd = std::chrono::steady_clock::now();
            cpuOnlyTotalMs +=
                std::chrono::duration<double, std::milli>(postprocessEnd - postprocessStart).count();
        }

        double tableOcrNpuMs = 0.0;
        const bool tableOcrEnabled =
            ctx.stages.enableOcr && (ocrPipeline_ || (ocrSubmitHook_ && ocrFetchHook_));
        if (tableOcrEnabled) {
            tableOcrNpuMs = runNpuSerialized([&]() {
                for (size_t i = 0; i < tableNpuResults.size(); ++i) {
                    auto& npuResult = tableNpuResults[i];
                    if (npuResult.hasFallback || !npuResult.hasTableResult) {
                        continue;
                    }
                    const size_t canonicalWorkIndex = tableCanonicalWorkItem[i];
                    if (canonicalWorkIndex != i) {
                        npuResult.ocrBoxes = tableNpuResults[canonicalWorkIndex].ocrBoxes;
                        continue;
                    }

                    const int64_t ocrTaskId = allocateOcrTaskId();
                    if (submitOcrTask(tableWorkItems[i].crop, ocrTaskId)) {
                        ++ocrSubmitCount;
                        bool ok = false;
                        bool bufferedHit = false;
                        const bool fetched = waitForOcrResult(
                            ocrTaskId, npuResult.ocrBoxes, ok, &bufferedHit);
                        if (bufferedHit) {
                            ++ocrBufferedResultHitCount;
                        }
                        if (!(fetched && ok)) {
                            if (!fetched) {
                                ++ocrTimeoutCount;
                            }
                            npuResult.ocrBoxes.clear();
                        }
                    }
                }
            }
            );
        }
        result.stats.tableTimeMs = tableNpuStageMs + tableOcrNpuMs;

        std::vector<ContentElement> tableElements;
        {
            auto assembleStart = std::chrono::steady_clock::now();
            std::vector<ContentElement> uniqueTableElements;
            uniqueTableElements.reserve(tableNpuResults.size());
            for (auto& npuResult : tableNpuResults) {
                if (npuResult.hasFallback) {
                    uniqueTableElements.push_back(makeTableFallbackElement(
                        npuResult.box, npuResult.pageIndex, npuResult.fallbackReason));
                    continue;
                }

                matchTableOcrToCells(npuResult.tableResult, npuResult.ocrBoxes);

                ContentElement elem;
                elem.type = ContentElement::Type::TABLE;
                elem.layoutBox = npuResult.box;
                elem.pageIndex = npuResult.pageIndex;

                try {
                    elem.html = generateTableHtml(npuResult.tableResult.cells);
                } catch (const std::exception& ex) {
                    LOG_WARN("Illegal table structure at page {}: {}",
                             npuResult.pageIndex, ex.what());
                    uniqueTableElements.push_back(makeTableFallbackElement(
                        npuResult.box, npuResult.pageIndex, "illegal_table_structure"));
                    continue;
                }

                if (elem.html.empty()) {
                    uniqueTableElements.push_back(makeTableFallbackElement(
                        npuResult.box, npuResult.pageIndex, "empty_table_html"));
                    continue;
                }

                elem.skipped = false;
                uniqueTableElements.push_back(std::move(elem));
            }

            tableElements.reserve(tableBoxes.size());
            for (size_t originalIndex = 0; originalIndex < tableBoxes.size(); ++originalIndex) {
                ContentElement elem = uniqueTableElements[tableWorkItemForOriginal[originalIndex]];
                elem.layoutBox = tableBoxes[originalIndex];
                elem.pageIndex = pageImage.pageIndex;
                tableElements.push_back(std::move(elem));
            }
            auto assembleEnd = std::chrono::steady_clock::now();
            cpuOnlyTotalMs +=
                std::chrono::duration<double, std::milli>(assembleEnd - assembleStart).count();
        }

        result.elements.insert(result.elements.end(), tableElements.begin(), tableElements.end());
    }

    result.stats.npuLockWaitTimeMs = npuLockWaitTotalMs;
    result.stats.npuLockHoldTimeMs = npuLockHoldTotalMs;
    result.stats.npuSerialTimeMs = npuSerialTotalMs;
    result.stats.textBoxesRawCount = static_cast<double>(textBoxesRawCount);
    result.stats.textBoxesAfterDedupCount = static_cast<double>(textBoxesAfterDedupCount);
    result.stats.tableBoxesRawCount = static_cast<double>(tableBoxesRawCount);
    result.stats.tableBoxesAfterDedupCount = static_cast<double>(tableBoxesAfterDedupCount);
    result.stats.ocrSubmitCount = static_cast<double>(ocrSubmitCount);
    result.stats.ocrDedupSkippedCount = static_cast<double>(ocrDedupSkippedCount);
    result.stats.tableNpuSubmitCount = static_cast<double>(tableNpuSubmitCount);
    result.stats.tableDedupSkippedCount = static_cast<double>(tableDedupSkippedCount);
    result.stats.ocrTimeoutCount = static_cast<double>(ocrTimeoutCount);
    result.stats.ocrBufferedResultHitCount = static_cast<double>(ocrBufferedResultHitCount);

    // CPU-only region (safe to execute outside NPU serial lock).
    auto cpuStart = std::chrono::steady_clock::now();

    if (ctx.runtime.saveVisualization && ctx.stages.enableLayout) {
        saveLayoutVisualization(image, result.layoutResult, pageImage.pageIndex, ctx);
    }

    // Handle figure/image regions (CPU)
    auto figureStart = std::chrono::steady_clock::now();
    saveExtractedImages(image, figureBoxes, pageImage.pageIndex, result.elements, ctx);
    auto figureEnd = std::chrono::steady_clock::now();
    result.stats.figureTimeMs =
        std::chrono::duration<double, std::milli>(figureEnd - figureStart).count();

    // Formula: save crop as image (CPU fallback path, no NPU call).
    if (ctx.stages.enableFormula) {
        auto formulaStart = std::chrono::steady_clock::now();
        saveFormulaImages(image, equationBoxes, pageImage.pageIndex, result.elements, ctx);
        auto formulaEnd = std::chrono::steady_clock::now();
        result.stats.formulaTimeMs =
            std::chrono::duration<double, std::milli>(formulaEnd - formulaStart).count();
    }

    // Handle truly unsupported elements (non-formula) (CPU)
    auto unsupportedStart = std::chrono::steady_clock::now();
    auto skipElements = handleUnsupportedElements(unsupportedBoxes, pageImage.pageIndex);
    auto unsupportedEnd = std::chrono::steady_clock::now();
    result.stats.unsupportedTimeMs =
        std::chrono::duration<double, std::milli>(unsupportedEnd - unsupportedStart).count();
    result.elements.insert(result.elements.end(), skipElements.begin(), skipElements.end());

    // Step 3: Reading order sort (CPU)
    if (ctx.stages.enableReadingOrder && !result.elements.empty()) {
        auto orderStart = std::chrono::steady_clock::now();
        // Extract layout boxes from elements for sorting
        std::vector<LayoutBox> sortBoxes;
        for (const auto& elem : result.elements) {
            sortBoxes.push_back(elem.layoutBox);
        }

        auto sortedIndices = xycutPlusSort(sortBoxes, pageWidth, pageHeight);
        
        // Reorder elements
        std::vector<ContentElement> sortedElements;
        sortedElements.reserve(result.elements.size());
        for (int i = 0; i < static_cast<int>(sortedIndices.size()); i++) {
            int idx = sortedIndices[i];
            result.elements[idx].readingOrder = i;
            sortedElements.push_back(result.elements[idx]);
        }
        result.elements = std::move(sortedElements);
        auto orderEnd = std::chrono::steady_clock::now();
        result.stats.readingOrderTimeMs =
            std::chrono::duration<double, std::milli>(orderEnd - orderStart).count();
    }

    auto cpuEnd = std::chrono::steady_clock::now();
    cpuOnlyTotalMs +=
        std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    result.stats.cpuOnlyTimeMs = cpuOnlyTotalMs;

    auto endTime = std::chrono::steady_clock::now();
    result.totalTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    return result;
}

// ---------------------------------------------------------------------------
// OCR helper: submit one crop and block-wait for result
// ---------------------------------------------------------------------------
std::string DocPipeline::ocrOnCrop(const cv::Mat& crop, int64_t taskId) {
    if (crop.empty()) return "";

    if (!submitOcrTask(crop, taskId))
        return "";

    std::vector<ocr::PipelineOCRResult> ocrResults;
    bool success = false;
    if (!waitForOcrResult(taskId, ocrResults, success)) {
        LOG_WARN("OCR timeout for task {}", taskId);
        return "";
    }
    if (!success || ocrResults.empty()) {
        return "";
    }

    std::string combined;
    for (const auto& r : ocrResults) {
        if (!combined.empty()) combined += "\n";
        combined += r.text;
    }
    return combined;
}

bool DocPipeline::submitOcrTask(const cv::Mat& crop, int64_t taskId) {
    if (ocrSubmitHook_) {
        return ocrSubmitHook_(crop, taskId);
    }
    if (!ocrPipeline_) {
        return false;
    }
    return ocrPipeline_->pushTask(crop, taskId);
}

bool DocPipeline::fetchOcrResult(
    std::vector<ocr::PipelineOCRResult>& results,
    int64_t& resultId,
    bool& success)
{
    if (ocrFetchHook_) {
        return ocrFetchHook_(results, resultId, success);
    }
    if (!ocrPipeline_) {
        return false;
    }
    return ocrPipeline_->getResult(results, resultId, nullptr, &success);
}

bool DocPipeline::waitForOcrResult(
    int64_t taskId,
    std::vector<ocr::PipelineOCRResult>& results,
    bool& success,
    bool* bufferedHit)
{
    results.clear();
    success = false;
    if (bufferedHit != nullptr) {
        *bufferedHit = false;
    }

    {
        std::lock_guard<std::mutex> lock(ocrStateMutex_);
        // Keep OCR transient state bounded without clearing active request data.
        const int64_t pruneBeforeTaskId = taskId - 4096;
        if (pruneBeforeTaskId > 0) {
            for (auto it = timedOutOcrTaskIds_.begin(); it != timedOutOcrTaskIds_.end();) {
                if (*it < pruneBeforeTaskId) {
                    it = timedOutOcrTaskIds_.erase(it);
                } else {
                    ++it;
                }
            }
            for (auto it = bufferedOcrResults_.begin(); it != bufferedOcrResults_.end();) {
                if (it->first < pruneBeforeTaskId) {
                    it = bufferedOcrResults_.erase(it);
                } else {
                    ++it;
                }
            }
        }

        auto buffered = bufferedOcrResults_.find(taskId);
        if (buffered != bufferedOcrResults_.end()) {
            if (bufferedHit != nullptr) {
                *bufferedHit = true;
            }
            results = std::move(buffered->second.results);
            success = buffered->second.success;
            bufferedOcrResults_.erase(buffered);
            return true;
        }
    }

    auto deadline = std::chrono::steady_clock::now() + ocrWaitTimeout_;
    while (std::chrono::steady_clock::now() <= deadline) {
        std::vector<ocr::PipelineOCRResult> fetchedResults;
        int64_t resultId = -1;
        bool fetchedSuccess = false;
        if (!fetchOcrResult(fetchedResults, resultId, fetchedSuccess)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        if (resultId == taskId) {
            results = std::move(fetchedResults);
            success = fetchedSuccess;
            return true;
        }

        bool isStale = false;
        {
            std::lock_guard<std::mutex> lock(ocrStateMutex_);
            isStale = (timedOutOcrTaskIds_.count(resultId) > 0 || resultId < taskId);
            if (!isStale) {
                bufferedOcrResults_[resultId] = {std::move(fetchedResults), fetchedSuccess};
            }
        }

        if (isStale) {
            LOG_WARN("Discarding stale OCR result {} while waiting for {}", resultId, taskId);
            continue;
        }

        LOG_DEBUG("Buffered out-of-order OCR result {} while waiting for {}", resultId, taskId);
    }

    {
        std::lock_guard<std::mutex> lock(ocrStateMutex_);
        timedOutOcrTaskIds_.insert(taskId);
    }
    return false;
}

int64_t DocPipeline::allocateOcrTaskId() {
    return nextOcrTaskId_.fetch_add(1);
}

// ---------------------------------------------------------------------------
// OCR on text regions detected by layout
// ---------------------------------------------------------------------------
std::vector<ContentElement> DocPipeline::runOcrOnRegions(
    const cv::Mat& image,
    const std::vector<LayoutBox>& textBoxes,
    int pageIndex)
{
    std::vector<ContentElement> elements;
    const auto ocrWorkItems = buildOcrWorkItems(image, textBoxes, pageIndex);
    std::vector<std::string> texts(ocrWorkItems.size());
    std::vector<bool> hasText(ocrWorkItems.size(), false);
    std::vector<size_t> workIndexForOriginal(textBoxes.size(), 0);

    for (size_t workIndex = 0; workIndex < ocrWorkItems.size(); ++workIndex) {
        const auto& item = ocrWorkItems[workIndex];
        for (size_t originalIndex : item.originalIndices) {
            workIndexForOriginal[originalIndex] = workIndex;
        }
        if (item.skipped || item.crop.empty()) {
            continue;
        }
        texts[workIndex] = ocrOnCrop(item.crop, allocateOcrTaskId());
        hasText[workIndex] = !texts[workIndex].empty();
    }

    elements.reserve(textBoxes.size());
    for (size_t bi = 0; bi < textBoxes.size(); ++bi) {
        const auto& box = textBoxes[bi];
        const auto& item = ocrWorkItems[workIndexForOriginal[bi]];
        ContentElement elem;
        elem.type = (box.category == LayoutCategory::TITLE)
                        ? ContentElement::Type::TITLE
                        : ContentElement::Type::TEXT;
        elem.layoutBox = box;
        elem.confidence = box.confidence;
        elem.pageIndex = pageIndex;

        if (item.skipped) {
            elem.skipped = true;
            elements.push_back(std::move(elem));
            continue;
        }

        if (hasText[workIndexForOriginal[bi]]) {
            elem.text = texts[workIndexForOriginal[bi]];
        }
        elements.push_back(std::move(elem));
    }

    return elements;
}

// ---------------------------------------------------------------------------
// Table recognition with cell-level OCR
// ---------------------------------------------------------------------------
std::vector<ContentElement> DocPipeline::runTableRecognition(
    const cv::Mat& image,
    const std::vector<LayoutBox>& tableBoxes,
    int pageIndex)
{
    return runTableRecognition(image, tableBoxes, pageIndex, makeExecutionContext(nullptr));
}

std::vector<ContentElement> DocPipeline::runTableRecognition(
    const cv::Mat& image,
    const std::vector<LayoutBox>& tableBoxes,
    int pageIndex,
    const ExecutionContext& ctx)
{
    std::vector<ContentElement> elements;
    const auto groups = groupExactRoiDuplicates(tableBoxes, image.size());
    std::vector<size_t> workIndexForOriginal(tableBoxes.size(), 0);
    std::vector<ContentElement> uniqueElements;
    uniqueElements.reserve(groups.size());

    for (size_t groupIndex = 0; groupIndex < groups.size(); ++groupIndex) {
        const auto& group = groups[groupIndex];
        const auto& box = tableBoxes[group.canonicalIndex];
        for (size_t originalIndex : group.originalIndices) {
            workIndexForOriginal[originalIndex] = groupIndex;
        }

        if (group.roi.width <= 0 || group.roi.height <= 0) {
            uniqueElements.push_back(makeTableFallbackElement(box, pageIndex, "invalid_table_bbox"));
            continue;
        }

        ContentElement elem;
        elem.type = ContentElement::Type::TABLE;
        elem.layoutBox = box;
        elem.pageIndex = pageIndex;

        cv::Mat tableCrop = image(group.roi).clone();
        TableResult tableResult = recognizeTable(tableCrop);

        if (!tableResult.supported) {
            const std::string reason = (tableResult.type == TableType::WIRELESS)
                                           ? "wireless_table"
                                           : "table_model_unavailable";
            uniqueElements.push_back(makeTableFallbackElement(box, pageIndex, reason));
            continue;
        }

        if (tableResult.cells.empty()) {
            uniqueElements.push_back(makeTableFallbackElement(box, pageIndex, "no_cell_table"));
            continue;
        }

        if (ctx.stages.enableOcr && (ocrPipeline_ || (ocrSubmitHook_ && ocrFetchHook_))) {
            const int64_t ocrTaskId = allocateOcrTaskId();
            if (submitOcrTask(tableCrop, ocrTaskId)) {
                std::vector<ocr::PipelineOCRResult> ocrBoxes;
                bool ok = false;
                if (waitForOcrResult(ocrTaskId, ocrBoxes, ok) && ok && !ocrBoxes.empty()) {
                    matchTableOcrToCells(tableResult, ocrBoxes);
                }
            }
        }

        try {
            elem.html = generateTableHtml(tableResult.cells);
        } catch (const std::exception& ex) {
            LOG_WARN("Illegal table structure at page {}: {}", pageIndex, ex.what());
            uniqueElements.push_back(makeTableFallbackElement(box, pageIndex, "illegal_table_structure"));
            continue;
        }

        if (elem.html.empty()) {
            uniqueElements.push_back(makeTableFallbackElement(box, pageIndex, "empty_table_html"));
            continue;
        }

        elem.skipped = false;
        uniqueElements.push_back(std::move(elem));
    }

    elements.reserve(tableBoxes.size());
    for (size_t originalIndex = 0; originalIndex < tableBoxes.size(); ++originalIndex) {
        ContentElement elem = uniqueElements[workIndexForOriginal[originalIndex]];
        elem.layoutBox = tableBoxes[originalIndex];
        elem.pageIndex = pageIndex;
        elements.push_back(std::move(elem));
    }

    return elements;
}

TableResult DocPipeline::recognizeTable(const cv::Mat& tableCrop) {
    if (tableRecognizeHook_) {
        return tableRecognizeHook_(tableCrop);
    }
    if (!tableRecognizer_) {
        TableResult unavailable;
        unavailable.type = TableType::UNKNOWN;
        unavailable.supported = false;
        return unavailable;
    }
    return tableRecognizer_->recognize(tableCrop);
}

TableRecognizer::NpuStageResult DocPipeline::recognizeTableNpuStage(const cv::Mat& tableCrop) {
    TableRecognizer::NpuStageResult unavailable;
    unavailable.type = TableType::UNKNOWN;
    unavailable.supported = false;
    if (!tableRecognizer_) {
        return unavailable;
    }
    return tableRecognizer_->recognizeNpuStage(tableCrop);
}

TableResult DocPipeline::finalizeTableRecognizePostprocess(
    const cv::Mat& tableCrop,
    const TableRecognizer::NpuStageResult& npuStage)
{
    if (!tableRecognizer_) {
        TableResult unavailable;
        unavailable.type = TableType::UNKNOWN;
        unavailable.supported = false;
        return unavailable;
    }
    return tableRecognizer_->finalizeRecognizePostprocess(tableCrop, npuStage);
}

std::string DocPipeline::generateTableHtml(const std::vector<TableCell>& cells) {
    if (tableHtmlHook_) {
        return tableHtmlHook_(cells);
    }
    if (!tableRecognizer_) {
        return "";
    }
    return tableRecognizer_->generateHtml(cells);
}

std::string DocPipeline::tableFallbackMessage(const std::string& reason) {
    return "[Unsupported table: " + reason + "]";
}

ContentElement DocPipeline::makeTableFallbackElement(
    const LayoutBox& box,
    int pageIndex,
    const std::string& reason) const
{
    ContentElement elem;
    elem.type = ContentElement::Type::TABLE;
    elem.layoutBox = box;
    elem.pageIndex = pageIndex;
    elem.skipped = true;
    elem.text = tableFallbackMessage(reason);
    return elem;
}

std::vector<ContentElement> DocPipeline::handleUnsupportedElements(
    const std::vector<LayoutBox>& unsupportedBoxes,
    int pageIndex)
{
    std::vector<ContentElement> elements;

    for (const auto& box : unsupportedBoxes) {
        // Equations are handled by saveFormulaImages(), skip here
        if (box.category == LayoutCategory::EQUATION ||
            box.category == LayoutCategory::INTERLINE_EQUATION)
            continue;

        ContentElement elem;
        elem.layoutBox = box;
        elem.pageIndex = pageIndex;
        elem.skipped = true;
        elem.type = ContentElement::Type::UNKNOWN;
        elem.text = "[Unsupported layout category: " +
                    std::string(layoutCategoryToString(box.category)) + "]";

        LOG_DEBUG("Skipping unsupported element: {} at ({}, {})",
                  layoutCategoryToString(box.category), box.x0, box.y0);

        elements.push_back(elem);
    }

    return elements;
}

void DocPipeline::saveExtractedImages(
    const cv::Mat& image,
    const std::vector<LayoutBox>& figureBoxes,
    int pageIndex,
    std::vector<ContentElement>& elements)
{
    saveExtractedImages(image, figureBoxes, pageIndex, elements, makeExecutionContext(nullptr));
}

void DocPipeline::saveExtractedImages(
    const cv::Mat& image,
    const std::vector<LayoutBox>& figureBoxes,
    int pageIndex,
    std::vector<ContentElement>& elements,
    const ExecutionContext& ctx)
{
    for (size_t i = 0; i < figureBoxes.size(); i++) {
        const auto& box = figureBoxes[i];
        cv::Rect roi = box.toRect() & cv::Rect(0, 0, image.cols, image.rows);
        if (roi.width <= 0 || roi.height <= 0) continue;

        std::string filename = "images/page" + std::to_string(pageIndex) +
                               "_fig" + std::to_string(i) + ".png";

        if (ctx.runtime.saveImages) {
            cv::Mat figureCrop = image(roi);
            std::string filepath = ctx.runtime.outputDir + "/" + filename;
            std::filesystem::create_directories(
                std::filesystem::path(filepath).parent_path());
            cv::imwrite(filepath, figureCrop);
        }

        ContentElement elem;
        elem.type = ContentElement::Type::IMAGE;
        elem.layoutBox = box;
        elem.pageIndex = pageIndex;
        if (ctx.runtime.saveImages) {
            elem.imagePath = filename;
        }
        elements.push_back(elem);
    }
}

void DocPipeline::saveFormulaImages(
    const cv::Mat& image,
    const std::vector<LayoutBox>& equationBoxes,
    int pageIndex,
    std::vector<ContentElement>& elements)
{
    saveFormulaImages(image, equationBoxes, pageIndex, elements, makeExecutionContext(nullptr));
}

void DocPipeline::saveFormulaImages(
    const cv::Mat& image,
    const std::vector<LayoutBox>& equationBoxes,
    int pageIndex,
    std::vector<ContentElement>& elements,
    const ExecutionContext& ctx)
{
    // Python behavior: formula regions are saved as images, rendered as ![]()
    // No LaTeX recognition (onnx model not available on NPU).
    for (size_t i = 0; i < equationBoxes.size(); i++) {
        const auto& box = equationBoxes[i];
        cv::Rect roi = box.toRect() & cv::Rect(0, 0, image.cols, image.rows);
        if (roi.width <= 0 || roi.height <= 0) continue;

        std::string filename = "images/page" + std::to_string(pageIndex) +
                               "_eq" + std::to_string(i) + ".png";

        if (ctx.runtime.saveImages) {
            cv::Mat crop = image(roi);
            std::string filepath = ctx.runtime.outputDir + "/" + filename;
            std::filesystem::create_directories(
                std::filesystem::path(filepath).parent_path());
            cv::imwrite(filepath, crop);
        }

        ContentElement elem;
        elem.type = ContentElement::Type::EQUATION;
        elem.layoutBox = box;
        elem.pageIndex = pageIndex;
        if (ctx.runtime.saveImages) {
            elem.imagePath = filename;  // store path for image-based formula output
        }
        elements.push_back(elem);
    }
}

void DocPipeline::saveLayoutVisualization(
    const cv::Mat& image,
    const LayoutResult& layoutResult,
    int pageIndex)
{
    saveLayoutVisualization(image, layoutResult, pageIndex, makeExecutionContext(nullptr));
}

void DocPipeline::saveLayoutVisualization(
    const cv::Mat& image,
    const LayoutResult& layoutResult,
    int pageIndex,
    const ExecutionContext& ctx)
{
    if (image.empty()) {
        return;
    }

    cv::Mat vis = image.clone();
    for (const auto& box : layoutResult.boxes) {
        cv::Rect rect = box.toRect() & cv::Rect(0, 0, image.cols, image.rows);
        if (rect.width <= 0 || rect.height <= 0) {
            continue;
        }

        const cv::Scalar color(
            (37 * (box.clsId + 3)) % 255,
            (67 * (box.clsId + 5)) % 255,
            (97 * (box.clsId + 7)) % 255
        );

        cv::rectangle(vis, rect, color, 2);

        std::ostringstream label;
        label << (box.label.empty() ? layoutCategoryToString(box.category) : box.label)
              << " " << std::fixed << std::setprecision(2) << box.confidence;
        const std::string text = label.str();

        int baseline = 0;
        cv::Size textSize = cv::getTextSize(
            text, cv::FONT_HERSHEY_SIMPLEX, 0.45, 1, &baseline);
        cv::Rect textBg(
            rect.x,
            std::max(0, rect.y - textSize.height - 8),
            std::min(image.cols - rect.x, textSize.width + 8),
            textSize.height + 8);

        cv::rectangle(vis, textBg, color, cv::FILLED);
        cv::putText(
            vis,
            text,
            cv::Point(textBg.x + 4, textBg.y + textBg.height - 4),
            cv::FONT_HERSHEY_SIMPLEX,
            0.45,
            cv::Scalar(255, 255, 255),
            1,
            cv::LINE_AA);
    }

    std::ostringstream filename;
    filename << "layout/page_" << std::setw(4) << std::setfill('0') << pageIndex << "_layout.png";
    const std::string filepath = ctx.runtime.outputDir + "/" + filename.str();
    std::filesystem::create_directories(std::filesystem::path(filepath).parent_path());
    cv::imwrite(filepath, vis);
}

void DocPipeline::reportProgress(const std::string& stage, int current, int total) {
    if (progressCallback_) {
        progressCallback_(stage, current, total);
    }
}

} // namespace rapid_doc
