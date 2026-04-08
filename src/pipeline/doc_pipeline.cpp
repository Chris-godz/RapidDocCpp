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
#include <map>

namespace fs = std::filesystem;

namespace rapid_doc {

namespace {

void finalizeDocumentStats(DocumentResult& result) {
    const double pdfRenderTimeMs = result.stats.pdfRenderTimeMs;
    const double outputGenTimeMs = result.stats.outputGenTimeMs;
    const std::string pipelineMode = result.stats.pipelineMode;
    result.stats = accumulateDocumentStageStats(result.pages);
    result.stats.pdfRenderTimeMs = pdfRenderTimeMs;
    result.stats.outputGenTimeMs = outputGenTimeMs;
    result.stats.pipelineMode = pipelineMode.empty() ? result.stats.pipelineMode : pipelineMode;
}

using detail::OcrFetchResult;
using detail::OcrWorkItem;
using detail::TableNpuResult;
using detail::TableWorkItem;

template <typename T>
class StageQueue {
public:
    explicit StageQueue(size_t capacity)
        : capacity_(std::max<size_t>(1, capacity))
    {
    }

    bool push(T value, double* blockTimeMs = nullptr) {
        auto waitStart = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(mutex_);
        notFullCv_.wait(lock, [&]() {
            return closed_ || queue_.size() < capacity_;
        });
        auto waitEnd = std::chrono::steady_clock::now();
        if (blockTimeMs != nullptr) {
            *blockTimeMs +=
                std::chrono::duration<double, std::milli>(waitEnd - waitStart).count();
        }
        if (closed_) {
            return false;
        }

        queue_.push_back(std::move(value));
        if (queue_.size() > maxDepth_) {
            maxDepth_ = queue_.size();
        }
        lock.unlock();
        notEmptyCv_.notify_one();
        return true;
    }

    std::optional<T> pop(double* waitTimeMs = nullptr) {
        auto waitStart = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(mutex_);
        notEmptyCv_.wait(lock, [&]() {
            return closed_ || !queue_.empty();
        });
        auto waitEnd = std::chrono::steady_clock::now();
        if (waitTimeMs != nullptr) {
            *waitTimeMs +=
                std::chrono::duration<double, std::milli>(waitEnd - waitStart).count();
        }
        if (queue_.empty()) {
            return std::nullopt;
        }

        T value = std::move(queue_.front());
        queue_.pop_front();
        lock.unlock();
        notFullCv_.notify_one();
        return value;
    }

    void close() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            closed_ = true;
        }
        notEmptyCv_.notify_all();
        notFullCv_.notify_all();
    }

    size_t maxDepth() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return maxDepth_;
    }

private:
    const size_t capacity_;
    mutable std::mutex mutex_;
    std::condition_variable notEmptyCv_;
    std::condition_variable notFullCv_;
    std::deque<T> queue_;
    bool closed_ = false;
    size_t maxDepth_ = 0;
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
    if (overrides->pipelineMode.has_value()) ctx.runtime.pipelineMode = *overrides->pipelineMode;
    if (overrides->ocrOuterMode.has_value()) ctx.runtime.ocrOuterMode = *overrides->ocrOuterMode;
    if (overrides->ocrShadowWindow.has_value()) ctx.runtime.ocrShadowWindow = *overrides->ocrShadowWindow;
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

DocumentResult DocPipeline::processPdfSerial(
    PdfRenderer& renderer,
    const std::function<bool(PdfRenderer&, const PdfRenderer::PageVisitor&)>& renderFn,
    const ExecutionContext& ctx)
{
    (void)renderer;
    return processPdfStreaming(renderer, renderFn, ctx);
}

DocumentResult DocPipeline::processPdfPagePipelineMvp(
    PdfRenderer& renderer,
    const std::function<bool(PdfRenderer&, const PdfRenderer::PageVisitor&)>& renderFn,
    const ExecutionContext& ctx)
{
    using PageContextPtr = std::shared_ptr<PageContext>;

    DocumentResult result;
    result.stats.pipelineMode = pipelineModeToString(ctx.runtime.pipelineMode);

    StageQueue<PageContextPtr> qRendered(ctx.runtime.stageQueueRendered);
    StageQueue<PageContextPtr> qPlanned(ctx.runtime.stageQueuePlanned);
    StageQueue<PageContextPtr> qOcrTable(ctx.runtime.stageQueueOcrTable);
    StageQueue<PageContextPtr> qFinalize(ctx.runtime.stageQueueFinalize);

    std::mutex errorMutex;
    std::exception_ptr pipelineError;
    auto closeAllQueues = [&]() {
        qRendered.close();
        qPlanned.close();
        qOcrTable.close();
        qFinalize.close();
    };
    auto failPipeline = [&](std::exception_ptr eptr) {
        {
            std::lock_guard<std::mutex> lock(errorMutex);
            if (!pipelineError) {
                pipelineError = eptr;
            }
        }
        closeAllQueues();
    };

    std::mutex collectorMutex;
    std::map<int, PageResult> collectedPages;
    int scheduledPages = 0;
    double pdfRenderTimeMs = 0.0;

    std::thread producer([&]() {
        try {
            renderFn(renderer, [&](PageImage&& page) {
                auto pageCtx = std::make_shared<PageContext>();
                pageCtx->wallStartTime = std::chrono::steady_clock::now();
                pageCtx->pageImage = std::move(page);
                pageCtx->pageResult.pageIndex = pageCtx->pageImage.pageIndex;
                pageCtx->pageResult.pageWidth = pageCtx->pageImage.image.cols;
                pageCtx->pageResult.pageHeight = pageCtx->pageImage.image.rows;
                pageCtx->pageResult.stats.pipelineMode =
                    pipelineModeToString(ctx.runtime.pipelineMode);

                ++scheduledPages;
                pdfRenderTimeMs += pageCtx->pageImage.renderTimeMs;

                double pushBlockMs = 0.0;
                const bool pushed = qRendered.push(pageCtx, &pushBlockMs);
                pageCtx->pageResult.stats.renderQueuePushBlockTimeMs += pushBlockMs;
                pageCtx->pageResult.stats.queueBackpressureTimeMs += pushBlockMs;
                return pushed;
            });
        } catch (...) {
            failPipeline(std::current_exception());
        }
        qRendered.close();
    });

    std::thread layoutWorker([&]() {
        try {
            while (true) {
                double queueWaitMs = 0.0;
                auto pageCtx = qRendered.pop(&queueWaitMs);
                if (!pageCtx.has_value()) {
                    break;
                }
                (*pageCtx)->pageResult.stats.layoutQueueWaitTimeMs += queueWaitMs;
                runLayoutStage(*(*pageCtx), ctx);
                double pushBlockMs = 0.0;
                if (!qPlanned.push(*pageCtx, &pushBlockMs)) {
                    break;
                }
                (*pageCtx)->pageResult.stats.layoutQueuePushBlockTimeMs += pushBlockMs;
                (*pageCtx)->pageResult.stats.queueBackpressureTimeMs += pushBlockMs;
            }
        } catch (...) {
            failPipeline(std::current_exception());
        }
        qPlanned.close();
    });

    std::thread planWorker([&]() {
        try {
            while (true) {
                double queueWaitMs = 0.0;
                auto pageCtx = qPlanned.pop(&queueWaitMs);
                if (!pageCtx.has_value()) {
                    break;
                }
                (*pageCtx)->pageResult.stats.planQueueWaitTimeMs += queueWaitMs;
                runPlanStage(*(*pageCtx), ctx);
                double pushBlockMs = 0.0;
                if (!qOcrTable.push(*pageCtx, &pushBlockMs)) {
                    break;
                }
                (*pageCtx)->pageResult.stats.planQueuePushBlockTimeMs += pushBlockMs;
                (*pageCtx)->pageResult.stats.queueBackpressureTimeMs += pushBlockMs;
            }
        } catch (...) {
            failPipeline(std::current_exception());
        }
        qOcrTable.close();
    });

    std::thread ocrTableWorker([&]() {
        try {
            while (true) {
                double queueWaitMs = 0.0;
                auto pageCtx = qOcrTable.pop(&queueWaitMs);
                if (!pageCtx.has_value()) {
                    break;
                }
                (*pageCtx)->pageResult.stats.ocrTableQueueWaitTimeMs += queueWaitMs;
                runOcrTableStage(*(*pageCtx), ctx);
                double pushBlockMs = 0.0;
                if (!qFinalize.push(*pageCtx, &pushBlockMs)) {
                    break;
                }
                (*pageCtx)->pageResult.stats.ocrTableQueuePushBlockTimeMs += pushBlockMs;
                (*pageCtx)->pageResult.stats.queueBackpressureTimeMs += pushBlockMs;
            }
        } catch (...) {
            failPipeline(std::current_exception());
        }
        qFinalize.close();
    });

    std::thread finalizeWorker([&]() {
        try {
            while (true) {
                double queueWaitMs = 0.0;
                auto pageCtx = qFinalize.pop(&queueWaitMs);
                if (!pageCtx.has_value()) {
                    break;
                }
                (*pageCtx)->pageResult.stats.finalizeQueueWaitTimeMs += queueWaitMs;
                runFinalizeStage(*(*pageCtx), ctx);
                std::lock_guard<std::mutex> lock(collectorMutex);
                collectedPages[(*pageCtx)->pageResult.pageIndex] =
                    std::move((*pageCtx)->pageResult);
            }
        } catch (...) {
            failPipeline(std::current_exception());
        }
    });

    producer.join();
    layoutWorker.join();
    planWorker.join();
    ocrTableWorker.join();
    finalizeWorker.join();

    if (pipelineError) {
        std::rethrow_exception(pipelineError);
    }

    result.pages.reserve(collectedPages.size());
    for (auto& [pageIndex, page] : collectedPages) {
        (void)pageIndex;
        result.pages.push_back(std::move(page));
    }
    result.totalPages = scheduledPages;
    result.processedPages = static_cast<int>(result.pages.size());
    result.stats.pdfRenderTimeMs = pdfRenderTimeMs;
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

    resetOcrTransientStateForRun();

    reportProgress("PDF Render", 0, 1);
    if (ctx.stages.enablePdfRender) {
        PdfRenderConfig pdfCfg;
        pdfCfg.dpi = ctx.runtime.pdfDpi;
        pdfCfg.maxPages = ctx.runtime.maxPages;
        pdfCfg.startPageId = ctx.runtime.startPageId;
        pdfCfg.endPageId = ctx.runtime.endPageId;
        pdfCfg.maxConcurrentRenders = ctx.runtime.maxConcurrentPages;
        PdfRenderer renderer(pdfCfg);
        if (ctx.runtime.pipelineMode == PipelineMode::PagePipelineMvp) {
            result = processPdfPagePipelineMvp(
                renderer,
                [&pdfPath](PdfRenderer& activeRenderer, const PdfRenderer::PageVisitor& visitor) {
                    return activeRenderer.renderFileStreaming(pdfPath, visitor);
                },
                ctx);
        } else if (shouldUsePdfStreaming(ctx)) {
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
    result.stats.pipelineOverlapFactor =
        computePipelineOverlapFactor(result.stats, result.totalTimeMs);
    result.stats.pipelineMode = pipelineModeToString(ctx.runtime.pipelineMode);

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
    resetOcrTransientStateForRun();

    if (ctx.stages.enablePdfRender) {
        PdfRenderConfig pdfCfg;
        pdfCfg.dpi = ctx.runtime.pdfDpi;
        pdfCfg.maxPages = ctx.runtime.maxPages;
        pdfCfg.startPageId = ctx.runtime.startPageId;
        pdfCfg.endPageId = ctx.runtime.endPageId;
        pdfCfg.maxConcurrentRenders = ctx.runtime.maxConcurrentPages;
        PdfRenderer renderer(pdfCfg);
        if (ctx.runtime.pipelineMode == PipelineMode::PagePipelineMvp) {
            result = processPdfPagePipelineMvp(
                renderer,
                [data, size](PdfRenderer& activeRenderer, const PdfRenderer::PageVisitor& visitor) {
                    return activeRenderer.renderFromMemoryStreaming(data, size, visitor);
                },
                ctx);
        } else if (shouldUsePdfStreaming(ctx)) {
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
    result.stats.pipelineOverlapFactor =
        computePipelineOverlapFactor(result.stats, result.totalTimeMs);
    result.stats.pipelineMode = pipelineModeToString(ctx.runtime.pipelineMode);

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
    resetOcrTransientStateForRun();

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
    result.stats.pipelineOverlapFactor =
        computePipelineOverlapFactor(result.stats, result.totalTimeMs);
    result.stats.pipelineMode = pipelineModeToString(ctx.runtime.pipelineMode);
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
    PageContext pageCtx;
    pageCtx.wallStartTime = std::chrono::steady_clock::now();
    pageCtx.pageImage = pageImage;
    pageCtx.pageResult.pageIndex = pageImage.pageIndex;
    pageCtx.pageResult.pageWidth = pageImage.image.cols;
    pageCtx.pageResult.pageHeight = pageImage.image.rows;
    pageCtx.pageResult.stats.pipelineMode = pipelineModeToString(ctx.runtime.pipelineMode);

    runLayoutStage(pageCtx, ctx);
    runPlanStage(pageCtx, ctx);
    runOcrTableStage(pageCtx, ctx);
    runFinalizeStage(pageCtx, ctx);
    return pageCtx.pageResult;
}

void DocPipeline::runLayoutStage(PageContext& pageCtx, const ExecutionContext& ctx) {
    auto runNpuSerialized = [&](auto&& fn) -> double {
        auto lockWaitStart = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> npuLock(npuSerialMutex());
        auto lockAcquired = std::chrono::steady_clock::now();
        const double waitMs =
            std::chrono::duration<double, std::milli>(lockAcquired - lockWaitStart).count();
        pageCtx.npuLockWaitTotalMs += waitMs;
        pageCtx.layoutNpuSlotWaitTotalMs += waitMs;

        auto serialStart = lockAcquired;
        fn();
        auto serialEnd = std::chrono::steady_clock::now();
        const double serialMs =
            std::chrono::duration<double, std::milli>(serialEnd - serialStart).count();
        pageCtx.npuSerialTotalMs += serialMs;
        pageCtx.npuLockHoldTotalMs +=
            std::chrono::duration<double, std::milli>(serialEnd - lockAcquired).count();
        pageCtx.layoutNpuServiceTotalMs += serialMs;
        return serialMs;
    };

    if (layoutDetector_ && ctx.stages.enableLayout) {
        runNpuSerialized([&]() {
            auto layoutStart = std::chrono::steady_clock::now();
            pageCtx.pageResult.layoutResult = layoutDetector_->detect(pageCtx.pageImage.image);
            auto layoutEnd = std::chrono::steady_clock::now();
            pageCtx.pageResult.layoutResult.inferenceTimeMs =
                std::chrono::duration<double, std::milli>(layoutEnd - layoutStart).count();
            pageCtx.pageResult.stats.layoutTimeMs = pageCtx.pageResult.layoutResult.inferenceTimeMs;
        });

        LOG_DEBUG("Page {}: detected {} layout boxes",
                  pageCtx.pageImage.pageIndex, pageCtx.pageResult.layoutResult.boxes.size());
    }

    if (ctx.runtime.saveVisualization && ctx.stages.enableLayout) {
        auto vizStart = std::chrono::steady_clock::now();
        saveLayoutVisualization(
            pageCtx.pageImage.image,
            pageCtx.pageResult.layoutResult,
            pageCtx.pageImage.pageIndex,
            ctx);
        auto vizEnd = std::chrono::steady_clock::now();
        pageCtx.cpuOnlyTotalMs +=
            std::chrono::duration<double, std::milli>(vizEnd - vizStart).count();
    }
}

void DocPipeline::runPlanStage(PageContext& pageCtx, const ExecutionContext& ctx) {
    const cv::Mat& image = pageCtx.pageImage.image;
    auto stageStart = std::chrono::steady_clock::now();

    pageCtx.textBoxes = pageCtx.pageResult.layoutResult.getTextBoxes();
    pageCtx.tableBoxes = pageCtx.pageResult.layoutResult.getTableBoxes();
    pageCtx.figureBoxes = pageCtx.pageResult.layoutResult.getBoxesByCategory(LayoutCategory::FIGURE);
    pageCtx.equationBoxes = pageCtx.pageResult.layoutResult.getEquationBoxes();
    pageCtx.unsupportedBoxes = pageCtx.pageResult.layoutResult.getUnsupportedBoxes();
    pageCtx.textBoxesRawCount = pageCtx.textBoxes.size();
    pageCtx.tableBoxesRawCount = pageCtx.tableBoxes.size();
    pageCtx.ocrSubmitAreas.reserve(pageCtx.textBoxesRawCount + pageCtx.tableBoxesRawCount);

    if (ctx.stages.enableOcr) {
        pageCtx.ocrWorkItems =
            buildOcrWorkItems(image, pageCtx.textBoxes, pageCtx.pageImage.pageIndex);
        pageCtx.textBoxesAfterDedupCount = pageCtx.ocrWorkItems.size();
        if (pageCtx.textBoxesRawCount > pageCtx.textBoxesAfterDedupCount) {
            pageCtx.ocrDedupSkippedCount +=
                (pageCtx.textBoxesRawCount - pageCtx.textBoxesAfterDedupCount);
        }

        pageCtx.ocrWorkItemForOriginal.assign(pageCtx.textBoxes.size(), 0);
        pageCtx.ocrCanonicalWorkItem.assign(pageCtx.ocrWorkItems.size(), 0);
        std::unordered_map<ExactRoiKey, size_t, ExactRoiKeyHasher> ocrCanonicalByKey;
        const cv::Rect bounds(0, 0, image.cols, image.rows);
        for (size_t workIndex = 0; workIndex < pageCtx.ocrWorkItems.size(); ++workIndex) {
            pageCtx.ocrCanonicalWorkItem[workIndex] = workIndex;
            const cv::Rect roi = pageCtx.ocrWorkItems[workIndex].box.toRect() & bounds;
            if (roi.width > 0 && roi.height > 0) {
                const ExactRoiKey key{roi.x, roi.y, roi.width, roi.height};
                const auto [it, inserted] = ocrCanonicalByKey.emplace(key, workIndex);
                if (!inserted) {
                    pageCtx.ocrCanonicalWorkItem[workIndex] = it->second;
                    ++pageCtx.ocrDedupSkippedCount;
                }
            }
            for (size_t originalIndex : pageCtx.ocrWorkItems[workIndex].originalIndices) {
                pageCtx.ocrWorkItemForOriginal[originalIndex] = workIndex;
            }
        }
    }

    if (ctx.stages.enableWiredTable) {
        const auto groups = groupExactRoiDuplicates(pageCtx.tableBoxes, image.size());
        pageCtx.tableWorkItems.reserve(groups.size());
        pageCtx.tableWorkItemForOriginal.assign(pageCtx.tableBoxes.size(), 0);
        pageCtx.tableCanonicalWorkItem.assign(groups.size(), 0);
        std::unordered_map<ExactRoiKey, size_t, ExactRoiKeyHasher> tableCanonicalByKey;
        const cv::Rect bounds(0, 0, image.cols, image.rows);
        for (const auto& group : groups) {
            const auto& box = pageCtx.tableBoxes[group.canonicalIndex];
            TableWorkItem item;
            item.box = box;
            item.pageIndex = pageCtx.pageImage.pageIndex;
            item.originalIndices = group.originalIndices;
            if (group.roi.width <= 0 || group.roi.height <= 0) {
                item.invalidRoi = true;
            } else {
                item.crop = image(group.roi).clone();
            }

            const size_t workIndex = pageCtx.tableWorkItems.size();
            pageCtx.tableCanonicalWorkItem[workIndex] = workIndex;
            const cv::Rect roi = box.toRect() & bounds;
            if (roi.width > 0 && roi.height > 0) {
                const ExactRoiKey key{roi.x, roi.y, roi.width, roi.height};
                const auto [it, inserted] = tableCanonicalByKey.emplace(key, workIndex);
                if (!inserted) {
                    pageCtx.tableCanonicalWorkItem[workIndex] = it->second;
                    ++pageCtx.tableDedupSkippedCount;
                }
            }
            for (size_t originalIndex : group.originalIndices) {
                pageCtx.tableWorkItemForOriginal[originalIndex] = workIndex;
            }
            pageCtx.tableWorkItems.push_back(std::move(item));
        }

        pageCtx.tableBoxesAfterDedupCount = pageCtx.tableWorkItems.size();
        if (pageCtx.tableBoxesRawCount > pageCtx.tableBoxesAfterDedupCount) {
            pageCtx.tableDedupSkippedCount +=
                (pageCtx.tableBoxesRawCount - pageCtx.tableBoxesAfterDedupCount);
        }
    }

    auto stageEnd = std::chrono::steady_clock::now();
    pageCtx.cpuOnlyTotalMs +=
        std::chrono::duration<double, std::milli>(stageEnd - stageStart).count();
}

void DocPipeline::runOcrTableStage(PageContext& pageCtx, const ExecutionContext& ctx) {
    auto elapsedMs = [](const auto& start, const auto& end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    };
    struct SlotTiming {
        double waitMs = 0.0;
        double holdMs = 0.0;
    };
    auto addCpuPre = [&](double ms) {
        pageCtx.cpuOnlyTotalMs += ms;
        pageCtx.ocrTableCpuPreTotalMs += ms;
    };
    auto addCpuPost = [&](double ms) {
        pageCtx.cpuOnlyTotalMs += ms;
        pageCtx.ocrTableCpuPostTotalMs += ms;
    };
    auto addOcrCollectOrMerge = [&](double ms) {
        pageCtx.ocrCollectOrMergeTotalMs += ms;
    };
    auto runOcrTableLocked = [&](auto&& fn) -> SlotTiming {
        auto lockWaitStart = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> npuLock(npuSerialMutex());
        auto lockAcquired = std::chrono::steady_clock::now();
        SlotTiming timing;
        timing.waitMs = elapsedMs(lockWaitStart, lockAcquired);
        pageCtx.npuLockWaitTotalMs += timing.waitMs;

        auto holdStart = lockAcquired;
        fn();
        auto holdEnd = std::chrono::steady_clock::now();
        timing.holdMs = elapsedMs(holdStart, holdEnd);
        pageCtx.npuSerialTotalMs += timing.holdMs;
        pageCtx.npuLockHoldTotalMs += timing.holdMs;
        return timing;
    };

    const double pageArea = std::max(
        1.0,
        static_cast<double>(pageCtx.pageImage.image.cols) * pageCtx.pageImage.image.rows);
    auto recordOcrSubmitProfile = [&](LayoutCategory category, const cv::Mat& crop) {
        if (crop.empty()) {
            return;
        }

        const double cropArea = static_cast<double>(crop.cols) * crop.rows;
        pageCtx.ocrSubmitAreaSum += cropArea;
        pageCtx.ocrSubmitAreas.push_back(cropArea);

        const double areaRatio = cropArea / pageArea;
        if (areaRatio < 0.005) {
            ++pageCtx.ocrSubmitSmallCount;
        } else if (areaRatio < 0.02) {
            ++pageCtx.ocrSubmitMediumCount;
        } else {
            ++pageCtx.ocrSubmitLargeCount;
        }

        if (category == LayoutCategory::TABLE) {
            return;
        }

        switch (category) {
            case LayoutCategory::TITLE:
                ++pageCtx.ocrSubmitTitleCount;
                break;
            case LayoutCategory::CODE:
                ++pageCtx.ocrSubmitCodeCount;
                break;
            case LayoutCategory::LIST:
                ++pageCtx.ocrSubmitListCount;
                break;
            default:
                ++pageCtx.ocrSubmitTextCount;
                break;
        }
    };

    if (ctx.stages.enableOcr) {
        pageCtx.ocrFetchResults.assign(pageCtx.ocrWorkItems.size(), {});
        double ocrWindowMs = 0.0;
        std::deque<size_t> inflightOcrTasks;

        auto collectTextTask = [&](size_t workIndex) {
            auto& fetch = pageCtx.ocrFetchResults[workIndex];
            if (!fetch.submitted) {
                return;
            }

            bool bufferedHit = false;
            size_t bufferedOutOfOrderCount = 0;
            const auto collectStart = std::chrono::steady_clock::now();
            fetch.fetched = waitForOcrResult(
                fetch.taskId,
                fetch.results,
                fetch.success,
                &bufferedHit,
                &bufferedOutOfOrderCount);
            const auto collectEnd = std::chrono::steady_clock::now();

            fetch.bufferedHit = bufferedHit;
            fetch.bufferedOutOfOrderCount += bufferedOutOfOrderCount;
            fetch.collectWaitMs += elapsedMs(collectStart, collectEnd);
            pageCtx.ocrCollectWaitTotalMs += fetch.collectWaitMs;
            pageCtx.ocrBufferedOutOfOrderCount += bufferedOutOfOrderCount;

            if (fetch.submitTime.has_value()) {
                fetch.submoduleWindowMs = elapsedMs(*fetch.submitTime, collectEnd);
                pageCtx.ocrSubmoduleWindowTotalMs += fetch.submoduleWindowMs;
                pageCtx.ocrTableNpuServiceTotalMs += fetch.submoduleWindowMs;
                ocrWindowMs += fetch.submoduleWindowMs;
            }
            if (bufferedHit) {
                ++pageCtx.ocrBufferedResultHitCount;
            }
            if (!fetch.fetched) {
                ++pageCtx.ocrTimeoutCount;
                LOG_WARN("OCR timeout for task {}", fetch.taskId);
            }
        };

        for (size_t i = 0; i < pageCtx.ocrWorkItems.size(); ++i) {
            const auto& item = pageCtx.ocrWorkItems[i];
            const size_t canonicalWorkIndex = pageCtx.ocrCanonicalWorkItem[i];
            if (canonicalWorkIndex != i) {
                continue;
            }
            if (item.skipped || item.crop.empty()) {
                continue;
            }

            const int64_t taskId = allocateOcrTaskId();
            auto& fetch = pageCtx.ocrFetchResults[i];
            fetch = {};
            fetch.taskId = taskId;
            const bool shadowCollect =
                ctx.runtime.ocrOuterMode == OcrOuterMode::ShadowWindowedCollect;
            if (shadowCollect) {
                SlotTiming slotTiming = runOcrTableLocked([&]() {
                    fetch.submitted = submitOcrTask(item.crop, taskId);
                });
                fetch.slotWaitMs += slotTiming.waitMs;
                fetch.outerSlotHoldMs += slotTiming.holdMs;
                pageCtx.ocrSlotWaitTotalMs += slotTiming.waitMs;
                pageCtx.ocrOuterSlotHoldTotalMs += slotTiming.holdMs;
                pageCtx.ocrTableNpuSlotWaitTotalMs += slotTiming.waitMs;
                if (fetch.submitted) {
                    fetch.submitTime = std::chrono::steady_clock::now();
                    ++pageCtx.ocrSubmitCount;
                    recordOcrSubmitProfile(item.box.category, item.crop);
                    inflightOcrTasks.push_back(i);
                    pageCtx.ocrInflightPeak =
                        std::max(pageCtx.ocrInflightPeak, inflightOcrTasks.size());
                }
            } else {
                bool bufferedHit = false;
                size_t bufferedOutOfOrderCount = 0;
                auto serviceStart = std::chrono::steady_clock::now();
                SlotTiming slotTiming = runOcrTableLocked([&]() {
                    serviceStart = std::chrono::steady_clock::now();
                    fetch.submitted = submitOcrTask(item.crop, taskId);
                    if (fetch.submitted) {
                        fetch.submitTime = serviceStart;
                        fetch.fetched = waitForOcrResult(
                            taskId,
                            fetch.results,
                            fetch.success,
                            &bufferedHit,
                            &bufferedOutOfOrderCount);
                    }
                });
                const auto serviceEnd = std::chrono::steady_clock::now();

                fetch.slotWaitMs += slotTiming.waitMs;
                fetch.outerSlotHoldMs += slotTiming.holdMs;
                fetch.bufferedHit = bufferedHit;
                fetch.bufferedOutOfOrderCount += bufferedOutOfOrderCount;
                pageCtx.ocrSlotWaitTotalMs += slotTiming.waitMs;
                pageCtx.ocrOuterSlotHoldTotalMs += slotTiming.holdMs;
                pageCtx.ocrTableNpuSlotWaitTotalMs += slotTiming.waitMs;
                pageCtx.ocrBufferedOutOfOrderCount += bufferedOutOfOrderCount;

                if (fetch.submitted) {
                    ++pageCtx.ocrSubmitCount;
                    recordOcrSubmitProfile(item.box.category, item.crop);
                    fetch.submoduleWindowMs = elapsedMs(serviceStart, serviceEnd);
                    pageCtx.ocrSubmoduleWindowTotalMs += fetch.submoduleWindowMs;
                    pageCtx.ocrTableNpuServiceTotalMs += fetch.submoduleWindowMs;
                    ocrWindowMs += fetch.submoduleWindowMs;
                    pageCtx.ocrInflightPeak = std::max<size_t>(pageCtx.ocrInflightPeak, 1);
                }
                if (bufferedHit) {
                    ++pageCtx.ocrBufferedResultHitCount;
                }
                if (!fetch.submitted) {
                    continue;
                }
                if (!fetch.fetched) {
                    ++pageCtx.ocrTimeoutCount;
                    LOG_WARN("OCR timeout for task {}", taskId);
                }
            }
            if (!fetch.submitted) {
                continue;
            }

            if (shadowCollect && inflightOcrTasks.size() >= ctx.runtime.ocrShadowWindow) {
                const size_t oldestWorkIndex = inflightOcrTasks.front();
                inflightOcrTasks.pop_front();
                collectTextTask(oldestWorkIndex);
            }
        }
        while (!inflightOcrTasks.empty()) {
            const size_t workIndex = inflightOcrTasks.front();
            inflightOcrTasks.pop_front();
            collectTextTask(workIndex);
        }
        pageCtx.pageResult.stats.ocrTimeMs = ocrWindowMs;

        auto assembleStart = std::chrono::steady_clock::now();
        pageCtx.textElements.clear();
        pageCtx.textElements.reserve(pageCtx.textBoxes.size());
        for (size_t originalIndex = 0; originalIndex < pageCtx.textBoxes.size(); ++originalIndex) {
            const auto& box = pageCtx.textBoxes[originalIndex];
            const size_t workIndex = pageCtx.ocrWorkItemForOriginal[originalIndex];
            const size_t canonicalIndex = pageCtx.ocrCanonicalWorkItem[workIndex];
            const auto& item = pageCtx.ocrWorkItems[workIndex];
            ContentElement elem;
            elem.type = (box.category == LayoutCategory::TITLE)
                            ? ContentElement::Type::TITLE
                            : ContentElement::Type::TEXT;
            elem.layoutBox = box;
            elem.confidence = box.confidence;
            elem.pageIndex = pageCtx.pageImage.pageIndex;

            if (item.skipped) {
                elem.skipped = true;
                pageCtx.textElements.push_back(std::move(elem));
                continue;
            }

            const auto& fetch = pageCtx.ocrFetchResults[canonicalIndex];
            if (fetch.fetched && fetch.success && !fetch.results.empty()) {
                elem.text = combineOcrTextLines(fetch.results);
            }
            pageCtx.textElements.push_back(std::move(elem));
        }
        auto assembleEnd = std::chrono::steady_clock::now();
        const double assembleMs = elapsedMs(assembleStart, assembleEnd);
        addCpuPost(assembleMs);
        addOcrCollectOrMerge(assembleMs);
    }

    if (ctx.stages.enableWiredTable) {
        auto markTableFallback = [](TableNpuResult& npuResult, const std::string& reason) {
            npuResult.hasFallback = true;
            npuResult.fallbackReason = reason;
        };

        double tableNpuStageMs = 0.0;
        pageCtx.tableNpuResults.clear();
        pageCtx.tableNpuResults.reserve(pageCtx.tableWorkItems.size());
        for (size_t i = 0; i < pageCtx.tableWorkItems.size(); ++i) {
            const auto& item = pageCtx.tableWorkItems[i];
            TableNpuResult npuResult;
            npuResult.box = item.box;
            npuResult.pageIndex = item.pageIndex;
            const size_t canonicalWorkIndex = pageCtx.tableCanonicalWorkItem[i];
            if (canonicalWorkIndex != i) {
                npuResult = pageCtx.tableNpuResults[canonicalWorkIndex];
                npuResult.box = item.box;
                npuResult.pageIndex = item.pageIndex;
                pageCtx.tableNpuResults.push_back(std::move(npuResult));
                continue;
            }

            if (item.invalidRoi || item.crop.empty()) {
                markTableFallback(npuResult, "invalid_table_bbox");
                pageCtx.tableNpuResults.push_back(std::move(npuResult));
                continue;
            }
            ++pageCtx.tableNpuSubmitCount;

            if (tableRecognizeHook_) {
                double serviceMs = 0.0;
                SlotTiming slotTiming = runOcrTableLocked([&]() {
                    const auto serviceStart = std::chrono::steady_clock::now();
                    npuResult.tableResult = recognizeTable(item.crop);
                    const auto serviceEnd = std::chrono::steady_clock::now();
                    serviceMs = elapsedMs(serviceStart, serviceEnd);
                });
                tableNpuStageMs += serviceMs;
                pageCtx.ocrTableNpuSlotWaitTotalMs += slotTiming.waitMs;
                pageCtx.tableNpuSlotWaitTotalMs += slotTiming.waitMs;
                pageCtx.ocrTableNpuServiceTotalMs += serviceMs;
                pageCtx.tableNpuServiceTotalMs += serviceMs;
                npuResult.hasTableResult = true;
            } else {
                auto cpuPreStart = std::chrono::steady_clock::now();
                if (tableRecognizer_) {
                    npuResult.npuStage = tableRecognizer_->prepareNpuInput(item.crop);
                } else {
                    npuResult.npuStage.type = TableType::UNKNOWN;
                    npuResult.npuStage.supported = false;
                }
                auto cpuPreEnd = std::chrono::steady_clock::now();
                addCpuPre(elapsedMs(cpuPreStart, cpuPreEnd));

                if (!npuResult.npuStage.supported) {
                    markTableFallback(
                        npuResult,
                        (npuResult.npuStage.type == TableType::WIRELESS)
                            ? "wireless_table"
                            : "table_model_unavailable");
                    pageCtx.tableNpuResults.push_back(std::move(npuResult));
                    continue;
                }

                double serviceMs = 0.0;
                SlotTiming slotTiming = runOcrTableLocked([&]() {
                    if (tableRecognizer_) {
                        tableRecognizer_->runPreparedNpu(npuResult.npuStage);
                        serviceMs = npuResult.npuStage.dxRunMs;
                    }
                });
                tableNpuStageMs += serviceMs;
                pageCtx.ocrTableNpuSlotWaitTotalMs += slotTiming.waitMs;
                pageCtx.tableNpuSlotWaitTotalMs += slotTiming.waitMs;
                pageCtx.ocrTableNpuServiceTotalMs += serviceMs;
                pageCtx.tableNpuServiceTotalMs += serviceMs;

            }

            if (npuResult.hasTableResult && !npuResult.tableResult.supported) {
                markTableFallback(
                    npuResult,
                    (npuResult.tableResult.type == TableType::WIRELESS)
                        ? "wireless_table"
                        : "table_model_unavailable");
                pageCtx.tableNpuResults.push_back(std::move(npuResult));
                continue;
            }
            if (npuResult.hasTableResult && npuResult.tableResult.cells.empty()) {
                markTableFallback(npuResult, "no_cell_table");
            }

            pageCtx.tableNpuResults.push_back(std::move(npuResult));
        }

        double tableOcrNpuMs = 0.0;
        const bool tableOcrEnabled =
            ctx.stages.enableOcr && (ocrPipeline_ || (ocrSubmitHook_ && ocrFetchHook_));
        if (tableOcrEnabled) {
            for (size_t i = 0; i < pageCtx.tableNpuResults.size(); ++i) {
                auto& npuResult = pageCtx.tableNpuResults[i];
                if (npuResult.hasFallback) {
                    continue;
                }
                const size_t canonicalWorkIndex = pageCtx.tableCanonicalWorkItem[i];
                if (canonicalWorkIndex != i) {
                    npuResult.ocrBoxes = pageCtx.tableNpuResults[canonicalWorkIndex].ocrBoxes;
                    continue;
                }

                const int64_t ocrTaskId = allocateOcrTaskId();
                bool submitted = false;
                bool ok = false;
                bool fetched = false;
                bool bufferedHit = false;
                size_t bufferedOutOfOrderCount = 0;
                double serviceMs = 0.0;
                SlotTiming slotTiming = runOcrTableLocked([&]() {
                    const auto serviceStart = std::chrono::steady_clock::now();
                    submitted = submitOcrTask(pageCtx.tableWorkItems[i].crop, ocrTaskId);
                    if (submitted) {
                        fetched = waitForOcrResult(
                            ocrTaskId,
                            npuResult.ocrBoxes,
                            ok,
                            &bufferedHit,
                            &bufferedOutOfOrderCount);
                    }
                    const auto serviceEnd = std::chrono::steady_clock::now();
                    serviceMs = elapsedMs(serviceStart, serviceEnd);
                });
                tableOcrNpuMs += serviceMs;
                pageCtx.ocrTableNpuSlotWaitTotalMs += slotTiming.waitMs;
                pageCtx.tableOcrSlotWaitTotalMs += slotTiming.waitMs;
                pageCtx.ocrTableNpuServiceTotalMs += serviceMs;
                pageCtx.tableOcrServiceTotalMs += serviceMs;
                pageCtx.ocrBufferedOutOfOrderCount += bufferedOutOfOrderCount;
                if (!submitted) {
                    continue;
                }

                ++pageCtx.ocrSubmitCount;
                recordOcrSubmitProfile(LayoutCategory::TABLE, pageCtx.tableWorkItems[i].crop);
                if (bufferedHit) {
                    ++pageCtx.ocrBufferedResultHitCount;
                }
                if (!(fetched && ok)) {
                    if (!fetched) {
                        ++pageCtx.ocrTimeoutCount;
                    }
                    npuResult.ocrBoxes.clear();
                }
            }
        }
        pageCtx.pageResult.stats.tableTimeMs = tableNpuStageMs + tableOcrNpuMs;
    }
}

void DocPipeline::runFinalizeStage(PageContext& pageCtx, const ExecutionContext& ctx) {
    auto finalizeStart = std::chrono::steady_clock::now();
    runTableFinalizeCpuSubstage(pageCtx, ctx);
    runDocumentFinalizeSubstage(pageCtx, ctx);
    auto finalizeEnd = std::chrono::steady_clock::now();
    const double finalizeCpuMs =
        std::chrono::duration<double, std::milli>(finalizeEnd - finalizeStart).count();
    pageCtx.finalizeCpuTotalMs += finalizeCpuMs;
    pageCtx.cpuOnlyTotalMs += finalizeCpuMs;

    auto& result = pageCtx.pageResult;
    result.stats.npuLockWaitTimeMs = pageCtx.npuLockWaitTotalMs;
    result.stats.npuLockHoldTimeMs = pageCtx.npuLockHoldTotalMs;
    result.stats.npuSerialTimeMs = pageCtx.npuSerialTotalMs;
    result.stats.cpuOnlyTimeMs = pageCtx.cpuOnlyTotalMs;
    result.stats.npuServiceTimeMs = pageCtx.ocrTableNpuServiceTotalMs;
    result.stats.npuSlotWaitTimeMs = pageCtx.ocrTableNpuSlotWaitTotalMs;
    result.stats.layoutNpuServiceTimeMs = pageCtx.layoutNpuServiceTotalMs;
    result.stats.layoutNpuSlotWaitTimeMs = pageCtx.layoutNpuSlotWaitTotalMs;
    result.stats.ocrOuterSlotHoldTimeMs = pageCtx.ocrOuterSlotHoldTotalMs;
    result.stats.ocrSubmoduleWindowTimeMs = pageCtx.ocrSubmoduleWindowTotalMs;
    result.stats.ocrSlotWaitTimeMs = pageCtx.ocrSlotWaitTotalMs;
    result.stats.ocrCollectWaitTimeMs = pageCtx.ocrCollectWaitTotalMs;
    result.stats.ocrInflightPeak = static_cast<double>(pageCtx.ocrInflightPeak);
    result.stats.ocrBufferedOutOfOrderCount =
        static_cast<double>(pageCtx.ocrBufferedOutOfOrderCount);
    result.stats.tableNpuServiceTimeMs = pageCtx.tableNpuServiceTotalMs;
    result.stats.tableNpuSlotWaitTimeMs = pageCtx.tableNpuSlotWaitTotalMs;
    result.stats.tableOcrServiceTimeMs = pageCtx.tableOcrServiceTotalMs;
    result.stats.tableOcrSlotWaitTimeMs = pageCtx.tableOcrSlotWaitTotalMs;
    result.stats.cpuPreTimeMs = pageCtx.ocrTableCpuPreTotalMs;
    result.stats.cpuPostTimeMs = pageCtx.ocrTableCpuPostTotalMs;
    result.stats.finalizeCpuTimeMs = pageCtx.finalizeCpuTotalMs;
    result.stats.tableFinalizeTimeMs = pageCtx.tableFinalizeTotalMs;
    result.stats.ocrCollectOrMergeTimeMs = pageCtx.ocrCollectOrMergeTotalMs;
    result.stats.textBoxesRawCount = static_cast<double>(pageCtx.textBoxesRawCount);
    result.stats.textBoxesAfterDedupCount = static_cast<double>(pageCtx.textBoxesAfterDedupCount);
    result.stats.tableBoxesRawCount = static_cast<double>(pageCtx.tableBoxesRawCount);
    result.stats.tableBoxesAfterDedupCount = static_cast<double>(pageCtx.tableBoxesAfterDedupCount);
    result.stats.ocrSubmitCount = static_cast<double>(pageCtx.ocrSubmitCount);
    result.stats.ocrSubmitAreaSum = pageCtx.ocrSubmitAreaSum;
    if (!pageCtx.ocrSubmitAreas.empty()) {
        const PercentileSummary submitAreaSummary =
            summarizeSamples(std::move(pageCtx.ocrSubmitAreas));
        result.stats.ocrSubmitAreaMean = submitAreaSummary.meanMs;
        result.stats.ocrSubmitAreaP50 = submitAreaSummary.p50Ms;
        result.stats.ocrSubmitAreaP95 = submitAreaSummary.p95Ms;
    }
    result.stats.ocrSubmitSmallCount = static_cast<double>(pageCtx.ocrSubmitSmallCount);
    result.stats.ocrSubmitMediumCount = static_cast<double>(pageCtx.ocrSubmitMediumCount);
    result.stats.ocrSubmitLargeCount = static_cast<double>(pageCtx.ocrSubmitLargeCount);
    result.stats.ocrSubmitTextCount = static_cast<double>(pageCtx.ocrSubmitTextCount);
    result.stats.ocrSubmitTitleCount = static_cast<double>(pageCtx.ocrSubmitTitleCount);
    result.stats.ocrSubmitCodeCount = static_cast<double>(pageCtx.ocrSubmitCodeCount);
    result.stats.ocrSubmitListCount = static_cast<double>(pageCtx.ocrSubmitListCount);
    result.stats.ocrDedupSkippedCount = static_cast<double>(pageCtx.ocrDedupSkippedCount);
    result.stats.tableNpuSubmitCount = static_cast<double>(pageCtx.tableNpuSubmitCount);
    result.stats.tableDedupSkippedCount = static_cast<double>(pageCtx.tableDedupSkippedCount);
    result.stats.ocrTimeoutCount = static_cast<double>(pageCtx.ocrTimeoutCount);
    result.stats.ocrBufferedResultHitCount =
        static_cast<double>(pageCtx.ocrBufferedResultHitCount);
    result.stats.queueBackpressureTimeMs =
        result.stats.renderQueuePushBlockTimeMs +
        result.stats.layoutQueuePushBlockTimeMs +
        result.stats.planQueuePushBlockTimeMs +
        result.stats.ocrTableQueuePushBlockTimeMs;
    result.stats.pipelineMode = pipelineModeToString(ctx.runtime.pipelineMode);

    result.totalTimeMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - pageCtx.wallStartTime).count();
    result.stats.pipelineOverlapFactor =
        computePipelineOverlapFactor(result.stats, result.totalTimeMs);
}

void DocPipeline::runTableFinalizeCpuSubstage(PageContext& pageCtx, const ExecutionContext& ctx) {
    (void)ctx;
    auto elapsedMs = [](const auto& start, const auto& end) {
        return std::chrono::duration<double, std::milli>(end - start).count();
    };
    auto addTableFinalize = [&](double ms) {
        pageCtx.tableFinalizeTotalMs += ms;
    };
    auto addOcrCollectOrMerge = [&](double ms) {
        pageCtx.ocrCollectOrMergeTotalMs += ms;
    };

    pageCtx.pageResult.tableResults.clear();
    pageCtx.tableElements.clear();
    if (pageCtx.tableNpuResults.empty()) {
        return;
    }

    auto markTableFallback = [](TableNpuResult& npuResult, const std::string& reason) {
        npuResult.hasFallback = true;
        npuResult.fallbackReason = reason;
    };

    std::vector<ContentElement> uniqueTableElements;
    uniqueTableElements.reserve(pageCtx.tableNpuResults.size());
    pageCtx.pageResult.tableResults.reserve(pageCtx.tableNpuResults.size());

    for (size_t i = 0; i < pageCtx.tableNpuResults.size(); ++i) {
        auto& npuResult = pageCtx.tableNpuResults[i];
        const auto& item = pageCtx.tableWorkItems[i];

        if (!npuResult.hasFallback && !npuResult.hasTableResult) {
            auto finalizeStart = std::chrono::steady_clock::now();
            if (tableRecognizer_) {
                tableRecognizer_->decodePreparedMask(npuResult.npuStage);
            }
            npuResult.tableResult =
                finalizeTableRecognizePostprocess(item.crop, npuResult.npuStage);
            npuResult.hasTableResult = true;
            auto finalizeEnd = std::chrono::steady_clock::now();
            addTableFinalize(elapsedMs(finalizeStart, finalizeEnd));
        }

        if (!npuResult.hasFallback && !npuResult.tableResult.supported) {
            markTableFallback(
                npuResult,
                (npuResult.tableResult.type == TableType::WIRELESS)
                    ? "wireless_table"
                    : "table_model_unavailable");
        }
        if (!npuResult.hasFallback && npuResult.tableResult.cells.empty()) {
            markTableFallback(npuResult, "no_cell_table");
        }

        if (npuResult.hasFallback) {
            auto finalizeStart = std::chrono::steady_clock::now();
            uniqueTableElements.push_back(makeTableFallbackElement(
                npuResult.box, npuResult.pageIndex, npuResult.fallbackReason));
            auto finalizeEnd = std::chrono::steady_clock::now();
            addTableFinalize(elapsedMs(finalizeStart, finalizeEnd));
            continue;
        }

        auto mergeStart = std::chrono::steady_clock::now();
        matchTableOcrToCells(npuResult.tableResult, npuResult.ocrBoxes);
        auto mergeEnd = std::chrono::steady_clock::now();
        addOcrCollectOrMerge(elapsedMs(mergeStart, mergeEnd));
        pageCtx.pageResult.tableResults.push_back(npuResult.tableResult);

        ContentElement elem;
        elem.type = ContentElement::Type::TABLE;
        elem.layoutBox = npuResult.box;
        elem.pageIndex = npuResult.pageIndex;

        auto finalizeStart = std::chrono::steady_clock::now();
        try {
            elem.html = generateTableHtml(npuResult.tableResult.cells);
        } catch (const std::exception& ex) {
            LOG_WARN("Illegal table structure at page {}: {}",
                     npuResult.pageIndex, ex.what());
            uniqueTableElements.push_back(makeTableFallbackElement(
                npuResult.box, npuResult.pageIndex, "illegal_table_structure"));
            auto finalizeEnd = std::chrono::steady_clock::now();
            addTableFinalize(elapsedMs(finalizeStart, finalizeEnd));
            continue;
        }

        if (elem.html.empty()) {
            uniqueTableElements.push_back(makeTableFallbackElement(
                npuResult.box, npuResult.pageIndex, "empty_table_html"));
            auto finalizeEnd = std::chrono::steady_clock::now();
            addTableFinalize(elapsedMs(finalizeStart, finalizeEnd));
            continue;
        }

        elem.skipped = false;
        uniqueTableElements.push_back(std::move(elem));
        auto finalizeEnd = std::chrono::steady_clock::now();
        addTableFinalize(elapsedMs(finalizeStart, finalizeEnd));
    }

    auto fanoutStart = std::chrono::steady_clock::now();
    pageCtx.tableElements.reserve(pageCtx.tableBoxes.size());
    for (size_t originalIndex = 0; originalIndex < pageCtx.tableBoxes.size(); ++originalIndex) {
        ContentElement elem =
            uniqueTableElements[pageCtx.tableWorkItemForOriginal[originalIndex]];
        elem.layoutBox = pageCtx.tableBoxes[originalIndex];
        elem.pageIndex = pageCtx.pageImage.pageIndex;
        pageCtx.tableElements.push_back(std::move(elem));
    }
    auto fanoutEnd = std::chrono::steady_clock::now();
    addTableFinalize(elapsedMs(fanoutStart, fanoutEnd));
}

void DocPipeline::runDocumentFinalizeSubstage(PageContext& pageCtx, const ExecutionContext& ctx) {
    auto& result = pageCtx.pageResult;
    const cv::Mat& image = pageCtx.pageImage.image;
    result.elements.clear();
    result.elements.reserve(pageCtx.textElements.size() +
                            pageCtx.tableElements.size() +
                            pageCtx.figureBoxes.size() +
                            pageCtx.equationBoxes.size() +
                            pageCtx.unsupportedBoxes.size());
    result.elements.insert(
        result.elements.end(), pageCtx.textElements.begin(), pageCtx.textElements.end());
    result.elements.insert(
        result.elements.end(), pageCtx.tableElements.begin(), pageCtx.tableElements.end());

    auto figureStart = std::chrono::steady_clock::now();
    saveExtractedImages(image, pageCtx.figureBoxes, pageCtx.pageImage.pageIndex, result.elements, ctx);
    auto figureEnd = std::chrono::steady_clock::now();
    result.stats.figureTimeMs =
        std::chrono::duration<double, std::milli>(figureEnd - figureStart).count();

    if (ctx.stages.enableFormula) {
        auto formulaStart = std::chrono::steady_clock::now();
        saveFormulaImages(
            image, pageCtx.equationBoxes, pageCtx.pageImage.pageIndex, result.elements, ctx);
        auto formulaEnd = std::chrono::steady_clock::now();
        result.stats.formulaTimeMs =
            std::chrono::duration<double, std::milli>(formulaEnd - formulaStart).count();
    }

    auto unsupportedStart = std::chrono::steady_clock::now();
    auto skipElements =
        handleUnsupportedElements(pageCtx.unsupportedBoxes, pageCtx.pageImage.pageIndex);
    auto unsupportedEnd = std::chrono::steady_clock::now();
    result.stats.unsupportedTimeMs =
        std::chrono::duration<double, std::milli>(unsupportedEnd - unsupportedStart).count();
    result.elements.insert(result.elements.end(), skipElements.begin(), skipElements.end());

    if (ctx.stages.enableReadingOrder && !result.elements.empty()) {
        auto orderStart = std::chrono::steady_clock::now();
        std::vector<LayoutBox> sortBoxes;
        sortBoxes.reserve(result.elements.size());
        for (const auto& elem : result.elements) {
            sortBoxes.push_back(elem.layoutBox);
        }

        const auto sortedIndices = xycutPlusSort(sortBoxes, image.cols, image.rows);
        std::vector<ContentElement> sortedElements;
        sortedElements.reserve(result.elements.size());
        for (int i = 0; i < static_cast<int>(sortedIndices.size()); ++i) {
            const int idx = sortedIndices[i];
            result.elements[idx].readingOrder = i;
            sortedElements.push_back(result.elements[idx]);
        }
        result.elements = std::move(sortedElements);
        auto orderEnd = std::chrono::steady_clock::now();
        result.stats.readingOrderTimeMs =
            std::chrono::duration<double, std::milli>(orderEnd - orderStart).count();
    }
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
    bool* bufferedHit,
    size_t* bufferedOutOfOrderCount)
{
    results.clear();
    success = false;
    if (bufferedHit != nullptr) {
        *bufferedHit = false;
    }
    if (bufferedOutOfOrderCount != nullptr) {
        *bufferedOutOfOrderCount = 0;
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
        if (bufferedOutOfOrderCount != nullptr) {
            ++(*bufferedOutOfOrderCount);
        }
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
