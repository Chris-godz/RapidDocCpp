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
#include <cctype>
#include <cstdlib>
#include <future>
#include <iomanip>
#include <sstream>
#include <utility>

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

void finalizeFormulaCounters(DocumentResult& result) {
    result.formulaElementCount = 0;
    result.formulaLatexCount = 0;
    result.formulaImageFallbackCount = 0;
    for (const auto& page : result.pages) {
        for (const auto& elem : page.elements) {
            if (elem.type != ContentElement::Type::EQUATION) {
                continue;
            }
            ++result.formulaElementCount;
            if (!elem.text.empty()) {
                ++result.formulaLatexCount;
            } else if (!elem.imagePath.empty()) {
                ++result.formulaImageFallbackCount;
            }
        }
    }
}

struct OcrWorkItem {
    LayoutBox box;
    ContentElement::Type type = ContentElement::Type::TEXT;
    int pageIndex = 0;
    float confidence = 0.0f;
    cv::Mat crop;
    bool skipped = false;
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

bool envFlagEnabled(const char* key) {
    const char* raw = std::getenv(key);
    if (raw == nullptr) {
        return false;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return value == "1" || value == "true" || value == "yes" || value == "on";
}

int envIntOrDefault(const char* key, int defaultValue) {
    const char* raw = std::getenv(key);
    if (raw == nullptr || *raw == '\0') {
        return defaultValue;
    }
    try {
        return std::stoi(raw);
    } catch (...) {
        return defaultValue;
    }
}

cv::Rect clampRectToPage(const LayoutBox& box, int pageWidth, int pageHeight) {
    return box.toRect() & cv::Rect(0, 0, pageWidth, pageHeight);
}

double intersectionArea(const cv::Rect& a, const cv::Rect& b) {
    const cv::Rect inter = a & b;
    if (inter.width <= 0 || inter.height <= 0) {
        return 0.0;
    }
    return static_cast<double>(inter.width) * static_cast<double>(inter.height);
}

bool centerInside(const cv::Rect& outer, const cv::Rect& inner) {
    const cv::Point center(
        outer.x + (outer.width / 2),
        outer.y + (outer.height / 2));
    return center.x >= inner.x &&
           center.x < (inner.x + inner.width) &&
           center.y >= inner.y &&
           center.y < (inner.y + inner.height);
}

cv::Rect inflateRectClamped(
    const cv::Rect& rect,
    int inflateX,
    int inflateY,
    int pageWidth,
    int pageHeight)
{
    const int x0 = std::max(0, rect.x - inflateX);
    const int y0 = std::max(0, rect.y - inflateY);
    const int x1 = std::min(pageWidth, rect.x + rect.width + inflateX);
    const int y1 = std::min(pageHeight, rect.y + rect.height + inflateY);
    return cv::Rect(x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0));
}

std::string normalizeAsciiNoSpaceLower(const std::string& text) {
    std::string out;
    out.reserve(text.size());
    for (unsigned char c : text) {
        if (std::isspace(c)) {
            continue;
        }
        out.push_back(static_cast<char>(std::tolower(c)));
    }
    return out;
}

bool looksLikeFigureContextText(const std::string& text) {
    const std::string normalized = normalizeAsciiNoSpaceLower(text);
    if (normalized.empty()) {
        return false;
    }
    return normalized.find("fig") != std::string::npos ||
           normalized.find("figure") != std::string::npos;
}

std::vector<bool> buildContainedEquationMask(
    const std::vector<LayoutBox>& equationBoxes,
    int pageWidth,
    int pageHeight)
{
    std::vector<bool> contained(equationBoxes.size(), false);
    std::vector<cv::Rect> rois(equationBoxes.size());
    std::vector<double> areas(equationBoxes.size(), 0.0);
    for (size_t i = 0; i < equationBoxes.size(); ++i) {
        rois[i] = clampRectToPage(equationBoxes[i], pageWidth, pageHeight);
        if (rois[i].width > 0 && rois[i].height > 0) {
            areas[i] = static_cast<double>(rois[i].width) * static_cast<double>(rois[i].height);
        }
    }

    constexpr double kContainerAreaRatio = 2.5;
    constexpr double kContainmentOverlap = 0.92;
    for (size_t i = 0; i < equationBoxes.size(); ++i) {
        if (areas[i] <= 0.0) {
            continue;
        }
        for (size_t j = 0; j < equationBoxes.size(); ++j) {
            if (i == j || areas[j] <= 0.0) {
                continue;
            }
            if (areas[j] <= areas[i] * kContainerAreaRatio) {
                continue;
            }
            const double overlap = intersectionArea(rois[i], rois[j]) / areas[i];
            if (overlap >= kContainmentOverlap) {
                contained[i] = true;
                break;
            }
        }
    }
    return contained;
}

bool shouldFilterFormulaCandidateConservative(
    const LayoutBox& candidate,
    const std::vector<cv::Rect>& figureRois,
    const std::vector<cv::Rect>& figureCaptionRois,
    const std::vector<cv::Rect>& figureTextContextRois,
    int pageWidth,
    int pageHeight)
{
    if (pageWidth <= 0 || pageHeight <= 0) {
        return false;
    }

    const cv::Rect candidateRoi = clampRectToPage(candidate, pageWidth, pageHeight);
    if (candidateRoi.width <= 0 || candidateRoi.height <= 0) {
        return false;
    }

    const double pageArea = static_cast<double>(pageWidth) * static_cast<double>(pageHeight);
    const double candidateArea =
        static_cast<double>(candidateRoi.width) * static_cast<double>(candidateRoi.height);
    if (candidateArea <= 0.0 || pageArea <= 0.0) {
        return false;
    }

    const double areaRatio = candidateArea / pageArea;
    const double widthRatio = static_cast<double>(candidateRoi.width) / pageWidth;
    const double heightRatio = static_cast<double>(candidateRoi.height) / pageHeight;

    // Conservative gate: only touch small candidates that are likely panel markers.
    constexpr double kSmallAreaRatio = 0.0035;
    constexpr double kSmallWidthRatio = 0.14;
    constexpr double kSmallHeightRatio = 0.10;
    const bool isSmallCandidate =
        areaRatio <= kSmallAreaRatio &&
        widthRatio <= kSmallWidthRatio &&
        heightRatio <= kSmallHeightRatio;
    if (!isSmallCandidate) {
        return false;
    }

    double maxFigureOverlap = 0.0;
    bool centerInFigure = false;
    for (const auto& figureRoi : figureRois) {
        if (figureRoi.width <= 0 || figureRoi.height <= 0) {
            continue;
        }
        const double overlap = intersectionArea(candidateRoi, figureRoi) / candidateArea;
        maxFigureOverlap = std::max(maxFigureOverlap, overlap);
        if (!centerInFigure && centerInside(candidateRoi, figureRoi)) {
            centerInFigure = true;
        }
    }

    double maxCaptionOverlap = 0.0;
    for (const auto& captionRoi : figureCaptionRois) {
        if (captionRoi.width <= 0 || captionRoi.height <= 0) {
            continue;
        }
        const double overlap = intersectionArea(candidateRoi, captionRoi) / candidateArea;
        maxCaptionOverlap = std::max(maxCaptionOverlap, overlap);
    }

    constexpr double kTinyAreaRatio = 0.0018;
    constexpr double kFigureOverlapGate = 0.55;
    constexpr double kCaptionOverlapGate = 0.65;
    if (areaRatio <= kTinyAreaRatio && centerInFigure) {
        return true;
    }
    if (maxFigureOverlap >= kFigureOverlapGate) {
        return true;
    }
    if (maxCaptionOverlap >= kCaptionOverlapGate) {
        return true;
    }

    for (const auto& textContextRoi : figureTextContextRois) {
        if (textContextRoi.width <= 0 || textContextRoi.height <= 0) {
            continue;
        }
        const double overlap = intersectionArea(candidateRoi, textContextRoi) / candidateArea;
        if (overlap >= 0.50) {
            return true;
        }
        const cv::Rect expanded = inflateRectClamped(
            textContextRoi,
            24,
            12,
            pageWidth,
            pageHeight);
        if (expanded.width > 0 && expanded.height > 0 &&
            centerInside(candidateRoi, expanded) &&
            areaRatio <= kSmallAreaRatio) {
            return true;
        }
    }
    return false;
}

std::vector<OcrWorkItem> buildOcrWorkItems(
    const cv::Mat& image,
    const std::vector<LayoutBox>& textBoxes,
    int pageIndex)
{
    std::vector<OcrWorkItem> items;
    items.reserve(textBoxes.size());

    for (const auto& box : textBoxes) {
        OcrWorkItem item;
        item.box = box;
        item.type = (box.category == LayoutCategory::TITLE)
                        ? ContentElement::Type::TITLE
                        : ContentElement::Type::TEXT;
        item.pageIndex = pageIndex;
        item.confidence = box.confidence;

        cv::Rect roi = box.toRect() & cv::Rect(0, 0, image.cols, image.rows);
        if (roi.width <= 0 || roi.height <= 0) {
            item.skipped = true;
        } else {
            // CPU-only crop clone: safe outside NPU serial lock.
            item.crop = image(roi).clone();
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

    if (config_.stages.enableFormula) {
        FormulaRecognizerConfig formulaCfg;
        formulaCfg.onnxModelPath = config_.models.formulaOnnxModel;
        formulaCfg.inputSize = 384;
        formulaCfg.enableCpuMemArena = false;
        formulaCfg.maxBatchSize = 8;

        const char* ortProfileRaw = std::getenv("RAPIDDOC_FORMULA_ORT_PROFILE");
        const std::string ortProfile = ortProfileRaw == nullptr ? "" : std::string(ortProfileRaw);
        if (ortProfile == "constrained") {
            formulaCfg.sequentialExecution = true;
            formulaCfg.intraOpThreads = 1;
            formulaCfg.interOpThreads = 1;
        } else if (ortProfile == "sequential") {
            formulaCfg.sequentialExecution = true;
        }

        const int intraThreads = envIntOrDefault("RAPIDDOC_FORMULA_ORT_INTRA_THREADS", -1);
        const int interThreads = envIntOrDefault("RAPIDDOC_FORMULA_ORT_INTER_THREADS", -1);
        if (intraThreads > 0) {
            formulaCfg.intraOpThreads = intraThreads;
        }
        if (interThreads > 0) {
            formulaCfg.interOpThreads = interThreads;
        }

        formulaRecognizer_ = std::make_unique<FormulaRecognizer>(formulaCfg);
        if (!formulaRecognizer_->initialize()) {
            LOG_ERROR("Failed to initialize formula recognizer");
            return false;
        }
        formulaDualSessionEnabled_ = envFlagEnabled("RAPIDDOC_FORMULA_DUAL_SESSION");
        formulaDualSessionMinCrops_ = static_cast<size_t>(std::max(
            1,
            envIntOrDefault("RAPIDDOC_FORMULA_DUAL_SESSION_MIN_CROPS", 96)));
        formulaRecognizerSecondary_.reset();
        if (formulaDualSessionEnabled_) {
            formulaRecognizerSecondary_ = std::make_unique<FormulaRecognizer>(formulaCfg);
            if (!formulaRecognizerSecondary_->initialize()) {
                LOG_WARN("Secondary formula recognizer init failed, fallback to single session");
                formulaRecognizerSecondary_.reset();
                formulaDualSessionEnabled_ = false;
            }
        }

        LOG_INFO("Formula recognizer initialized (ONNX Runtime)");
        LOG_INFO(
            "Formula infer profile: ort_profile='{}', dual_session={}, dual_session_min_crops={}",
            ortProfile.empty() ? "default" : ortProfile,
            formulaDualSessionEnabled_ ? "enabled" : "disabled",
            formulaDualSessionMinCrops_);
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

        // Detection model path aligned with Python README lane model naming.
        // Keep single-det in C++ pipeline core to avoid reintroducing legacy outer schedulers.
        ocrCfg.detectorConfig.model640Path = config_.models.ocrModelDir + "/det_v5_640_640.dxnn";
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
        ocrCfg.deviceId = config_.runtime.deviceId;

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

std::string DocPipeline::resolveFormulaCapability(const ExecutionContext& ctx) const {
    if (!ctx.stages.enableFormula) {
        return "disabled";
    }
    if (formulaRecognizeHook_) {
        return "latex_onnxruntime";
    }
    if (formulaRecognizer_ && formulaRecognizer_->isInitialized()) {
        return "latex_onnxruntime";
    }
    return "disabled";
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

DocumentResult DocPipeline::processPdfInternal(
    const std::string& pdfPath,
    const ExecutionContext& ctx)
{
    LOG_INFO("Processing PDF: {}", pdfPath);
    auto startTime = std::chrono::steady_clock::now();

    DocumentResult result;
    result.formulaCapability = resolveFormulaCapability(ctx);

    if (!initialized_) {
        LOG_ERROR("Pipeline not initialized");
        return result;
    }

    reportProgress("PDF Render", 0, 1);
    auto renderStart = std::chrono::steady_clock::now();

    std::vector<PageImage> pageImages;
    if (ctx.stages.enablePdfRender) {
        PdfRenderConfig pdfCfg;
        pdfCfg.dpi = ctx.runtime.pdfDpi;
        pdfCfg.maxPages = ctx.runtime.maxPages;
        pdfCfg.startPageId = ctx.runtime.startPageId;
        pdfCfg.endPageId = ctx.runtime.endPageId;
        pdfCfg.maxConcurrentRenders = ctx.runtime.maxConcurrentPages;
        PdfRenderer renderer(pdfCfg);
        pageImages = renderer.renderFile(pdfPath);
    }

    auto renderEnd = std::chrono::steady_clock::now();
    result.stats.pdfRenderTimeMs =
        std::chrono::duration<double, std::milli>(renderEnd - renderStart).count();
    result.totalPages = static_cast<int>(pageImages.size());

    if (pageImages.empty()) {
        LOG_WARN("No pages rendered from PDF");
        return result;
    }

    LOG_INFO("Rendered {} pages from PDF", pageImages.size());

    const bool deferFormulaStage = ctx.stages.enableFormula;
    for (size_t i = 0; i < pageImages.size(); i++) {
        reportProgress("Processing", static_cast<int>(i + 1), static_cast<int>(pageImages.size()));

        PageResult pageResult = processPage(
            pageImages[i],
            ctx,
            deferFormulaStage,
            deferFormulaStage && ctx.stages.enableReadingOrder);
        result.pages.push_back(std::move(pageResult));
        result.processedPages++;
    }
    if (deferFormulaStage) {
        runDocumentFormulaStage(pageImages, result, ctx);
    }

    finalizeDocumentStats(result);
    finalizeFormulaCounters(result);

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
    result.formulaCapability = resolveFormulaCapability(ctx);
    if (!initialized_) {
        LOG_ERROR("Pipeline not initialized");
        return result;
    }

    auto startTime = std::chrono::steady_clock::now();

    std::vector<PageImage> pageImages;
    auto renderStart = std::chrono::steady_clock::now();
    if (ctx.stages.enablePdfRender) {
        PdfRenderConfig pdfCfg;
        pdfCfg.dpi = ctx.runtime.pdfDpi;
        pdfCfg.maxPages = ctx.runtime.maxPages;
        pdfCfg.startPageId = ctx.runtime.startPageId;
        pdfCfg.endPageId = ctx.runtime.endPageId;
        pdfCfg.maxConcurrentRenders = ctx.runtime.maxConcurrentPages;
        PdfRenderer renderer(pdfCfg);
        pageImages = renderer.renderFromMemory(data, size);
    }
    auto renderEnd = std::chrono::steady_clock::now();

    result.stats.pdfRenderTimeMs =
        std::chrono::duration<double, std::milli>(renderEnd - renderStart).count();
    result.totalPages = static_cast<int>(pageImages.size());

    const bool deferFormulaStage = ctx.stages.enableFormula;
    for (size_t i = 0; i < pageImages.size(); i++) {
        PageResult pageResult = processPage(
            pageImages[i],
            ctx,
            deferFormulaStage,
            deferFormulaStage && ctx.stages.enableReadingOrder);
        result.pages.push_back(std::move(pageResult));
        result.processedPages++;
    }
    if (deferFormulaStage) {
        runDocumentFormulaStage(pageImages, result, ctx);
    }

    finalizeDocumentStats(result);
    finalizeFormulaCounters(result);

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
    result.formulaCapability = resolveFormulaCapability(ctx);
    if (!initialized_) {
        LOG_ERROR("Pipeline not initialized");
        return result;
    }

    PageImage pageImage;
    pageImage.image = image.clone();
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
    finalizeFormulaCounters(result);

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
    pageImage.image = image.clone();
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
    return processPage(pageImage, ctx, false, false);
}

PageResult DocPipeline::processPage(
    const PageImage& pageImage,
    const ExecutionContext& ctx,
    bool deferFormulaStage,
    bool deferReadingOrderStage)
{
    auto startTime = std::chrono::steady_clock::now();
    PageResult result;
    result.pageIndex = pageImage.pageIndex;
    result.pageWidth = pageImage.image.cols;
    result.pageHeight = pageImage.image.rows;

    const cv::Mat& image = pageImage.image;

    std::vector<LayoutBox> textBoxes;
    std::vector<LayoutBox> tableBoxes;
    std::vector<LayoutBox> figureBoxes;
    std::vector<LayoutBox> equationBoxes;
    std::vector<LayoutBox> unsupportedBoxes;

    double npuLockWaitTotalMs = 0.0;
    double npuLockHoldTotalMs = 0.0;
    double npuSerialTotalMs = 0.0;
    double cpuOnlyTotalMs = 0.0;

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
        auto bucketEnd = std::chrono::steady_clock::now();
        cpuOnlyTotalMs +=
            std::chrono::duration<double, std::milli>(bucketEnd - bucketStart).count();
    }

    // OCR on text regions: move ROI crop, element assembly, and text concatenation to CPU-only.
    if (ctx.stages.enableOcr) {
        std::vector<OcrWorkItem> ocrWorkItems;
        {
            auto prepStart = std::chrono::steady_clock::now();
            ocrWorkItems = buildOcrWorkItems(image, textBoxes, pageImage.pageIndex);
            auto prepEnd = std::chrono::steady_clock::now();
            cpuOnlyTotalMs +=
                std::chrono::duration<double, std::milli>(prepEnd - prepStart).count();
        }

        std::vector<OcrFetchResult> fetchResults(ocrWorkItems.size());
        result.stats.ocrTimeMs = runNpuSerialized([&]() {
            for (size_t i = 0; i < ocrWorkItems.size(); ++i) {
                const auto& item = ocrWorkItems[i];
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
                fetch.fetched = waitForOcrResult(taskId, fetch.results, fetch.success);
                if (!fetch.fetched) {
                    LOG_WARN("OCR timeout for task {}", taskId);
                }
            }
        });

        {
            auto assembleStart = std::chrono::steady_clock::now();
            for (size_t i = 0; i < ocrWorkItems.size(); ++i) {
                const auto& item = ocrWorkItems[i];
                ContentElement elem;
                elem.type = item.type;
                elem.layoutBox = item.box;
                elem.confidence = item.confidence;
                elem.pageIndex = item.pageIndex;

                if (item.skipped) {
                    elem.skipped = true;
                    result.elements.push_back(std::move(elem));
                    continue;
                }

                const auto& fetch = fetchResults[i];
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
        {
            auto prepStart = std::chrono::steady_clock::now();
            tableWorkItems.reserve(tableBoxes.size());
            for (const auto& box : tableBoxes) {
                TableWorkItem item;
                item.box = box;
                item.pageIndex = pageImage.pageIndex;
                cv::Rect roi = box.toRect() & cv::Rect(0, 0, image.cols, image.rows);
                if (roi.width <= 0 || roi.height <= 0) {
                    item.invalidRoi = true;
                } else {
                    // CPU-only crop clone: safe outside NPU serial lock.
                    item.crop = image(roi).clone();
                }
                tableWorkItems.push_back(std::move(item));
            }
            auto prepEnd = std::chrono::steady_clock::now();
            cpuOnlyTotalMs +=
                std::chrono::duration<double, std::milli>(prepEnd - prepStart).count();
        }

        std::vector<TableNpuResult> tableNpuResults;
        const double tableNpuStageMs = runNpuSerialized([&]() {
            tableNpuResults.reserve(tableWorkItems.size());
            for (const auto& item : tableWorkItems) {
                TableNpuResult npuResult;
                npuResult.box = item.box;
                npuResult.pageIndex = item.pageIndex;

                if (item.invalidRoi || item.crop.empty()) {
                    npuResult.hasFallback = true;
                    npuResult.fallbackReason = "invalid_table_bbox";
                    tableNpuResults.push_back(std::move(npuResult));
                    continue;
                }

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

                    const int64_t ocrTaskId = allocateOcrTaskId();
                    if (submitOcrTask(tableWorkItems[i].crop, ocrTaskId)) {
                        bool ok = false;
                        if (!(waitForOcrResult(ocrTaskId, npuResult.ocrBoxes, ok) && ok)) {
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
            tableElements.reserve(tableNpuResults.size());
            for (auto& npuResult : tableNpuResults) {
                if (npuResult.hasFallback) {
                    tableElements.push_back(makeTableFallbackElement(
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
                    tableElements.push_back(makeTableFallbackElement(
                        npuResult.box, npuResult.pageIndex, "illegal_table_structure"));
                    continue;
                }

                if (elem.html.empty()) {
                    tableElements.push_back(makeTableFallbackElement(
                        npuResult.box, npuResult.pageIndex, "empty_table_html"));
                    continue;
                }

                elem.skipped = false;
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

    // Formula: LaTeX-first recognition with per-element image fallback.
    if (ctx.stages.enableFormula && !deferFormulaStage) {
        auto formulaStart = std::chrono::steady_clock::now();
        auto formulaElements =
            runFormulaRecognition(image, equationBoxes, pageImage.pageIndex, ctx);
        result.elements.insert(
            result.elements.end(),
            formulaElements.begin(),
            formulaElements.end());
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

    if (!deferReadingOrderStage) {
        runReadingOrderStage(result, ctx);
    }

    auto cpuEnd = std::chrono::steady_clock::now();
    cpuOnlyTotalMs +=
        std::chrono::duration<double, std::milli>(cpuEnd - cpuStart).count();
    result.stats.cpuOnlyTimeMs = cpuOnlyTotalMs;

    auto endTime = std::chrono::steady_clock::now();
    result.totalTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    return result;
}

void DocPipeline::runReadingOrderStage(PageResult& result, const ExecutionContext& ctx) {
    if (!ctx.stages.enableReadingOrder || result.elements.empty()) {
        return;
    }

    auto orderStart = std::chrono::steady_clock::now();
    std::vector<LayoutBox> sortBoxes;
    sortBoxes.reserve(result.elements.size());
    for (const auto& elem : result.elements) {
        sortBoxes.push_back(elem.layoutBox);
    }

    const auto sortedIndices = xycutPlusSort(sortBoxes, result.pageWidth, result.pageHeight);
    std::vector<ContentElement> sortedElements;
    sortedElements.reserve(result.elements.size());
    for (int i = 0; i < static_cast<int>(sortedIndices.size()); i++) {
        const int idx = sortedIndices[i];
        result.elements[idx].readingOrder = i;
        sortedElements.push_back(result.elements[idx]);
    }
    result.elements = std::move(sortedElements);

    auto orderEnd = std::chrono::steady_clock::now();
    const double orderMs =
        std::chrono::duration<double, std::milli>(orderEnd - orderStart).count();
    result.stats.readingOrderTimeMs += orderMs;
    result.stats.cpuOnlyTimeMs += orderMs;
    result.totalTimeMs += orderMs;
}

void DocPipeline::runDocumentFormulaStage(
    const std::vector<PageImage>& pageImages,
    DocumentResult& result,
    const ExecutionContext& ctx)
{
    if (!ctx.stages.enableFormula || pageImages.empty() || result.pages.empty()) {
        for (auto& page : result.pages) {
            runReadingOrderStage(page, ctx);
        }
        return;
    }

    struct FormulaCropRef {
        size_t pageResultIndex = 0;
        size_t equationIndex = 0;
    };

    const size_t pageCount = std::min(pageImages.size(), result.pages.size());
    std::vector<std::vector<LayoutBox>> equationBoxesByPage(pageCount);
    std::vector<std::vector<cv::Rect>> equationRoisByPage(pageCount);
    std::vector<std::vector<std::string>> latexByPage(pageCount);
    std::vector<int> equationCountByPage(pageCount, 0);

    std::vector<cv::Mat> allCrops;
    std::vector<FormulaCropRef> cropRefs;

    result.formulaTimingBill = FormulaTimingBill{};
    const auto formulaStart = std::chrono::steady_clock::now();

    const auto regionCollectStart = std::chrono::steady_clock::now();
    int totalEquationRawCount = 0;
    int totalEquationDropped = 0;
    int totalEquationDroppedContained = 0;
    int totalEquationDroppedFigureContext = 0;
    for (size_t pageIdx = 0; pageIdx < pageCount; ++pageIdx) {
        const auto& pageResult = result.pages[pageIdx];
        const auto& pageImage = pageImages[pageIdx].image;
        const int pageWidth = pageImage.cols;
        const int pageHeight = pageImage.rows;

        const auto rawEquationBoxes = pageResult.layoutResult.getEquationBoxes();
        totalEquationRawCount += static_cast<int>(rawEquationBoxes.size());

        std::vector<cv::Rect> figureRois;
        std::vector<cv::Rect> figureCaptionRois;
        std::vector<cv::Rect> figureTextContextRois;
        figureRois.reserve(pageResult.layoutResult.boxes.size());
        figureCaptionRois.reserve(pageResult.layoutResult.boxes.size());
        figureTextContextRois.reserve(pageResult.elements.size());
        for (const auto& box : pageResult.layoutResult.boxes) {
            if (box.category == LayoutCategory::FIGURE) {
                const cv::Rect roi = clampRectToPage(box, pageWidth, pageHeight);
                if (roi.width > 0 && roi.height > 0) {
                    figureRois.push_back(roi);
                }
            } else if (box.category == LayoutCategory::FIGURE_CAPTION) {
                const cv::Rect roi = clampRectToPage(box, pageWidth, pageHeight);
                if (roi.width > 0 && roi.height > 0) {
                    figureCaptionRois.push_back(roi);
                }
            }
        }
        for (const auto& elem : pageResult.elements) {
            if (elem.type != ContentElement::Type::TEXT &&
                elem.type != ContentElement::Type::TITLE) {
                continue;
            }
            if (!looksLikeFigureContextText(elem.text)) {
                continue;
            }
            const cv::Rect roi = clampRectToPage(elem.layoutBox, pageWidth, pageHeight);
            if (roi.width > 0 && roi.height > 0) {
                figureTextContextRois.push_back(roi);
            }
        }

        const auto containedMask = buildContainedEquationMask(
            rawEquationBoxes,
            pageWidth,
            pageHeight);

        std::vector<LayoutBox> filteredEquationBoxes;
        filteredEquationBoxes.reserve(rawEquationBoxes.size());
        int droppedOnPage = 0;
        int droppedContainedOnPage = 0;
        int droppedFigureCtxOnPage = 0;
        for (size_t equationIdx = 0; equationIdx < rawEquationBoxes.size(); ++equationIdx) {
            const auto& equationBox = rawEquationBoxes[equationIdx];
            if (equationIdx < containedMask.size() && containedMask[equationIdx]) {
                ++droppedOnPage;
                ++droppedContainedOnPage;
                continue;
            }
            if (shouldFilterFormulaCandidateConservative(
                    equationBox,
                    figureRois,
                    figureCaptionRois,
                    figureTextContextRois,
                    pageWidth,
                    pageHeight)) {
                ++droppedOnPage;
                ++droppedFigureCtxOnPage;
                continue;
            }
            filteredEquationBoxes.push_back(equationBox);
        }
        totalEquationDropped += droppedOnPage;
        totalEquationDroppedContained += droppedContainedOnPage;
        totalEquationDroppedFigureContext += droppedFigureCtxOnPage;

        equationBoxesByPage[pageIdx] = std::move(filteredEquationBoxes);
        equationRoisByPage[pageIdx].resize(equationBoxesByPage[pageIdx].size());
        latexByPage[pageIdx].resize(equationBoxesByPage[pageIdx].size());
        equationCountByPage[pageIdx] = static_cast<int>(equationBoxesByPage[pageIdx].size());
    }
    const auto regionCollectEnd = std::chrono::steady_clock::now();
    result.formulaTimingBill.regionCollectMs =
        std::chrono::duration<double, std::milli>(regionCollectEnd - regionCollectStart).count();
    if (totalEquationDropped > 0) {
        LOG_INFO(
            "Formula pre-infer conservative gate dropped {}/{} candidates (contained={}, figure_context={})",
            totalEquationDropped,
            totalEquationRawCount,
            totalEquationDroppedContained,
            totalEquationDroppedFigureContext);
    }

    const auto cropPrepareStart = std::chrono::steady_clock::now();
    for (size_t pageIdx = 0; pageIdx < pageCount; ++pageIdx) {
        const auto& image = pageImages[pageIdx].image;
        const auto& equationBoxes = equationBoxesByPage[pageIdx];
        for (size_t equationIdx = 0; equationIdx < equationBoxes.size(); ++equationIdx) {
            const auto& box = equationBoxes[equationIdx];
            const cv::Rect roi = box.toRect() & cv::Rect(0, 0, image.cols, image.rows);
            equationRoisByPage[pageIdx][equationIdx] = roi;
            if (roi.width <= 0 || roi.height <= 0) {
                continue;
            }
            allCrops.emplace_back(image(roi));
            cropRefs.push_back(FormulaCropRef{
                pageIdx,
                equationIdx,
            });
        }
    }
    const auto cropPrepareEnd = std::chrono::steady_clock::now();
    result.formulaTimingBill.cropPrepareMs =
        std::chrono::duration<double, std::milli>(cropPrepareEnd - cropPrepareStart).count();
    result.formulaTimingBill.cropCount = static_cast<int>(allCrops.size());

    std::vector<std::string> latexes(allCrops.size());
    FormulaRecognizer::BatchTiming batchTimingPrimary;
    FormulaRecognizer::BatchTiming batchTimingSecondary;
    double inferWallMs = 0.0;
    if (!allCrops.empty()) {
        const auto inferWallStart = std::chrono::steady_clock::now();
        if (formulaRecognizeHook_) {
            latexes = formulaRecognizeHook_(allCrops);
            batchTimingPrimary.inferMs =
                std::chrono::duration<double, std::milli>(
                    std::chrono::steady_clock::now() - inferWallStart).count();
            batchTimingPrimary.totalMs = batchTimingPrimary.inferMs;
            batchTimingPrimary.cropCount = static_cast<int>(allCrops.size());
            batchTimingPrimary.batchCount = 1;
        } else if (formulaRecognizer_ && formulaRecognizer_->isInitialized()) {
            const bool useDualSession =
                formulaDualSessionEnabled_ &&
                formulaRecognizerSecondary_ &&
                formulaRecognizerSecondary_->isInitialized() &&
                allCrops.size() >= formulaDualSessionMinCrops_;
            if (!useDualSession) {
                latexes = formulaRecognizer_->recognizeBatch(allCrops, &batchTimingPrimary);
            } else {
                const size_t splitIndex = allCrops.size() / 2;
                std::vector<cv::Mat> primaryCrops;
                std::vector<cv::Mat> secondaryCrops;
                primaryCrops.reserve(splitIndex);
                secondaryCrops.reserve(allCrops.size() - splitIndex);
                for (size_t i = 0; i < splitIndex; ++i) {
                    primaryCrops.emplace_back(allCrops[i]);
                }
                for (size_t i = splitIndex; i < allCrops.size(); ++i) {
                    secondaryCrops.emplace_back(allCrops[i]);
                }

                auto secondaryFuture = std::async(
                    std::launch::async,
                    [this, crops = std::move(secondaryCrops)]() mutable {
                        FormulaRecognizer::BatchTiming timing;
                        auto values = formulaRecognizerSecondary_->recognizeBatch(crops, &timing);
                        return std::make_pair(std::move(values), timing);
                    });

                auto primaryLatex = formulaRecognizer_->recognizeBatch(
                    primaryCrops,
                    &batchTimingPrimary);
                auto secondaryResult = secondaryFuture.get();
                auto& secondaryLatex = secondaryResult.first;
                batchTimingSecondary = secondaryResult.second;

                latexes.assign(allCrops.size(), "");
                for (size_t i = 0; i < primaryLatex.size() && i < splitIndex; ++i) {
                    latexes[i] = primaryLatex[i];
                }
                for (size_t i = 0;
                     i < secondaryLatex.size() && (splitIndex + i) < latexes.size();
                     ++i) {
                    latexes[splitIndex + i] = secondaryLatex[i];
                }
            }
        }
        inferWallMs = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - inferWallStart).count();
    }
    result.formulaTimingBill.cropPrepareMs +=
        (batchTimingPrimary.preprocessMs + batchTimingSecondary.preprocessMs);
    result.formulaTimingBill.inferMs = inferWallMs;
    result.formulaTimingBill.decodeMs =
        batchTimingPrimary.decodeMs + batchTimingSecondary.decodeMs;
    result.formulaTimingBill.normalizeMs =
        batchTimingPrimary.normalizeMs + batchTimingSecondary.normalizeMs;
    result.formulaTimingBill.batchCount =
        batchTimingPrimary.batchCount + batchTimingSecondary.batchCount;

    const auto writebackStart = std::chrono::steady_clock::now();
    for (size_t i = 0; i < cropRefs.size(); ++i) {
        const auto& ref = cropRefs[i];
        if (ref.pageResultIndex >= latexByPage.size() ||
            ref.equationIndex >= latexByPage[ref.pageResultIndex].size()) {
            continue;
        }
        if (i < latexes.size()) {
            latexByPage[ref.pageResultIndex][ref.equationIndex] = latexes[i];
        }
    }

    for (size_t pageIdx = 0; pageIdx < pageCount; ++pageIdx) {
        auto& pageResult = result.pages[pageIdx];
        const auto& pageImage = pageImages[pageIdx];
        const auto& image = pageImage.image;
        const auto& equationBoxes = equationBoxesByPage[pageIdx];
        const auto& rois = equationRoisByPage[pageIdx];
        const auto& latexesForPage = latexByPage[pageIdx];

        pageResult.elements.reserve(pageResult.elements.size() + equationBoxes.size());
        for (size_t equationIdx = 0; equationIdx < equationBoxes.size(); ++equationIdx) {
            const auto& box = equationBoxes[equationIdx];
            const auto& roi = rois[equationIdx];
            const bool validRoi = roi.width > 0 && roi.height > 0;

            ContentElement elem;
            elem.type = ContentElement::Type::EQUATION;
            elem.layoutBox = box;
            elem.pageIndex = pageResult.pageIndex;
            elem.confidence = box.confidence;

            const std::string latex =
                (equationIdx < latexesForPage.size()) ? latexesForPage[equationIdx] : std::string();
            if (!latex.empty()) {
                elem.text = latex;
                elem.skipped = false;
            } else if (validRoi && ctx.runtime.saveImages) {
                const std::string filename = "images/page" + std::to_string(pageResult.pageIndex) +
                                             "_eq" + std::to_string(equationIdx) + ".png";
                const std::string filepath = ctx.runtime.outputDir + "/" + filename;
                std::filesystem::create_directories(
                    std::filesystem::path(filepath).parent_path());
                cv::imwrite(filepath, image(roi));
                elem.imagePath = filename;
                elem.skipped = false;
            } else {
                elem.skipped = true;
            }
            pageResult.elements.push_back(std::move(elem));
        }
    }
    const auto writebackEnd = std::chrono::steady_clock::now();
    result.formulaTimingBill.writebackMs =
        std::chrono::duration<double, std::milli>(writebackEnd - writebackStart).count();

    const auto formulaEnd = std::chrono::steady_clock::now();
    result.formulaTimingBill.totalMs =
        std::chrono::duration<double, std::milli>(formulaEnd - formulaStart).count();
    const double formulaMs = result.formulaTimingBill.totalMs;

    int totalEquationCount = 0;
    for (int count : equationCountByPage) {
        totalEquationCount += count;
    }
    if (totalEquationCount <= 0) {
        totalEquationCount = 1;
    }

    for (size_t pageIdx = 0; pageIdx < pageCount; ++pageIdx) {
        auto& pageResult = result.pages[pageIdx];
        const double pageFormulaMs =
            formulaMs * static_cast<double>(equationCountByPage[pageIdx]) /
            static_cast<double>(totalEquationCount);
        pageResult.stats.formulaTimeMs += pageFormulaMs;
        pageResult.stats.cpuOnlyTimeMs += pageFormulaMs;
        pageResult.totalTimeMs += pageFormulaMs;

        runReadingOrderStage(pageResult, ctx);
    }

    for (size_t pageIdx = pageCount; pageIdx < result.pages.size(); ++pageIdx) {
        runReadingOrderStage(result.pages[pageIdx], ctx);
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
    bool& success)
{
    results.clear();
    success = false;

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

    for (size_t bi = 0; bi < textBoxes.size(); ++bi) {
        const auto& box = textBoxes[bi];
        ContentElement elem;
        elem.type = (box.category == LayoutCategory::TITLE)
                    ? ContentElement::Type::TITLE
                    : ContentElement::Type::TEXT;
        elem.layoutBox = box;
        elem.confidence = box.confidence;
        elem.pageIndex = pageIndex;

        cv::Rect roi = box.toRect() & cv::Rect(0, 0, image.cols, image.rows);
        if (roi.width <= 0 || roi.height <= 0) {
            elem.skipped = true;
            elements.push_back(elem);
            continue;
        }

        elem.text = ocrOnCrop(image(roi).clone(), allocateOcrTaskId());
        elements.push_back(elem);
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

    for (const auto& box : tableBoxes) {
        ContentElement elem;
        elem.type = ContentElement::Type::TABLE;
        elem.layoutBox = box;
        elem.pageIndex = pageIndex;

        cv::Rect roi = box.toRect() & cv::Rect(0, 0, image.cols, image.rows);
        if (roi.width <= 0 || roi.height <= 0) {
            elements.push_back(makeTableFallbackElement(box, pageIndex, "invalid_table_bbox"));
            continue;
        }

        cv::Mat tableCrop = image(roi);

        TableResult tableResult = recognizeTable(tableCrop);

        if (!tableResult.supported) {
            const std::string reason = (tableResult.type == TableType::WIRELESS)
                                           ? "wireless_table"
                                           : "table_model_unavailable";
            elements.push_back(makeTableFallbackElement(box, pageIndex, reason));
            continue;
        }

        if (tableResult.cells.empty()) {
            elements.push_back(makeTableFallbackElement(box, pageIndex, "no_cell_table"));
            continue;
        }

        // Match approach (like Python match_ocr_cell):
        // 1. Run OCR on the FULL table image once
        // 2. Match each OCR text box to the nearest cell by spatial overlap
        if (ctx.stages.enableOcr && (ocrPipeline_ || (ocrSubmitHook_ && ocrFetchHook_))) {
            const int64_t ocrTaskId = allocateOcrTaskId();
            if (submitOcrTask(tableCrop.clone(), ocrTaskId)) {
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
            elements.push_back(makeTableFallbackElement(box, pageIndex, "illegal_table_structure"));
            continue;
        }

        if (elem.html.empty()) {
            elements.push_back(makeTableFallbackElement(box, pageIndex, "empty_table_html"));
            continue;
        }

        elem.skipped = false;
        elements.push_back(elem);
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

std::vector<ContentElement> DocPipeline::runFormulaRecognition(
    const cv::Mat& image,
    const std::vector<LayoutBox>& equationBoxes,
    int pageIndex)
{
    return runFormulaRecognition(
        image, equationBoxes, pageIndex, makeExecutionContext(nullptr));
}

std::vector<ContentElement> DocPipeline::runFormulaRecognition(
    const cv::Mat& image,
    const std::vector<LayoutBox>& equationBoxes,
    int pageIndex,
    const ExecutionContext& ctx)
{
    std::vector<ContentElement> elements;
    elements.reserve(equationBoxes.size());
    if (equationBoxes.empty()) {
        return elements;
    }

    std::vector<cv::Mat> crops;
    std::vector<cv::Rect> rois;
    std::vector<size_t> validIndices;
    crops.reserve(equationBoxes.size());
    rois.reserve(equationBoxes.size());
    validIndices.reserve(equationBoxes.size());

    for (size_t i = 0; i < equationBoxes.size(); ++i) {
        const auto& box = equationBoxes[i];
        cv::Rect roi = box.toRect() & cv::Rect(0, 0, image.cols, image.rows);
        rois.push_back(roi);
        if (roi.width <= 0 || roi.height <= 0) {
            continue;
        }
        crops.emplace_back(image(roi));
        validIndices.push_back(i);
    }

    std::vector<std::string> latexes(crops.size());
    if (!crops.empty()) {
        if (formulaRecognizeHook_) {
            latexes = formulaRecognizeHook_(crops);
        } else if (formulaRecognizer_ && formulaRecognizer_->isInitialized()) {
            latexes = formulaRecognizer_->recognizeBatch(crops);
        }
    }

    size_t latexCursor = 0;
    for (size_t i = 0; i < equationBoxes.size(); ++i) {
        const auto& box = equationBoxes[i];
        ContentElement elem;
        elem.type = ContentElement::Type::EQUATION;
        elem.layoutBox = box;
        elem.pageIndex = pageIndex;
        elem.confidence = box.confidence;

        const cv::Rect roi = rois[i];
        const bool validRoi = roi.width > 0 && roi.height > 0;
        std::string latex;
        if (validRoi) {
            if (latexCursor < latexes.size()) {
                latex = latexes[latexCursor];
            }
            ++latexCursor;
        }

        if (!latex.empty()) {
            elem.text = latex;
            elem.skipped = false;
        } else if (validRoi && ctx.runtime.saveImages) {
            std::string filename = "images/page" + std::to_string(pageIndex) +
                                   "_eq" + std::to_string(i) + ".png";
            const std::string filepath = ctx.runtime.outputDir + "/" + filename;
            std::filesystem::create_directories(
                std::filesystem::path(filepath).parent_path());
            cv::imwrite(filepath, image(roi));
            elem.imagePath = filename;
            elem.skipped = false;
        } else {
            elem.skipped = true;
        }

        elements.push_back(std::move(elem));
    }

    return elements;
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
    // Utility fallback path: persist equation crops as images.
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
