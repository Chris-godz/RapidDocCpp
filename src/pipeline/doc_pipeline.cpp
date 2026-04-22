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
#include <nlohmann/json.hpp>
#include <filesystem>
#include <chrono>
#include <thread>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <future>
#include <iomanip>
#include <map>
#include <numeric>
#include <sstream>
#include <utility>

namespace fs = std::filesystem;
using json = nlohmann::json;

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
    // When true, the wired NPU path will not be attempted (or produced no
    // cells) and the SLANet+ wireless backend should be used as a second
    // try. OCR is still scheduled for this entry so the matcher has text.
    bool needsWirelessBackend = false;
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

bool envFlagOrDefault(const char* key, bool defaultValue) {
    const char* raw = std::getenv(key);
    if (raw == nullptr) {
        return defaultValue;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
    }
    return defaultValue;
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

json layoutBoxTraceJson(const LayoutBox& box) {
    return json{
        {"bbox", {box.x0, box.y0, box.x1, box.y1}},
        {"category", layoutCategoryToString(box.category)},
        {"category_id", box.clsId},
        {"label", box.label},
        {"confidence", box.confidence},
    };
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

double rectIou(const cv::Rect& a, const cv::Rect& b) {
    const double inter = intersectionArea(a, b);
    if (inter <= 0.0) {
        return 0.0;
    }
    const double areaA = static_cast<double>(a.width) * static_cast<double>(a.height);
    const double areaB = static_cast<double>(b.width) * static_cast<double>(b.height);
    const double denom = areaA + areaB - inter;
    return denom > 0.0 ? inter / denom : 0.0;
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

double verticalOverlapRatio(const cv::Rect& a, const cv::Rect& b) {
    const int y0 = std::max(a.y, b.y);
    const int y1 = std::min(a.y + a.height, b.y + b.height);
    const int overlap = std::max(0, y1 - y0);
    const int minHeight = std::min(a.height, b.height);
    if (minHeight <= 0) {
        return 0.0;
    }
    return static_cast<double>(overlap) / static_cast<double>(minHeight);
}

int horizontalGap(const cv::Rect& a, const cv::Rect& b) {
    if (a.x <= b.x) {
        return b.x - (a.x + a.width);
    }
    return a.x - (b.x + b.width);
}

bool shouldMergeFormulaFragments(
    const LayoutBox& lhs,
    const LayoutBox& rhs,
    int pageWidth,
    int pageHeight)
{
    const cv::Rect left = clampRectToPage(lhs, pageWidth, pageHeight);
    const cv::Rect right = clampRectToPage(rhs, pageWidth, pageHeight);
    if (left.width <= 0 || left.height <= 0 || right.width <= 0 || right.height <= 0) {
        return false;
    }

    const double overlap = verticalOverlapRatio(left, right);
    if (overlap < 0.72) {
        return false;
    }

    const double heightRatio =
        static_cast<double>(std::max(left.height, right.height)) /
        static_cast<double>(std::max(1, std::min(left.height, right.height)));
    if (heightRatio > 1.45) {
        return false;
    }
    if (std::min(left.width, right.width) < 96) {
        return false;
    }

    const int gap = horizontalGap(left, right);
    if (gap < -8) {
        return false;
    }

    const int maxGap = std::max(28, static_cast<int>(
        std::lround(static_cast<double>(std::max(left.height, right.height)) * 0.85)));
    if (gap > maxGap) {
        return false;
    }

    const cv::Rect merged = left | right;
    const double mergedWidthRatio =
        pageWidth > 0 ? static_cast<double>(merged.width) / static_cast<double>(pageWidth) : 1.0;
    // Keep this conservative: only merge near-neighbor fragments, not two formulas
    // that happen to sit on the same baseline across columns.
    return mergedWidthRatio <= 0.38;
}

LayoutBox mergeFormulaBoxes(const LayoutBox& lhs, const LayoutBox& rhs) {
    LayoutBox merged = lhs;
    merged.x0 = std::min(lhs.x0, rhs.x0);
    merged.y0 = std::min(lhs.y0, rhs.y0);
    merged.x1 = std::max(lhs.x1, rhs.x1);
    merged.y1 = std::max(lhs.y1, rhs.y1);
    merged.confidence = std::max(lhs.confidence, rhs.confidence);
    merged.category = LayoutCategory::EQUATION;
    merged.label = lhs.label == rhs.label ? lhs.label : "formula";
    merged.clsId = lhs.clsId == rhs.clsId ? lhs.clsId : 7;
    return merged;
}

bool shouldSuppressDuplicateFormulaCandidate(
    const LayoutBox& kept,
    const LayoutBox& candidate,
    int pageWidth,
    int pageHeight)
{
    const cv::Rect keptRoi = clampRectToPage(kept, pageWidth, pageHeight);
    const cv::Rect candidateRoi = clampRectToPage(candidate, pageWidth, pageHeight);
    if (keptRoi.width <= 0 || keptRoi.height <= 0 ||
        candidateRoi.width <= 0 || candidateRoi.height <= 0) {
        return false;
    }
    if (rectIou(keptRoi, candidateRoi) < 0.92) {
        return false;
    }
    if (verticalOverlapRatio(keptRoi, candidateRoi) < 0.90) {
        return false;
    }
    return kept.confidence >= candidate.confidence;
}

std::string formulaCandidateShapeClass(
    const LayoutBox& candidate,
    int pageWidth,
    int pageHeight)
{
    if (candidate.label == "formula_number" ||
        candidate.label == "interline_equation_number" ||
        candidate.clsId == 19) {
        return "formula_number";
    }

    const cv::Rect roi = clampRectToPage(candidate, pageWidth, pageHeight);
    if (roi.width <= 0 || roi.height <= 0 || pageWidth <= 0 || pageHeight <= 0) {
        return "invalid_roi";
    }
    const double pageArea = static_cast<double>(pageWidth) * static_cast<double>(pageHeight);
    const double areaRatio =
        static_cast<double>(roi.width) * static_cast<double>(roi.height) / pageArea;
    const double widthRatio = static_cast<double>(roi.width) / static_cast<double>(pageWidth);
    const double aspect = static_cast<double>(roi.width) / static_cast<double>(std::max(1, roi.height));

    if (areaRatio <= 0.0018 && widthRatio <= 0.08 && aspect <= 2.4) {
        return "panel_marker_sized";
    }
    if (areaRatio <= 0.0035 && widthRatio <= 0.14) {
        return "small_inline_candidate";
    }
    if (aspect >= 4.0 && widthRatio <= 0.22) {
        return "wide_formula_fragment";
    }
    return "formula_candidate";
}

json formulaBatchProfileJson(
    const FormulaRecognizer::BatchTiming& primary,
    const FormulaRecognizer::BatchTiming& secondary,
    int configuredBatchSize)
{
    json batches = json::array();
    std::map<int, int> batchSizeHistogram;
    std::map<std::string, int> shapeHistogram;
    int singletonBatchCount = 0;
    int maxEffectiveBatchSize = 0;

    auto append = [&](const FormulaRecognizer::BatchTiming& timing, const std::string& session) {
        for (const auto& record : timing.batches) {
            const std::string shape =
                std::to_string(record.targetH) + "x" + std::to_string(record.targetW);
            ++batchSizeHistogram[record.batchSize];
            ++shapeHistogram[shape];
            if (record.batchSize == 1) {
                ++singletonBatchCount;
            }
            maxEffectiveBatchSize = std::max(maxEffectiveBatchSize, record.batchSize);
            batches.push_back(json{
                {"session", session},
                {"batch_size", record.batchSize},
                {"target_h", record.targetH},
                {"target_w", record.targetW},
                {"shape", shape},
                {"infer_ms", record.inferMs},
                {"crop_indices", record.cropIndices},
            });
        }
    };

    append(primary, "primary");
    append(secondary, "secondary");

    json sizeHistogramJson = json::object();
    for (const auto& kv : batchSizeHistogram) {
        sizeHistogramJson[std::to_string(kv.first)] = kv.second;
    }
    json shapeHistogramJson = json::object();
    for (const auto& kv : shapeHistogram) {
        shapeHistogramJson[kv.first] = kv.second;
    }

    return json{
        {"configured_batch_size", configuredBatchSize},
        {"max_effective_batch_size", maxEffectiveBatchSize},
        {"singleton_batch_count", singletonBatchCount},
        {"batch_size_histogram", std::move(sizeHistogramJson)},
        {"tensor_shape_histogram", std::move(shapeHistogramJson)},
        {"batches", std::move(batches)},
    };
}

std::vector<LayoutBox> refineFormulaCandidatesConservative(
    const std::vector<LayoutBox>& rawEquationBoxes,
    int pageIndex,
    int pageWidth,
    int pageHeight,
    json* formulaTrace)
{
    std::vector<LayoutBox> refined;
    refined.reserve(rawEquationBoxes.size());
    std::vector<bool> used(rawEquationBoxes.size(), false);

    for (size_t i = 0; i < rawEquationBoxes.size(); ++i) {
        if (used[i]) {
            continue;
        }
        LayoutBox current = rawEquationBoxes[i];
        used[i] = true;
        json mergedFrom = json::array({static_cast<int>(i)});

        bool changed = true;
        while (changed) {
            changed = false;
            for (size_t j = i + 1; j < rawEquationBoxes.size(); ++j) {
                if (used[j]) {
                    continue;
                }
                if (!shouldMergeFormulaFragments(
                        current,
                        rawEquationBoxes[j],
                        pageWidth,
                        pageHeight)) {
                    continue;
                }
                current = mergeFormulaBoxes(current, rawEquationBoxes[j]);
                used[j] = true;
                mergedFrom.push_back(static_cast<int>(j));
                changed = true;
            }
        }

        if (formulaTrace != nullptr && mergedFrom.size() > 1) {
            json refinement = layoutBoxTraceJson(current);
            refinement["page"] = pageIndex;
            refinement["action"] = "merge_same_line_fragments";
            refinement["source_raw_candidate_indices"] = std::move(mergedFrom);
            (*formulaTrace)["candidate_refinements"].push_back(std::move(refinement));
        }
        refined.push_back(std::move(current));
    }

    std::vector<LayoutBox> deduped;
    deduped.reserve(refined.size());
    for (size_t candidateIdx = 0; candidateIdx < refined.size(); ++candidateIdx) {
        const auto& candidate = refined[candidateIdx];
        bool suppressed = false;
        for (size_t keptIdx = 0; keptIdx < deduped.size(); ++keptIdx) {
            if (!shouldSuppressDuplicateFormulaCandidate(
                    deduped[keptIdx],
                    candidate,
                    pageWidth,
                    pageHeight)) {
                continue;
            }
            if (formulaTrace != nullptr) {
                json refinement = layoutBoxTraceJson(candidate);
                refinement["page"] = pageIndex;
                refinement["action"] = "suppress_duplicate_formula_candidate";
                refinement["source_refined_candidate_index"] =
                    static_cast<int>(candidateIdx);
                refinement["kept_refined_candidate_index"] =
                    static_cast<int>(keptIdx);
                refinement["kept_bbox"] = {
                    deduped[keptIdx].x0,
                    deduped[keptIdx].y0,
                    deduped[keptIdx].x1,
                    deduped[keptIdx].y1,
                };
                (*formulaTrace)["candidate_refinements"].push_back(std::move(refinement));
            }
            suppressed = true;
            break;
        }
        if (!suppressed) {
            deduped.push_back(candidate);
        }
    }
    return deduped;
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

// Adapt pipeline OCR results to the wireless recognizer's decoupled input.
// The wireless recognizer is deliberately headerless w.r.t. DXNN-OCR-cpp.
std::vector<WirelessOcrBox> toWirelessOcrBoxes(
    const std::vector<ocr::PipelineOCRResult>& in)
{
    std::vector<WirelessOcrBox> out;
    out.reserve(in.size());
    for (const auto& r : in) {
        WirelessOcrBox b;
        b.aabb = r.getBoundingRect();
        b.text = r.text;
        out.push_back(std::move(b));
    }
    return out;
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
        layoutCfg.emitDebugBoxes =
            envFlagEnabled("RAPIDDOC_FORMULA_TRACE") ||
            envFlagEnabled("RAPIDDOC_LAYOUT_TRACE_RAW");
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
        formulaCfg.enableCpuMemArena =
            envFlagOrDefault("RAPIDDOC_FORMULA_CPU_MEM_ARENA", true);
        formulaCfg.maxBatchSize =
            std::max(1, envIntOrDefault("RAPIDDOC_FORMULA_MAX_BATCH_SIZE", 8));
        formulaMaxBatchSize_ = formulaCfg.maxBatchSize;
        formulaCfg.packDynamicShapes =
            envFlagOrDefault("RAPIDDOC_FORMULA_PACK_DYNAMIC_SHAPES", true);

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
        // Physics papers etc. often produce 150+ formula crops. CPU ONNX at
        // batch=8 costs ~9s/batch, so serial inference produces a 3-minute
        // tail. Enable a second parallel session by default (env override
        // still supported via RAPIDDOC_FORMULA_DUAL_SESSION=0).
        formulaDualSessionEnabled_ =
            envFlagOrDefault("RAPIDDOC_FORMULA_DUAL_SESSION", true);
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
            "Formula infer profile: ort_profile='{}', max_batch_size={}, cpu_mem_arena={}, dynamic_shape_packing={}, dual_session={}, dual_session_min_crops={}",
            ortProfile.empty() ? "default" : ortProfile,
            formulaMaxBatchSize_,
            formulaCfg.enableCpuMemArena ? "enabled" : "disabled",
            formulaCfg.packDynamicShapes ? "enabled" : "disabled",
            formulaDualSessionEnabled_ ? "enabled" : "disabled",
            formulaDualSessionMinCrops_);
    }

    // Initialize Table recognizer (wired tables only)
    if (config_.stages.enableWiredTable) {
        TableRecognizerConfig tableCfg;
        tableCfg.unetDxnnModelPath = config_.models.tableUnetDxnnModel;
        tableCfg.threshold = config_.runtime.tableConfThreshold;
        tableCfg.deviceId = config_.runtime.deviceId;
        // PaddleCls wired/wireless router (parity with Python TableCls).
        // A missing classifier model is non-fatal: TableRecognizer falls back
        // to its legacy lineRatio heuristic.
        tableCfg.enableTableCls = config_.stages.enableTableClassify;
        tableCfg.tableClsOnnxModelPath = config_.models.tableClsOnnxModel;
        tableRecognizer_ = std::make_unique<TableRecognizer>(tableCfg);
        if (!tableRecognizer_->initialize()) {
            LOG_ERROR("Failed to initialize table recognizer");
            return false;
        }
        LOG_INFO("Table recognizer initialized (wired tables only)");
    }

    // Initialize wireless table recognizer (SLANet+). Non-fatal if the ONNX
    // model is missing: the pipeline falls back to the legacy unsupported
    // placeholder — same behavior as before the port.
    if (config_.stages.enableWirelessTable) {
        TableWirelessRecognizerConfig wCfg;
        wCfg.onnxModelPath = config_.models.slanetPlusOnnxModel;
        wirelessRecognizer_ = std::make_unique<TableWirelessRecognizer>(wCfg);
        if (!wirelessRecognizer_->initialize()) {
            LOG_WARN("Wireless table recognizer disabled (model unavailable): {}",
                     config_.models.slanetPlusOnnxModel);
            wirelessRecognizer_.reset();
        } else {
            LOG_INFO("Wireless table recognizer initialized (SLANet+)");
        }
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

                    // Env-gated routing dump: persist the table crop before recognition.
                    const char* tableDumpDir = std::getenv("RAPIDDOC_TABLE_DUMP_CROPS_DIR");
                    if (tableDumpDir != nullptr && tableDumpDir[0] != '\0') {
                        const fs::path dumpRoot(tableDumpDir);
                        std::error_code ec;
                        fs::create_directories(dumpRoot, ec);
                        if (!ec) {
                            std::ostringstream name;
                            name << "table_p" << std::setw(3) << std::setfill('0')
                                 << item.pageIndex
                                 << "_" << roi.x << "_" << roi.y
                                 << "_" << (roi.x + roi.width) << "_" << (roi.y + roi.height)
                                 << ".png";
                            const fs::path outPath = dumpRoot / name.str();
                            if (!cv::imwrite(outPath.string(), item.crop)) {
                                LOG_WARN("TABLE_ROUTE_TRACE imwrite failed: {}", outPath.string());
                            }
                        } else {
                            LOG_WARN("TABLE_ROUTE_TRACE mkdir {} failed: {}",
                                     dumpRoot.string(), ec.message());
                        }
                    }

                    const char* tableTrace = std::getenv("RAPIDDOC_TABLE_TRACE");
                    if (tableTrace != nullptr && tableTrace[0] != '\0' && tableTrace[0] != '0') {
                        LOG_INFO(
                            "TABLE_ROUTE_TRACE pipeline page={} bbox=[{},{},{},{}] crop={}x{}",
                            item.pageIndex, roi.x, roi.y, roi.x + roi.width, roi.y + roi.height,
                            item.crop.cols, item.crop.rows);
                    }
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
                        if (npuResult.tableResult.type == TableType::WIRELESS
                            && wirelessRecognizer_) {
                            // Defer OCR + wireless recovery; don't fallback yet.
                            npuResult.needsWirelessBackend = true;
                            npuResult.hasTableResult = false;
                            tableNpuResults.push_back(std::move(npuResult));
                            continue;
                        }
                        npuResult.hasFallback = true;
                        npuResult.fallbackReason =
                            (npuResult.tableResult.type == TableType::WIRELESS)
                                ? "wireless_table"
                                : "table_model_unavailable";
                        tableNpuResults.push_back(std::move(npuResult));
                        continue;
                    }

                    if (npuResult.tableResult.cells.empty()) {
                        // wired decode but no cells → try SLANet+ second-pass.
                        if (wirelessRecognizer_) {
                            npuResult.needsWirelessBackend = true;
                            npuResult.hasTableResult = false;
                            tableNpuResults.push_back(std::move(npuResult));
                            continue;
                        }
                        npuResult.hasFallback = true;
                        npuResult.fallbackReason = "no_cell_table";
                        tableNpuResults.push_back(std::move(npuResult));
                        continue;
                    }
                } else {
                    npuResult.npuStage = recognizeTableNpuStage(item.crop);
                    if (!npuResult.npuStage.supported) {
                        if (npuResult.npuStage.type == TableType::WIRELESS
                            && wirelessRecognizer_) {
                            // Wireless routing: skip wired NPU decode, defer
                            // for SLANet+ second-pass after OCR.
                            npuResult.needsWirelessBackend = true;
                            tableNpuResults.push_back(std::move(npuResult));
                            continue;
                        }
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
                if (npuResult.hasFallback
                    || npuResult.hasTableResult
                    || npuResult.needsWirelessBackend) {
                    continue;
                }

                npuResult.tableResult =
                    finalizeTableRecognizePostprocess(tableWorkItems[i].crop, npuResult.npuStage);
                npuResult.hasTableResult = true;

                if (!npuResult.tableResult.supported) {
                    if (npuResult.tableResult.type == TableType::WIRELESS
                        && wirelessRecognizer_) {
                        npuResult.needsWirelessBackend = true;
                        npuResult.hasTableResult = false;
                        continue;
                    }
                    npuResult.hasFallback = true;
                    npuResult.fallbackReason =
                        (npuResult.tableResult.type == TableType::WIRELESS)
                            ? "wireless_table"
                            : "table_model_unavailable";
                    continue;
                }

                if (npuResult.tableResult.cells.empty()) {
                    // wired backend produced no cells — try SLANet+ as a
                    // second pass (covers weakly bordered tables that UNET
                    // misses, e.g. 表格1 page1-6 case).
                    if (wirelessRecognizer_) {
                        npuResult.needsWirelessBackend = true;
                        npuResult.hasTableResult = false;
                        continue;
                    }
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
                    // OCR runs for both wired-decoded tables (to fill cell
                    // text via matchTableOcrToCells) AND wireless candidates
                    // (SLANet+ matcher consumes OCR boxes to emit HTML with
                    // text).
                    const bool needsOcr =
                        (!npuResult.hasFallback && npuResult.hasTableResult)
                        || npuResult.needsWirelessBackend;
                    if (!needsOcr) {
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
            for (size_t ti = 0; ti < tableNpuResults.size(); ++ti) {
                auto& npuResult = tableNpuResults[ti];
                // Wireless second-pass: run SLANet+ with the OCR boxes we
                // already collected for this crop. If it produces HTML, we
                // promote it to the primary table result; otherwise we fall
                // back to the unsupported-table placeholder.
                if (npuResult.needsWirelessBackend && wirelessRecognizer_) {
                    TableResult wr = wirelessRecognizer_->recognize(
                        tableWorkItems[ti].crop,
                        toWirelessOcrBoxes(npuResult.ocrBoxes));
                    if (wr.supported && !wr.html.empty()) {
                        npuResult.tableResult = std::move(wr);
                        npuResult.tableResult.type = TableType::WIRELESS;
                        npuResult.hasTableResult = true;
                        npuResult.hasFallback = false;
                    } else {
                        npuResult.hasFallback = true;
                        npuResult.fallbackReason = "wireless_table";
                    }
                    npuResult.needsWirelessBackend = false;
                }

                if (npuResult.hasFallback) {
                    tableElements.push_back(makeTableFallbackElement(
                        npuResult.box, npuResult.pageIndex, npuResult.fallbackReason));
                    continue;
                }

                // Only the wired path uses cell-grid OCR matching; the
                // wireless backend already consumed OCR boxes to emit HTML.
                if (npuResult.tableResult.html.empty()) {
                    matchTableOcrToCells(npuResult.tableResult, npuResult.ocrBoxes);
                }

                ContentElement elem;
                elem.type = ContentElement::Type::TABLE;
                elem.layoutBox = npuResult.box;
                elem.pageIndex = npuResult.pageIndex;

                try {
                    elem.html = !npuResult.tableResult.html.empty()
                                    ? npuResult.tableResult.html
                                    : generateTableHtml(npuResult.tableResult.cells);
                } catch (const std::exception& ex) {
                    LOG_WARN("Illegal table structure at page {}: {}",
                             npuResult.pageIndex, ex.what());
                    // Wired HTML assembly threw — try SLANet+ as a last
                    // resort. This fires for genuinely degenerate UNET outputs
                    // (rowspan overflow, etc). Keeps fallback counts at 0
                    // when the wireless backend can recover.
                    if (wirelessRecognizer_) {
                        TableResult wr = wirelessRecognizer_->recognize(
                            tableWorkItems[ti].crop,
                            toWirelessOcrBoxes(npuResult.ocrBoxes));
                        if (wr.supported && !wr.html.empty()) {
                            elem.html = wr.html;
                        }
                    }
                    if (elem.html.empty()) {
                        tableElements.push_back(makeTableFallbackElement(
                            npuResult.box, npuResult.pageIndex, "illegal_table_structure"));
                        continue;
                    }
                }

                if (elem.html.empty()) {
                    // Retry wireless second-pass if the wired HTML is empty.
                    if (wirelessRecognizer_) {
                        TableResult wr = wirelessRecognizer_->recognize(
                            tableWorkItems[ti].crop,
                            toWirelessOcrBoxes(npuResult.ocrBoxes));
                        if (wr.supported && !wr.html.empty()) {
                            elem.html = wr.html;
                        }
                    }
                    if (elem.html.empty()) {
                        tableElements.push_back(makeTableFallbackElement(
                            npuResult.box, npuResult.pageIndex, "empty_table_html"));
                        continue;
                    }
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
    const bool formulaTraceEnabled = envFlagEnabled("RAPIDDOC_FORMULA_TRACE");
    const bool filterContainedEnabled = envFlagEnabled("RAPIDDOC_FORMULA_FILTER_CONTAINED");
    const bool filterFigureContextEnabled =
        envFlagEnabled("RAPIDDOC_FORMULA_FILTER_FIGURE_CONTEXT");
    if (!formulaTraceEnabled) {
        result.formulaTraceJson.clear();
    }
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
    json formulaTrace;
    if (formulaTraceEnabled) {
        formulaTrace = json{
            {"layout_raw_boxes", json::array()},
            {"layout_prefilter_boxes", json::array()},
            {"layout_boxes", json::array()},
            {"raw_candidates", json::array()},
            {"candidate_refinements", json::array()},
            {"filtered_candidates", json::array()},
            {"crop_mappings", json::array()},
            {"crop_outputs", json::array()},
            {"batch_profile", json::object()},
            {"timing_bill", json::object()},
            {"summary", json::object()},
        };
    }

    const auto regionCollectStart = std::chrono::steady_clock::now();
    int totalEquationRawCount = 0;
    int totalEquationRefinedCount = 0;
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
        if (formulaTraceEnabled) {
            for (size_t rawIdx = 0; rawIdx < pageResult.layoutResult.rawDebugBoxes.size(); ++rawIdx) {
                const auto& debug = pageResult.layoutResult.rawDebugBoxes[rawIdx];
                json rawLayoutBox = layoutBoxTraceJson(debug.box);
                rawLayoutBox["page"] = pageResult.pageIndex;
                rawLayoutBox["layout_raw_index"] = static_cast<int>(rawIdx);
                rawLayoutBox["layout_raw_source_index"] = debug.rawIndex;
                rawLayoutBox["confidence_threshold"] = debug.confidenceThreshold;
                rawLayoutBox["passed_confidence_threshold"] =
                    debug.box.confidence >= debug.confidenceThreshold;
                rawLayoutBox["shape_class"] = formulaCandidateShapeClass(
                    debug.box,
                    pageWidth,
                    pageHeight);
                formulaTrace["layout_raw_boxes"].push_back(std::move(rawLayoutBox));
            }
            for (size_t preIdx = 0; preIdx < pageResult.layoutResult.prefilterDebugBoxes.size(); ++preIdx) {
                const auto& debug = pageResult.layoutResult.prefilterDebugBoxes[preIdx];
                json prefilterLayoutBox = layoutBoxTraceJson(debug.box);
                prefilterLayoutBox["page"] = pageResult.pageIndex;
                prefilterLayoutBox["layout_prefilter_index"] = static_cast<int>(preIdx);
                prefilterLayoutBox["layout_raw_source_index"] = debug.rawIndex;
                prefilterLayoutBox["confidence_threshold"] = debug.confidenceThreshold;
                prefilterLayoutBox["shape_class"] = formulaCandidateShapeClass(
                    debug.box,
                    pageWidth,
                    pageHeight);
                formulaTrace["layout_prefilter_boxes"].push_back(std::move(prefilterLayoutBox));
            }
            for (size_t boxIdx = 0; boxIdx < pageResult.layoutResult.boxes.size(); ++boxIdx) {
                const auto& box = pageResult.layoutResult.boxes[boxIdx];
                json layoutBox = layoutBoxTraceJson(box);
                layoutBox["page"] = pageResult.pageIndex;
                layoutBox["layout_box_index"] = static_cast<int>(boxIdx);
                layoutBox["shape_class"] = formulaCandidateShapeClass(
                    box,
                    pageWidth,
                    pageHeight);
                formulaTrace["layout_boxes"].push_back(std::move(layoutBox));
            }
            for (size_t equationIdx = 0; equationIdx < rawEquationBoxes.size(); ++equationIdx) {
                json candidate = layoutBoxTraceJson(rawEquationBoxes[equationIdx]);
                candidate["page"] = pageResult.pageIndex;
                candidate["raw_candidate_index"] = static_cast<int>(equationIdx);
                candidate["shape_class"] = formulaCandidateShapeClass(
                    rawEquationBoxes[equationIdx],
                    pageWidth,
                    pageHeight);
                formulaTrace["raw_candidates"].push_back(std::move(candidate));
            }
        }
        const auto refinedEquationBoxes = refineFormulaCandidatesConservative(
            rawEquationBoxes,
            pageResult.pageIndex,
            pageWidth,
            pageHeight,
            formulaTraceEnabled ? &formulaTrace : nullptr);
        totalEquationRefinedCount += static_cast<int>(refinedEquationBoxes.size());

        std::vector<cv::Rect> figureRois;
        std::vector<cv::Rect> figureCaptionRois;
        std::vector<cv::Rect> figureTextContextRois;
        if (filterFigureContextEnabled) {
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
        }

        std::vector<bool> containedMask(refinedEquationBoxes.size(), false);
        if (filterContainedEnabled) {
            containedMask = buildContainedEquationMask(
                refinedEquationBoxes,
                pageWidth,
                pageHeight);
        }

        std::vector<LayoutBox> filteredEquationBoxes;
        filteredEquationBoxes.reserve(refinedEquationBoxes.size());
        int droppedOnPage = 0;
        int droppedContainedOnPage = 0;
        int droppedFigureCtxOnPage = 0;
        for (size_t equationIdx = 0; equationIdx < refinedEquationBoxes.size(); ++equationIdx) {
            const auto& equationBox = refinedEquationBoxes[equationIdx];
            if (formulaTraceEnabled) {
                json candidate = layoutBoxTraceJson(equationBox);
                candidate["page"] = pageResult.pageIndex;
                candidate["refined_candidate_index"] = static_cast<int>(equationIdx);
                candidate["filtered_candidate_index"] = -1;
                candidate["shape_class"] = formulaCandidateShapeClass(
                    equationBox,
                    pageWidth,
                    pageHeight);
                candidate["kept"] = false;
                candidate["drop_reason"] = "unknown";

                if (filterContainedEnabled &&
                    equationIdx < containedMask.size() &&
                    containedMask[equationIdx]) {
                    candidate["drop_reason"] = "contained";
                    formulaTrace["filtered_candidates"].push_back(std::move(candidate));
                    ++droppedOnPage;
                    ++droppedContainedOnPage;
                    continue;
                }
                if (filterFigureContextEnabled && shouldFilterFormulaCandidateConservative(
                        equationBox,
                        figureRois,
                        figureCaptionRois,
                        figureTextContextRois,
                        pageWidth,
                        pageHeight)) {
                    candidate["drop_reason"] = "figure_context";
                    formulaTrace["filtered_candidates"].push_back(std::move(candidate));
                    ++droppedOnPage;
                    ++droppedFigureCtxOnPage;
                    continue;
                }
                candidate["kept"] = true;
                candidate["drop_reason"] = "";
                candidate["filtered_candidate_index"] = static_cast<int>(filteredEquationBoxes.size());
                formulaTrace["filtered_candidates"].push_back(std::move(candidate));
                filteredEquationBoxes.push_back(equationBox);
                continue;
            }
            if (filterContainedEnabled &&
                equationIdx < containedMask.size() &&
                containedMask[equationIdx]) {
                ++droppedOnPage;
                ++droppedContainedOnPage;
                continue;
            }
            if (filterFigureContextEnabled && shouldFilterFormulaCandidateConservative(
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
    if ((filterContainedEnabled || filterFigureContextEnabled) && totalEquationDropped > 0) {
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
            const bool validRoi = roi.width > 0 && roi.height > 0;
            if (formulaTraceEnabled) {
                json mapping = layoutBoxTraceJson(box);
                mapping["page"] = result.pages[pageIdx].pageIndex;
                mapping["filtered_candidate_index"] = static_cast<int>(equationIdx);
                mapping["valid_roi"] = validRoi;
                mapping["roi"] = {roi.x, roi.y, roi.x + roi.width, roi.y + roi.height};
                mapping["crop_index"] = validRoi ? static_cast<int>(allCrops.size()) : -1;
                formulaTrace["crop_mappings"].push_back(std::move(mapping));
            }
            if (!validRoi) {
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

    const char* dumpCropsDirRaw = std::getenv("RAPIDDOC_FORMULA_DUMP_CROPS_DIR");
    if (dumpCropsDirRaw != nullptr && dumpCropsDirRaw[0] != '\0' && !allCrops.empty()) {
        const fs::path dumpRoot(dumpCropsDirRaw);
        std::error_code ec;
        fs::create_directories(dumpRoot, ec);
        if (ec) {
            LOG_WARN(
                "RAPIDDOC_FORMULA_DUMP_CROPS_DIR: failed to create {}: {}",
                dumpRoot.string(),
                ec.message());
        } else {
            for (size_t i = 0; i < allCrops.size(); ++i) {
                std::ostringstream name;
                name << "cpp_infer_order_" << std::setw(3) << std::setfill('0') << i << ".png";
                const fs::path outPath = dumpRoot / name.str();
                if (!cv::imwrite(outPath.string(), allCrops[i])) {
                    LOG_WARN("Formula crop dump failed: {}", outPath.string());
                }
            }
            LOG_INFO("Dumped {} formula crops to {}", allCrops.size(), dumpRoot.string());
        }
    }

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
            FormulaRecognizer::BatchRecord hookBatch;
            hookBatch.batchSize = static_cast<int>(allCrops.size());
            hookBatch.inferMs = batchTimingPrimary.inferMs;
            hookBatch.cropIndices.reserve(allCrops.size());
            for (size_t i = 0; i < allCrops.size(); ++i) {
                hookBatch.cropIndices.push_back(static_cast<int>(i));
            }
            batchTimingPrimary.batches.push_back(std::move(hookBatch));
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
        const std::string latex = i < latexes.size() ? latexes[i] : std::string();
        if (i < latexes.size()) {
            latexByPage[ref.pageResultIndex][ref.equationIndex] = latex;
        }
        if (formulaTraceEnabled) {
            const int pageIndex = static_cast<int>(result.pages[ref.pageResultIndex].pageIndex);
            const LayoutBox& box = equationBoxesByPage[ref.pageResultIndex][ref.equationIndex];
            json output = layoutBoxTraceJson(box);
            output["crop_index"] = static_cast<int>(i);
            output["page"] = pageIndex;
            output["filtered_candidate_index"] = static_cast<int>(ref.equationIndex);
            output["latex"] = latex;
            output["has_latex"] = !latex.empty();
            formulaTrace["crop_outputs"].push_back(std::move(output));
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
    if (formulaTraceEnabled) {
        formulaTrace["timing_bill"] = json{
            {"region_collect_ms", result.formulaTimingBill.regionCollectMs},
            {"crop_prepare_ms", result.formulaTimingBill.cropPrepareMs},
            {"infer_ms", result.formulaTimingBill.inferMs},
            {"decode_ms", result.formulaTimingBill.decodeMs},
            {"normalize_ms", result.formulaTimingBill.normalizeMs},
            {"writeback_ms", result.formulaTimingBill.writebackMs},
            {"total_ms", result.formulaTimingBill.totalMs},
            {"crop_count", result.formulaTimingBill.cropCount},
            {"batch_count", result.formulaTimingBill.batchCount},
        };
        formulaTrace["batch_profile"] = formulaBatchProfileJson(
            batchTimingPrimary,
            batchTimingSecondary,
            formulaMaxBatchSize_);
        formulaTrace["summary"] = json{
            {"raw_candidate_count", totalEquationRawCount},
            {"refined_candidate_count", totalEquationRefinedCount},
            {"kept_candidate_count",
             std::accumulate(equationCountByPage.begin(), equationCountByPage.end(), 0)},
            {"crop_count", static_cast<int>(allCrops.size())},
            {"dropped_candidate_count", totalEquationDropped},
            {"dropped_contained_count", totalEquationDroppedContained},
            {"dropped_figure_context_count", totalEquationDroppedFigureContext},
            {"filter_contained_enabled", filterContainedEnabled},
            {"filter_figure_context_enabled", filterFigureContextEnabled},
            {"has_dual_session", formulaDualSessionEnabled_ &&
                                     formulaRecognizerSecondary_ &&
                                     formulaRecognizerSecondary_->isInitialized()},
        };
        result.formulaTraceJson = formulaTrace.dump(2);
    }

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

        // Env-gated routing dump: persist the table crop before recognition.
        const char* tableDumpDir = std::getenv("RAPIDDOC_TABLE_DUMP_CROPS_DIR");
        if (tableDumpDir != nullptr && tableDumpDir[0] != '\0') {
            const fs::path dumpRoot(tableDumpDir);
            std::error_code ec;
            fs::create_directories(dumpRoot, ec);
            if (!ec) {
                std::ostringstream name;
                name << "table_p" << std::setw(3) << std::setfill('0') << pageIndex
                     << "_" << roi.x << "_" << roi.y
                     << "_" << (roi.x + roi.width) << "_" << (roi.y + roi.height)
                     << ".png";
                const fs::path outPath = dumpRoot / name.str();
                if (!cv::imwrite(outPath.string(), tableCrop)) {
                    LOG_WARN("TABLE_ROUTE_TRACE imwrite failed: {}", outPath.string());
                }
            } else {
                LOG_WARN("TABLE_ROUTE_TRACE mkdir {} failed: {}", dumpRoot.string(), ec.message());
            }
        }

        TableResult tableResult = recognizeTable(tableCrop);

        const char* tableTrace = std::getenv("RAPIDDOC_TABLE_TRACE");
        if (tableTrace != nullptr && tableTrace[0] != '\0' && tableTrace[0] != '0') {
            LOG_INFO(
                "TABLE_ROUTE_TRACE pipeline page={} bbox=[{},{},{},{}] crop={}x{} "
                "type={} supported={}",
                pageIndex, roi.x, roi.y, roi.x + roi.width, roi.y + roi.height,
                tableCrop.cols, tableCrop.rows,
                (tableResult.type == TableType::WIRED) ? "WIRED"
                : (tableResult.type == TableType::WIRELESS) ? "WIRELESS"
                                                            : "UNKNOWN",
                tableResult.supported ? "true" : "false");
        }

        // Decide whether to try the wireless (SLANet+) backend. Three cases:
        //   - classifier routed to WIRELESS → wired path set supported=false;
        //   - wired backend succeeded but produced zero cells;
        //   - wired path failed for other reasons (no recovery — we keep the
        //     legacy unsupported placeholder to preserve behavior).
        const bool classifiedWireless =
            (tableResult.type == TableType::WIRELESS) && !tableResult.supported;
        const bool wiredNoCells =
            tableResult.supported && tableResult.cells.empty();
        const bool needsWireless =
            (classifiedWireless || wiredNoCells) && wirelessRecognizer_;

        // We collect OCR once on the crop: it's reused by either matcher path
        // (wired cell grid or SLANet+ HTML emitter).
        std::vector<ocr::PipelineOCRResult> ocrBoxes;
        if (ctx.stages.enableOcr && (ocrPipeline_ || (ocrSubmitHook_ && ocrFetchHook_))) {
            const int64_t ocrTaskId = allocateOcrTaskId();
            if (submitOcrTask(tableCrop.clone(), ocrTaskId)) {
                bool ok = false;
                if (!waitForOcrResult(ocrTaskId, ocrBoxes, ok) || !ok) {
                    ocrBoxes.clear();
                }
            }
        }

        if (needsWireless) {
            TableResult wr = wirelessRecognizer_->recognize(
                tableCrop, toWirelessOcrBoxes(ocrBoxes));
            if (wr.supported && !wr.html.empty()) {
                tableResult = std::move(wr);
                tableResult.type = TableType::WIRELESS;
            }
        }

        if (!tableResult.supported && tableResult.html.empty()) {
            const std::string reason = (tableResult.type == TableType::WIRELESS)
                                           ? "wireless_table"
                                           : "table_model_unavailable";
            elements.push_back(makeTableFallbackElement(box, pageIndex, reason));
            continue;
        }

        if (tableResult.cells.empty() && tableResult.html.empty()) {
            elements.push_back(makeTableFallbackElement(box, pageIndex, "no_cell_table"));
            continue;
        }

        // Wired cell-grid OCR matching only when SLANet+ did not already
        // emit HTML (the wireless matcher consumed OCR boxes directly).
        if (tableResult.html.empty() && !ocrBoxes.empty()) {
            matchTableOcrToCells(tableResult, ocrBoxes);
        }

        try {
            elem.html = !tableResult.html.empty()
                            ? tableResult.html
                            : generateTableHtml(tableResult.cells);
        } catch (const std::exception& ex) {
            LOG_WARN("Illegal table structure at page {}: {}", pageIndex, ex.what());
            if (wirelessRecognizer_) {
                TableResult wr = wirelessRecognizer_->recognize(
                    tableCrop, toWirelessOcrBoxes(ocrBoxes));
                if (wr.supported && !wr.html.empty()) {
                    elem.html = wr.html;
                }
            }
            if (elem.html.empty()) {
                elements.push_back(makeTableFallbackElement(
                    box, pageIndex, "illegal_table_structure"));
                continue;
            }
        }

        if (elem.html.empty()) {
            if (wirelessRecognizer_) {
                TableResult wr = wirelessRecognizer_->recognize(
                    tableCrop, toWirelessOcrBoxes(ocrBoxes));
                if (wr.supported && !wr.html.empty()) {
                    elem.html = wr.html;
                }
            }
            if (elem.html.empty()) {
                elements.push_back(makeTableFallbackElement(
                    box, pageIndex, "empty_table_html"));
                continue;
            }
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
