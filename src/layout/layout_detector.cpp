/**
 * @file layout_detector.cpp
 * @brief Layout detection implementation (DEEPX NPU + ONNX RT post-processing)
 *
 * Aligned with Python:
 *   RapidDoc/rapid_doc/model/layout/rapid_layout_self/inference_engine/dxengine.py
 *   RapidDoc/rapid_doc/model/layout/rapid_layout_self/model_handler/pp_doclayout/pre_process.py
 *   RapidDoc/rapid_doc/model/layout/rapid_layout_self/model_handler/pp_doclayout/post_process.py
 */

#include "layout/layout_detector.h"
#include "common/logger.h"

#include <dxrt/inference_engine.h>

#ifdef HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

namespace rapid_doc {

// ---------------------------------------------------------------------------
// DXEngine label list (matches Python PPDocLayoutModelHandler for DXENGINE)
// ---------------------------------------------------------------------------
static const std::vector<std::string> kDxEngineLabels = {
    "paragraph_title", "image", "text", "number", "abstract", "content",
    "figure_title", "formula", "table", "table_title", "reference",
    "doc_title", "footnote", "header", "algorithm", "footer", "seal",
    "chart_title", "chart", "formula_number", "header_image",
    "footer_image", "aside_text"
};

// Per-category confidence thresholds (Python DXENGINE branch)
static const std::unordered_map<int, float> kPerCategoryConfThres = {
    {0,  0.3f},  // paragraph_title
    {1,  0.5f},  // image
    {2,  0.4f},  // text
    {3,  0.5f},  // number
    {4,  0.5f},  // abstract
    {5,  0.5f},  // content
    {6,  0.5f},  // figure_title
    {7,  0.3f},  // formula
    {8,  0.5f},  // table
    {9,  0.5f},  // table_title
    {10, 0.5f},  // reference
    {11, 0.5f},  // doc_title
    {12, 0.5f},  // footnote
    {13, 0.5f},  // header
    {14, 0.5f},  // algorithm
    {15, 0.5f},  // footer
    {16, 0.45f}, // seal
    {17, 0.5f},  // chart_title
    {18, 0.5f},  // chart
    {19, 0.5f},  // formula_number
    {20, 0.5f},  // header_image
    {21, 0.5f},  // footer_image
    {22, 0.5f},  // aside_text
};

constexpr int kFormulaClsId = 7;
constexpr float kFormulaCompactRescueThreshold = 0.25f;
constexpr float kFormulaCompactRescueMaxWidth = 80.0f;
constexpr float kFormulaCompactRescueMaxHeight = 40.0f;
constexpr float kFormulaMediumMinWidth = 95.0f;
constexpr float kFormulaMediumMaxWidth = 140.0f;
constexpr float kFormulaMediumMaxHeight = 40.0f;
constexpr float kFormulaIsolatedThreshold = 0.60f;
constexpr float kFormulaNarrowRescueThreshold = 0.10f;
constexpr float kFormulaNarrowRescueMaxWidth = 35.0f;
constexpr float kFormulaNarrowRescueMinHeight = 28.0f;
constexpr float kFormulaNarrowRescueMaxHeight = 40.0f;
constexpr float kFormulaNarrowSupportIou = 0.5f;
constexpr float kFormulaPartnerMinConfidence = 0.25f;
constexpr float kFormulaPartnerMinWidth = 45.0f;
constexpr float kFormulaPartnerMaxWidth = 70.0f;
constexpr float kFormulaPartnerIouMin = 0.35f;
constexpr float kFormulaPartnerIouMax = 0.70f;
constexpr float kFormulaPartnerAlignTolerance = 12.0f;
constexpr float kFormulaPartnerExtensionMin = 45.0f;

// Round-1 surgical rescues (profile-specific, not globals).
// Wide-extension rescue: a wide formula raw that right-extends a passed formula raw on the same row.
constexpr float kFormulaWideExtensionMinWidth = 200.0f;
constexpr float kFormulaWideExtensionMaxWidth = 400.0f;
constexpr float kFormulaWideExtensionMinHeight = 25.0f;
constexpr float kFormulaWideExtensionMaxHeight = 45.0f;
constexpr float kFormulaWideExtensionScoreFloor = 0.18f;
constexpr float kFormulaWideExtensionSeedMinScore = 0.50f;
constexpr float kFormulaWideExtensionLeftTolerance = 6.0f;
constexpr float kFormulaWideExtensionMinRightExtensionPx = 40.0f;
constexpr float kFormulaWideExtensionRowYTolerance = 4.0f;

// Narrow-sibling rescue: a narrow formula raw that is the mirror of an already-rescued narrow sibling.
constexpr float kFormulaNarrowSiblingMinWidth = 28.0f;
constexpr float kFormulaNarrowSiblingMaxWidth = 40.0f;
constexpr float kFormulaNarrowSiblingMinHeight = 30.0f;
constexpr float kFormulaNarrowSiblingMaxHeight = 40.0f;
constexpr float kFormulaNarrowSiblingScoreFloor = 0.03f;
constexpr float kFormulaNarrowSiblingMinIou = 0.50f;
constexpr float kFormulaNarrowSiblingPartnerScore = 0.10f;

// Round-2 surgical NMS guard: container-over-contained suppressor.
// When a formula raw A tightly contains a narrower formula raw B on the same row,
// with small score delta, prefer B (the inline-sized candidate) over A.
constexpr float kFormulaContainerOverInlineMinIou = 0.60f;
constexpr float kFormulaContainerOverInlineMaxWidthRatio = 0.85f;
constexpr float kFormulaContainerOverInlineMaxScoreDelta = 0.10f;
constexpr float kFormulaContainerOverInlineEdgeTolerance = 6.0f;
constexpr float kFormulaContainerOverInlineInnerMinScore = 0.30f;

// Map DXEngine label string -> LayoutCategory enum
static LayoutCategory labelToCategory(const std::string& label) {
    if (label == "text" || label == "content" || label == "reference" ||
        label == "footnote" || label == "number" || label == "abstract" ||
        label == "aside_text")
        return LayoutCategory::TEXT;
    if (label == "paragraph_title" || label == "doc_title" || label == "chart_title")
        return LayoutCategory::TITLE;
    if (label == "image" || label == "header_image" || label == "footer_image" || label == "chart")
        return LayoutCategory::FIGURE;
    if (label == "figure_title")
        return LayoutCategory::FIGURE_CAPTION;
    if (label == "table")
        return LayoutCategory::TABLE;
    if (label == "table_title")
        return LayoutCategory::TABLE_CAPTION;
    if (label == "header")
        return LayoutCategory::HEADER;
    if (label == "footer")
        return LayoutCategory::FOOTER;
    if (label == "formula" || label == "formula_number")
        return LayoutCategory::EQUATION;
    if (label == "algorithm" || label == "seal")
        return LayoutCategory::STAMP;
    if (label == "code")
        return LayoutCategory::CODE;
    return LayoutCategory::UNKNOWN;
}

static LayoutDebugBox makeLayoutDebugBox(
    const float* row,
    int rawIndex,
    float threshold,
    const cv::Size& imShape)
{
    const int clsId = static_cast<int>(row[0]);
    const float xmin = std::max(0.0f, row[2]);
    const float ymin = std::max(0.0f, row[3]);
    const float xmax = std::min(static_cast<float>(imShape.width), row[4]);
    const float ymax = std::min(static_cast<float>(imShape.height), row[5]);
    std::string label = (clsId >= 0 && clsId < static_cast<int>(kDxEngineLabels.size()))
                        ? kDxEngineLabels[clsId]
                        : "unknown";

    LayoutDebugBox debug;
    debug.rawIndex = rawIndex;
    debug.confidenceThreshold = threshold;
    debug.box.x0 = xmin;
    debug.box.y0 = ymin;
    debug.box.x1 = xmax;
    debug.box.y1 = ymax;
    debug.box.category = labelToCategory(label);
    debug.box.confidence = row[1];
    debug.box.index = rawIndex;
    debug.box.clsId = clsId;
    debug.box.label = label;
    return debug;
}

static bool isCompactFormulaRescueBox(const float* row) {
    const float width = row[4] - row[2];
    const float height = row[5] - row[3];
    return width > 0.0f &&
           height > 0.0f &&
           width <= kFormulaCompactRescueMaxWidth &&
           height <= kFormulaCompactRescueMaxHeight;
}

static bool isMediumFormulaBox(const float* row) {
    const float width = row[4] - row[2];
    const float height = row[5] - row[3];
    return width >= kFormulaMediumMinWidth &&
           width <= kFormulaMediumMaxWidth &&
           height > 0.0f &&
           height <= kFormulaMediumMaxHeight;
}

static bool isMediumFormulaBox(const std::vector<float>& box) {
    if (box.size() < 6) {
        return false;
    }
    const float width = box[4] - box[2];
    const float height = box[5] - box[3];
    return width >= kFormulaMediumMinWidth &&
           width <= kFormulaMediumMaxWidth &&
           height > 0.0f &&
           height <= kFormulaMediumMaxHeight;
}

static bool isNarrowFormulaRescueBox(const float* row) {
    const float width = row[4] - row[2];
    const float height = row[5] - row[3];
    return width > 0.0f &&
           height >= kFormulaNarrowRescueMinHeight &&
           width <= kFormulaNarrowRescueMaxWidth &&
           height <= kFormulaNarrowRescueMaxHeight;
}

static float computeIoU(const float* a, const float* b);

static bool hasFormulaSupportingNeighbor(
    const float* row,
    int rowIndex,
    const float* boxData,
    int totalBoxes);

static bool hasQualifiedRightFormulaPartner(
    const float* row,
    int rowIndex,
    const float* boxData,
    int totalBoxes);

static bool hasQualifiedLeftFormulaPartner(
    const float* row,
    int rowIndex,
    const float* boxData,
    int totalBoxes);

static bool isQualifiedRightFormulaPartner(
    const float* container,
    const float* fragment);

static bool isQualifiedLeftFormulaPartner(
    const float* container,
    const float* fragment);

static bool isWideExtensionRescueBox(const float* row) {
    const float width = row[4] - row[2];
    const float height = row[5] - row[3];
    return width >= kFormulaWideExtensionMinWidth &&
           width <= kFormulaWideExtensionMaxWidth &&
           height >= kFormulaWideExtensionMinHeight &&
           height <= kFormulaWideExtensionMaxHeight;
}

static bool hasPassedFormulaSeedExtendedOnRight(
    const float* row,
    int rowIndex,
    const float* boxData,
    int totalBoxes)
{
    const float rowX0 = row[2];
    const float rowY0 = row[3];
    const float rowX1 = row[4];
    const float rowY1 = row[5];
    for (int i = 0; i < totalBoxes; ++i) {
        if (i == rowIndex) continue;
        const float* other = boxData + i * 6;
        if (static_cast<int>(other[0]) != kFormulaClsId) continue;
        if (other[1] < kFormulaWideExtensionSeedMinScore) continue;
        const float seedX0 = other[2];
        const float seedY0 = other[3];
        const float seedX1 = other[4];
        const float seedY1 = other[5];
        if (std::fabs(seedX0 - rowX0) > kFormulaWideExtensionLeftTolerance) continue;
        if (std::fabs(seedY0 - rowY0) > kFormulaWideExtensionRowYTolerance) continue;
        if (std::fabs(seedY1 - rowY1) > kFormulaWideExtensionRowYTolerance) continue;
        if (seedX1 >= rowX1) continue;
        if ((rowX1 - seedX1) < kFormulaWideExtensionMinRightExtensionPx) continue;
        return true;
    }
    return false;
}

static bool isNarrowSiblingRescueBox(const float* row) {
    const float width = row[4] - row[2];
    const float height = row[5] - row[3];
    return width >= kFormulaNarrowSiblingMinWidth &&
           width <= kFormulaNarrowSiblingMaxWidth &&
           height >= kFormulaNarrowSiblingMinHeight &&
           height <= kFormulaNarrowSiblingMaxHeight;
}

static bool hasPassedNarrowFormulaSibling(
    const float* row,
    int rowIndex,
    const float* boxData,
    int totalBoxes)
{
    for (int i = 0; i < totalBoxes; ++i) {
        if (i == rowIndex) continue;
        const float* other = boxData + i * 6;
        if (static_cast<int>(other[0]) != kFormulaClsId) continue;
        if (other[1] < kFormulaNarrowSiblingPartnerScore) continue;
        if (!isNarrowFormulaRescueBox(other)) continue;
        if (computeIoU(row + 2, other + 2) < kFormulaNarrowSiblingMinIou) continue;
        return true;
    }
    return false;
}

static float effectiveConfidenceThreshold(
    const float* row,
    int rowIndex,
    const float* boxData,
    int totalBoxes)
{
    const int clsId = static_cast<int>(row[0]);
    const auto it = kPerCategoryConfThres.find(clsId);
    float threshold = (it != kPerCategoryConfThres.end()) ? it->second : 0.5f;
    if (clsId != kFormulaClsId) {
        return threshold;
    }
    if (isMediumFormulaBox(row)) {
        const bool hasRightPartner =
            hasQualifiedRightFormulaPartner(row, rowIndex, boxData, totalBoxes);
        const bool hasLeftPartner =
            hasQualifiedLeftFormulaPartner(row, rowIndex, boxData, totalBoxes);
        if (hasRightPartner && !hasLeftPartner) {
            threshold = std::min(threshold, kFormulaCompactRescueThreshold);
        } else if (!hasRightPartner && !hasLeftPartner) {
            threshold = std::max(threshold, kFormulaIsolatedThreshold);
        }
    }
    if (isCompactFormulaRescueBox(row)) {
        threshold = std::min(threshold, kFormulaCompactRescueThreshold);
    }
    if (row[1] >= kFormulaNarrowRescueThreshold &&
        isNarrowFormulaRescueBox(row) &&
        hasFormulaSupportingNeighbor(row, rowIndex, boxData, totalBoxes)) {
        threshold = std::min(threshold, kFormulaNarrowRescueThreshold);
    }
    if (row[1] >= kFormulaWideExtensionScoreFloor &&
        isWideExtensionRescueBox(row) &&
        hasPassedFormulaSeedExtendedOnRight(row, rowIndex, boxData, totalBoxes)) {
        threshold = std::min(threshold, kFormulaWideExtensionScoreFloor);
    }
    if (row[1] >= kFormulaNarrowSiblingScoreFloor &&
        isNarrowSiblingRescueBox(row) &&
        hasPassedNarrowFormulaSibling(row, rowIndex, boxData, totalBoxes)) {
        threshold = std::min(threshold, kFormulaNarrowSiblingScoreFloor);
    }
    return threshold;
}

// ---------------------------------------------------------------------------
// IoU helpers (matches Python post_process.py iou())
// ---------------------------------------------------------------------------
static float computeIoU(const float* a, const float* b) {
    float x1 = std::max(a[0], b[0]);
    float y1 = std::max(a[1], b[1]);
    float x2 = std::min(a[2], b[2]);
    float y2 = std::min(a[3], b[3]);
    float interArea = std::max(0.0f, x2 - x1 + 1) * std::max(0.0f, y2 - y1 + 1);
    float area1 = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
    float area2 = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
    return interArea / (area1 + area2 - interArea);
}

static bool hasFormulaSupportingNeighbor(
    const float* row,
    int rowIndex,
    const float* boxData,
    int totalBoxes)
{
    for (int i = 0; i < totalBoxes; ++i) {
        if (i == rowIndex) {
            continue;
        }
        const float* other = boxData + i * 6;
        if (static_cast<int>(other[0]) != kFormulaClsId) {
            continue;
        }
        if (computeIoU(row + 2, other + 2) >= kFormulaNarrowSupportIou) {
            return true;
        }
    }
    return false;
}

static bool isQualifiedRightFormulaPartner(
    const float* container,
    const float* fragment)
{
    if (container == nullptr || fragment == nullptr) {
        return false;
    }
    if (static_cast<int>(container[0]) != kFormulaClsId ||
        static_cast<int>(fragment[0]) != kFormulaClsId) {
        return false;
    }
    if (!isMediumFormulaBox(container)) {
        return false;
    }
    const float containerWidth = container[4] - container[2];
    const float containerHeight = container[5] - container[3];
    const float fragmentWidth = fragment[4] - fragment[2];
    const float fragmentHeight = fragment[5] - fragment[3];
    if (fragmentWidth < kFormulaPartnerMinWidth ||
        fragmentWidth > kFormulaPartnerMaxWidth ||
        containerWidth <= fragmentWidth ||
        containerHeight <= 0.0f ||
        fragmentHeight <= 0.0f ||
        std::fabs(containerHeight - fragmentHeight) > 8.0f) {
        return false;
    }
    const float overlap = computeIoU(container + 2, fragment + 2);
    if (overlap < kFormulaPartnerIouMin ||
        overlap > kFormulaPartnerIouMax) {
        return false;
    }
    const float rightGap = std::fabs(container[4] - fragment[4]);
    const float leftExtension = fragment[2] - container[2];
    return rightGap <= kFormulaPartnerAlignTolerance &&
           leftExtension >= kFormulaPartnerExtensionMin;
}

static bool isQualifiedLeftFormulaPartner(
    const float* container,
    const float* fragment)
{
    if (container == nullptr || fragment == nullptr) {
        return false;
    }
    if (static_cast<int>(container[0]) != kFormulaClsId ||
        static_cast<int>(fragment[0]) != kFormulaClsId) {
        return false;
    }
    if (!isMediumFormulaBox(container)) {
        return false;
    }
    const float containerWidth = container[4] - container[2];
    const float containerHeight = container[5] - container[3];
    const float fragmentWidth = fragment[4] - fragment[2];
    const float fragmentHeight = fragment[5] - fragment[3];
    if (fragmentWidth < kFormulaPartnerMinWidth ||
        fragmentWidth > kFormulaPartnerMaxWidth ||
        containerWidth <= fragmentWidth ||
        containerHeight <= 0.0f ||
        fragmentHeight <= 0.0f ||
        std::fabs(containerHeight - fragmentHeight) > 8.0f) {
        return false;
    }
    const float overlap = computeIoU(container + 2, fragment + 2);
    if (overlap < kFormulaPartnerIouMin ||
        overlap > kFormulaPartnerIouMax) {
        return false;
    }
    const float leftGap = std::fabs(container[2] - fragment[2]);
    const float rightExtension = container[4] - fragment[4];
    return leftGap <= kFormulaPartnerAlignTolerance &&
           rightExtension >= kFormulaPartnerExtensionMin;
}

static bool hasQualifiedRightFormulaPartner(
    const float* row,
    int rowIndex,
    const float* boxData,
    int totalBoxes)
{
    for (int i = 0; i < totalBoxes; ++i) {
        if (i == rowIndex) {
            continue;
        }
        const float* other = boxData + i * 6;
        if (other[1] < kFormulaPartnerMinConfidence) {
            continue;
        }
        if (isQualifiedRightFormulaPartner(row, other)) {
            return true;
        }
    }
    return false;
}

static bool hasQualifiedLeftFormulaPartner(
    const float* row,
    int rowIndex,
    const float* boxData,
    int totalBoxes)
{
    for (int i = 0; i < totalBoxes; ++i) {
        if (i == rowIndex) {
            continue;
        }
        const float* other = boxData + i * 6;
        if (other[1] < kFormulaPartnerMinConfidence) {
            continue;
        }
        if (isQualifiedLeftFormulaPartner(row, other)) {
            return true;
        }
    }
    return false;
}

static void suppressFormulaSuffixFragments(
    std::vector<std::vector<float>>& rawBoxes)
{
    if (rawBoxes.empty()) {
        return;
    }
    std::vector<bool> keep(rawBoxes.size(), true);
    for (size_t i = 0; i < rawBoxes.size(); ++i) {
        const auto& container = rawBoxes[i];
        if (container.size() < 6 ||
            static_cast<int>(container[0]) != kFormulaClsId ||
            container[1] < kFormulaCompactRescueThreshold ||
            !isMediumFormulaBox(container)) {
            continue;
        }
        bool hasLeftPartner = false;
        std::vector<size_t> rightPartners;
        for (size_t j = 0; j < rawBoxes.size(); ++j) {
            if (i == j || !keep[j]) {
                continue;
            }
            const auto& fragment = rawBoxes[j];
            if (fragment.size() < 6 ||
                static_cast<int>(fragment[0]) != kFormulaClsId ||
                fragment[1] < kFormulaPartnerMinConfidence) {
                continue;
            }
            if (isQualifiedLeftFormulaPartner(container.data(), fragment.data())) {
                hasLeftPartner = true;
            }
            if (isQualifiedRightFormulaPartner(container.data(), fragment.data())) {
                rightPartners.push_back(j);
            }
        }
        if (!hasLeftPartner) {
            for (size_t idx : rightPartners) {
                keep[idx] = false;
            }
        }
    }
    std::vector<std::vector<float>> filtered;
    filtered.reserve(rawBoxes.size());
    for (size_t i = 0; i < rawBoxes.size(); ++i) {
        if (keep[i]) {
            filtered.push_back(std::move(rawBoxes[i]));
        }
    }
    rawBoxes = std::move(filtered);
}

// Round-2 guard: when a formula raw A tightly contains a narrower formula raw B
// with nearly-identical score, drop A so that NMS keeps the inline-sized B.
// Pattern (py=11 case):
//   A.x0 <= B.x0, A.x1 >= B.x1, A.y0 ~ B.y0, A.y1 ~ B.y1
//   B.width / A.width < 0.85
//   A.score - B.score < 0.10
//   IoU(A, B) >= 0.60
// This rule inverts the default NMS choice (higher-score wins) for this
// specific pattern only; without it, NMS eats the inline-sized B via the
// container A that drags in extra pixels of surrounding text.
static void suppressFormulaContainersOverInline(
    std::vector<std::vector<float>>& rawBoxes)
{
    if (rawBoxes.empty()) {
        return;
    }
    std::vector<bool> keep(rawBoxes.size(), true);
    for (size_t i = 0; i < rawBoxes.size(); ++i) {
        const auto& A = rawBoxes[i];
        if (A.size() < 6) {
            continue;
        }
        if (static_cast<int>(A[0]) != kFormulaClsId) {
            continue;
        }
        const float aScore = A[1];
        const float aX0 = A[2];
        const float aY0 = A[3];
        const float aX1 = A[4];
        const float aY1 = A[5];
        const float aW = aX1 - aX0;
        if (aW <= 0.0f) {
            continue;
        }
        for (size_t j = 0; j < rawBoxes.size(); ++j) {
            if (i == j || !keep[i] || !keep[j]) {
                continue;
            }
            const auto& B = rawBoxes[j];
            if (B.size() < 6) {
                continue;
            }
            if (static_cast<int>(B[0]) != kFormulaClsId) {
                continue;
            }
            const float bScore = B[1];
            if (bScore < kFormulaContainerOverInlineInnerMinScore) {
                continue;
            }
            if (aScore <= bScore) {
                continue;
            }
            if (aScore - bScore >= kFormulaContainerOverInlineMaxScoreDelta) {
                continue;
            }
            const float bX0 = B[2];
            const float bY0 = B[3];
            const float bX1 = B[4];
            const float bY1 = B[5];
            const float bW = bX1 - bX0;
            if (bW <= 0.0f) {
                continue;
            }
            if (bX0 < aX0 - kFormulaContainerOverInlineEdgeTolerance) {
                continue;
            }
            if (bX1 > aX1 + kFormulaContainerOverInlineEdgeTolerance) {
                continue;
            }
            if (bY0 < aY0 - kFormulaContainerOverInlineEdgeTolerance) {
                continue;
            }
            if (bY1 > aY1 + kFormulaContainerOverInlineEdgeTolerance) {
                continue;
            }
            if (bW / aW >= kFormulaContainerOverInlineMaxWidthRatio) {
                continue;
            }
            if (computeIoU(A.data() + 2, B.data() + 2) <
                kFormulaContainerOverInlineMinIou) {
                continue;
            }
            keep[i] = false;
            break;
        }
    }
    std::vector<std::vector<float>> filtered;
    filtered.reserve(rawBoxes.size());
    for (size_t i = 0; i < rawBoxes.size(); ++i) {
        if (keep[i]) {
            filtered.push_back(std::move(rawBoxes[i]));
        }
    }
    rawBoxes = std::move(filtered);
}

// NMS with different IoU thresholds for same/different class (Python nms())
static std::vector<int> nms(const std::vector<std::vector<float>>& boxes,
                            float iouSame = 0.6f, float iouDiff = 0.98f) {
    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return boxes[a][1] > boxes[b][1];
    });

    std::vector<int> selected;
    while (!indices.empty()) {
        int current = indices.front();
        selected.push_back(current);
        const auto& curBox = boxes[current];
        float curClass = curBox[0];
        const float* curCoords = curBox.data() + 2;

        std::vector<int> remaining;
        for (size_t i = 1; i < indices.size(); ++i) {
            int idx = indices[i];
            const auto& box = boxes[idx];
            float threshold = (box[0] == curClass) ? iouSame : iouDiff;
            if (computeIoU(curCoords, box.data() + 2) < threshold) {
                remaining.push_back(idx);
            }
        }
        indices = std::move(remaining);
    }
    return selected;
}

// ---------------------------------------------------------------------------
// Pimpl
// ---------------------------------------------------------------------------
struct LayoutDetector::Impl {
    std::unique_ptr<dxrt::InferenceEngine> dxEngine;

#ifdef HAS_ONNXRUNTIME
    Ort::Env ortEnv{ORT_LOGGING_LEVEL_WARNING, "layout_nms"};
    std::unique_ptr<Ort::Session> ortSession;
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
#endif
};

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
LayoutDetector::LayoutDetector(const LayoutDetectorConfig& config)
    : impl_(std::make_unique<Impl>())
    , config_(config)
{
}

LayoutDetector::~LayoutDetector() = default;

// ---------------------------------------------------------------------------
// Initialization — load DX Engine + ONNX Runtime session
// ---------------------------------------------------------------------------
bool LayoutDetector::initialize() {
    LOG_INFO("Initializing Layout detector...");
    LOG_INFO("  DXNN model: {}", config_.dxnnModelPath);
    LOG_INFO("  ONNX sub-model: {}", config_.onnxSubModelPath);
    LOG_INFO("  Device ID: {}", config_.deviceId);

    try {
        if (config_.deviceId >= 0) {
            dxrt::InferenceOption option;
            option.devices.push_back(config_.deviceId);
            impl_->dxEngine = std::make_unique<dxrt::InferenceEngine>(config_.dxnnModelPath, option);
        } else {
            impl_->dxEngine = std::make_unique<dxrt::InferenceEngine>(config_.dxnnModelPath);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to load DX Engine model: {}", e.what());
        return false;
    }

#ifdef HAS_ONNXRUNTIME
    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        impl_->ortSession = std::make_unique<Ort::Session>(
            impl_->ortEnv, config_.onnxSubModelPath.c_str(), opts);
    } catch (const Ort::Exception& e) {
        LOG_ERROR("Failed to load ONNX sub-model: {}", e.what());
        return false;
    }
#else
    LOG_WARN("ONNX Runtime not available — post-processing will be skipped");
#endif

    initialized_ = true;
    LOG_INFO("Layout detector initialized successfully");
    return true;
}

// ---------------------------------------------------------------------------
// Preprocessing — matches Python PPPreProcess for DXENGINE path
//   resize to (inputSize, inputSize) with INTER_CUBIC
//   keep uint8, NHWC, no normalization
// ---------------------------------------------------------------------------
cv::Mat LayoutDetector::preprocess(const cv::Mat& image, cv::Point2f& scaleFactor) {
    int targetH = config_.inputSize;
    int targetW = config_.inputSize;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(targetW, targetH), 0, 0, cv::INTER_CUBIC);

    // scale_factor = [target_h / orig_h, target_w / orig_w]
    scaleFactor.x = static_cast<float>(targetW) / image.cols;  // w_scale
    scaleFactor.y = static_cast<float>(targetH) / image.rows;  // h_scale

    return resized;
}

// ---------------------------------------------------------------------------
// Post-processing — matches Python PPPostProcess + PPDocLayoutModelHandler
//   1. Per-category confidence filter
//   2. NMS (iou_same=0.6, iou_diff=0.98)
//   3. Large image-box filter (area_thres 0.82/0.93)
//   4. Coordinate clamping & LayoutCategory mapping
// ---------------------------------------------------------------------------
std::vector<LayoutBox> LayoutDetector::postprocess(
    const std::vector<std::vector<float>>& dxOutputs,
    const cv::Size& imShape,
    const cv::Point2f& scaleFactor,
    std::vector<LayoutDebugBox>* rawDebugBoxes,
    std::vector<LayoutDebugBox>* prefilterDebugBoxes)
{
#ifndef HAS_ONNXRUNTIME
    LOG_WARN("ONNX Runtime not available — cannot run NMS post-processing");
    return {};
#else
    if (!impl_->ortSession) return {};

    // ---- Run ONNX sub-model ----
    // dxOutputs[0] and dxOutputs[1] are the two DX Engine output tensors.
    // We build the ONNX feed dict with 4 inputs.
    int64_t inputH = config_.inputSize;
    int64_t inputW = config_.inputSize;

    // im_shape = [[inputH, inputW]]  float32
    std::vector<float> imShapeData = {static_cast<float>(inputH),
                                      static_cast<float>(inputW)};
    std::vector<int64_t> imShapeDims = {1, 2};

    // scale_factor = [[h_scale, w_scale]]
    std::vector<float> scaleData = {scaleFactor.y, scaleFactor.x};
    std::vector<int64_t> scaleDims = {1, 2};

    auto imShapeTensor = Ort::Value::CreateTensor<float>(
        impl_->memInfo, imShapeData.data(), imShapeData.size(),
        imShapeDims.data(), imShapeDims.size());
    auto scaleTensor = Ort::Value::CreateTensor<float>(
        impl_->memInfo, scaleData.data(), scaleData.size(),
        scaleDims.data(), scaleDims.size());

    // Determine DX output tensor shapes from the ONNX model input metadata
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputs = impl_->ortSession->GetInputCount();
    std::vector<const char*> inputNames;
    std::vector<Ort::AllocatedStringPtr> inputNamePtrs;
    for (size_t i = 0; i < numInputs; ++i) {
        auto namePtr = impl_->ortSession->GetInputNameAllocated(i, allocator);
        inputNames.push_back(namePtr.get());
        inputNamePtrs.push_back(std::move(namePtr));
    }

    size_t numOutputs = impl_->ortSession->GetOutputCount();
    std::vector<const char*> outputNames;
    std::vector<Ort::AllocatedStringPtr> outputNamePtrs;
    for (size_t i = 0; i < numOutputs; ++i) {
        auto namePtr = impl_->ortSession->GetOutputNameAllocated(i, allocator);
        outputNames.push_back(namePtr.get());
        outputNamePtrs.push_back(std::move(namePtr));
    }

    // Build tensors for DX outputs, matching the ONNX input names
    // We need mutable copies because CreateTensor takes non-const pointers
    std::vector<float> dxOut0 = dxOutputs[0];
    std::vector<float> dxOut1 = dxOutputs[1];

    // Get expected shapes from the ONNX model
    auto getInputShape = [&](size_t idx) -> std::vector<int64_t> {
        auto typeInfo = impl_->ortSession->GetInputTypeInfo(idx);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        return tensorInfo.GetShape();
    };

    std::vector<int64_t> shape0 = getInputShape(0);
    std::vector<int64_t> shape1 = getInputShape(1);

    // Fix dynamic dimensions (-1) based on actual data size
    auto fixDynamic = [](std::vector<int64_t>& shape, size_t dataSize) {
        int64_t known = 1;
        int dynIdx = -1;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] > 0) known *= shape[i];
            else dynIdx = static_cast<int>(i);
        }
        if (dynIdx >= 0 && known > 0) {
            shape[dynIdx] = static_cast<int64_t>(dataSize) / known;
        }
    };

    fixDynamic(shape0, dxOut0.size());
    fixDynamic(shape1, dxOut1.size());

    auto dxTensor0 = Ort::Value::CreateTensor<float>(
        impl_->memInfo, dxOut0.data(), dxOut0.size(),
        shape0.data(), shape0.size());
    auto dxTensor1 = Ort::Value::CreateTensor<float>(
        impl_->memInfo, dxOut1.data(), dxOut1.size(),
        shape1.data(), shape1.size());

    // Arrange tensors in the order expected by the ONNX model
    std::vector<Ort::Value> ortInputs;
    for (size_t i = 0; i < numInputs; ++i) {
        std::string name(inputNames[i]);
        if (name.find("concat") != std::string::npos) {
            ortInputs.push_back(std::move(dxTensor0));
        } else if (name.find("layer_norm") != std::string::npos) {
            ortInputs.push_back(std::move(dxTensor1));
        } else if (name == "im_shape") {
            ortInputs.push_back(std::move(imShapeTensor));
        } else if (name == "scale_factor") {
            ortInputs.push_back(std::move(scaleTensor));
        }
    }

    auto ortOutputs = impl_->ortSession->Run(
        Ort::RunOptions{nullptr},
        inputNames.data(), ortInputs.data(), ortInputs.size(),
        outputNames.data(), outputNames.size());

    // ---- Parse ONNX output: boxes and box_num ----
    // Output format follows PaddleDetection: pred[0] = boxes, pred[1] = box_nums
    // boxes: [N, 6]  each row = [cls_id, score, xmin, ymin, xmax, ymax]
    const float* boxData = ortOutputs[0].GetTensorData<float>();
    auto boxShape = ortOutputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int totalBoxes = static_cast<int>(boxShape[0]);

    // Collect raw boxes as vector of 6-element vectors
    std::vector<std::vector<float>> rawBoxes;
    rawBoxes.reserve(totalBoxes);
    for (int i = 0; i < totalBoxes; ++i) {
        const float* row = boxData + i * 6;
        int clsId = static_cast<int>(row[0]);
        float score = row[1];
        if (clsId < 0) continue;

        const float threshold =
            effectiveConfidenceThreshold(row, i, boxData, totalBoxes);
        if (rawDebugBoxes != nullptr) {
            LayoutDebugBox debug = makeLayoutDebugBox(row, i, threshold, imShape);
            if (debug.box.x1 > debug.box.x0 && debug.box.y1 > debug.box.y0) {
                rawDebugBoxes->push_back(std::move(debug));
            }
        }
        if (score < threshold) continue;

        if (prefilterDebugBoxes != nullptr) {
            LayoutDebugBox debug = makeLayoutDebugBox(row, i, threshold, imShape);
            if (debug.box.x1 > debug.box.x0 && debug.box.y1 > debug.box.y0) {
                prefilterDebugBoxes->push_back(std::move(debug));
            }
        }
        rawBoxes.push_back({row[0], row[1], row[2], row[3], row[4], row[5]});
    }

    suppressFormulaSuffixFragments(rawBoxes);
    suppressFormulaContainersOverInline(rawBoxes);

    // ---- NMS (same-class IoU=0.6, diff-class IoU=0.98) ----
    auto selected = nms(rawBoxes, 0.6f, 0.98f);

    // ---- Large image box filter ----
    float imgW = static_cast<float>(imShape.width);
    float imgH = static_cast<float>(imShape.height);
    float imgArea = imgW * imgH;
    float areaThreshold = (imgH > imgW) ? 0.82f : 0.93f;

    int imageClsIdx = -1;
    for (size_t i = 0; i < kDxEngineLabels.size(); ++i) {
        if (kDxEngineLabels[i] == "image") {
            imageClsIdx = static_cast<int>(i);
            break;
        }
    }

    std::vector<std::vector<float>> filteredBoxes;
    for (int idx : selected) {
        const auto& box = rawBoxes[idx];
        int clsId = static_cast<int>(box[0]);
        if (clsId == imageClsIdx && selected.size() > 1) {
            float xmin = std::max(0.0f, box[2]);
            float ymin = std::max(0.0f, box[3]);
            float xmax = std::min(imgW, box[4]);
            float ymax = std::min(imgH, box[5]);
            float boxArea = (xmax - xmin) * (ymax - ymin);
            if (boxArea > areaThreshold * imgArea) continue;
        }
        filteredBoxes.push_back(box);
    }
    if (filteredBoxes.empty() && !selected.empty()) {
        for (int idx : selected)
            filteredBoxes.push_back(rawBoxes[idx]);
    }

    // ---- Map to LayoutBox ----
    std::vector<LayoutBox> result;
    result.reserve(filteredBoxes.size());
    for (size_t i = 0; i < filteredBoxes.size(); ++i) {
        const auto& box = filteredBoxes[i];
        int clsId = static_cast<int>(box[0]);
        float score = box[1];
        float xmin = std::max(0.0f, box[2]);
        float ymin = std::max(0.0f, box[3]);
        float xmax = std::min(imgW, box[4]);
        float ymax = std::min(imgH, box[5]);
        if (xmax <= xmin || ymax <= ymin) continue;

        std::string label = (clsId >= 0 && clsId < static_cast<int>(kDxEngineLabels.size()))
                            ? kDxEngineLabels[clsId]
                            : "unknown";

        LayoutBox lb;
        lb.x0 = xmin;
        lb.y0 = ymin;
        lb.x1 = xmax;
        lb.y1 = ymax;
        lb.category = labelToCategory(label);
        lb.confidence = score;
        lb.index = static_cast<int>(i);
        lb.clsId = clsId;
        lb.label = label;
        result.push_back(lb);
    }

    return result;
#endif // HAS_ONNXRUNTIME
}

// ---------------------------------------------------------------------------
// Detect — full pipeline
// ---------------------------------------------------------------------------
LayoutResult LayoutDetector::detect(const cv::Mat& image) {
    LayoutResult result;

    if (!initialized_) {
        LOG_ERROR("Layout detector not initialized");
        return result;
    }

    auto tStart = std::chrono::steady_clock::now();

    // 1. Preprocess
    cv::Point2f scaleFactor;
    cv::Mat preprocessed = preprocess(image, scaleFactor);

    // 2. DX Engine inference — input is NHWC uint8
    // DX Engine Run() takes void* pointing to the raw data buffer.
    auto dxRawOutputs = impl_->dxEngine->Run(
        static_cast<void*>(preprocessed.data));

    // Convert raw DX outputs to float vectors
    std::vector<std::vector<float>> dxOutputs;
    for (auto& outPtr : dxRawOutputs) {
        const float* ptr = reinterpret_cast<const float*>(outPtr->data());
        size_t count = 1;
        for (auto d : outPtr->shape()) count *= d;
        dxOutputs.emplace_back(ptr, ptr + count);
    }

    // 3. Post-process (ONNX NMS + category mapping)
    cv::Size origShape(image.cols, image.rows);
    result.boxes = postprocess(
        dxOutputs,
        origShape,
        scaleFactor,
        config_.emitDebugBoxes ? &result.rawDebugBoxes : nullptr,
        config_.emitDebugBoxes ? &result.prefilterDebugBoxes : nullptr);

    auto tEnd = std::chrono::steady_clock::now();
    result.inferenceTimeMs =
        std::chrono::duration<double, std::milli>(tEnd - tStart).count();

    LOG_INFO("Layout detection: {} boxes in {:.1f}ms",
             result.boxes.size(), result.inferenceTimeMs);

    return result;
}

// ---------------------------------------------------------------------------
// Async detect — runs synchronously for now, async DX Engine integration later
// ---------------------------------------------------------------------------
void LayoutDetector::detectAsync(const cv::Mat& image, DetectionCallback callback) {
    auto result = detect(image);
    if (callback) {
        callback(result);
    }
}

} // namespace rapid_doc
