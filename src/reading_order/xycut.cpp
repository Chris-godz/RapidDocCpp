/**
 * @file xycut.cpp
 * @brief XY-Cut++ reading order algorithm implementation
 * 
 * Pure geometric algorithm — ported from Python xycut_plus.py.
 * No model inference, no external dependencies beyond std/opencv.
 */

#include "reading_order/xycut.h"
#include "common/logger.h"
#include <algorithm>
#include <numeric>
#include <cmath>

namespace rapid_doc {

TextDirection detectTextDirection(const std::vector<LayoutBox>& boxes) {
    if (boxes.empty()) return TextDirection::HORIZONTAL;

    int horizontalCount = 0;
    int totalCount = 0;

    for (const auto& box : boxes) {
        float w = box.width();
        float h = box.height();
        if (w > 0 && h > 0) {
            if (w >= h * 1.5f) {
                horizontalCount++;
            }
            totalCount++;
        }
    }

    if (totalCount == 0) return TextDirection::HORIZONTAL;

    float ratio = static_cast<float>(horizontalCount) / totalCount;
    return (ratio >= 0.5f) ? TextDirection::HORIZONTAL : TextDirection::VERTICAL;
}

namespace detail {

std::vector<int> projectionByBboxes(
    const std::vector<LayoutBox>& boxes,
    int axis,
    int size)
{
    std::vector<int> projection(size, 0);

    for (const auto& box : boxes) {
        int start, end;
        if (axis == 0) {
            // X-axis projection
            start = std::max(0, static_cast<int>(box.x0));
            end = std::min(size, static_cast<int>(box.x1));
        } else {
            // Y-axis projection
            start = std::max(0, static_cast<int>(box.y0));
            end = std::min(size, static_cast<int>(box.y1));
        }

        for (int i = start; i < end; i++) {
            projection[i]++;
        }
    }

    return projection;
}

std::vector<std::pair<int, int>> splitProjectionProfile(
    const std::vector<int>& values,
    int minValue,
    int minGap)
{
    std::vector<std::pair<int, int>> segments;

    bool inSegment = false;
    int segStart = 0;
    int gapCount = 0;

    for (int i = 0; i < static_cast<int>(values.size()); i++) {
        if (values[i] > minValue) {
            if (!inSegment) {
                segStart = i;
                inSegment = true;
            }
            gapCount = 0;
        } else {
            if (inSegment) {
                gapCount++;
                if (gapCount >= minGap) {
                    segments.emplace_back(segStart, i - gapCount + 1);
                    inSegment = false;
                    gapCount = 0;
                }
            }
        }
    }

    // Close last segment
    if (inSegment) {
        segments.emplace_back(segStart, static_cast<int>(values.size()));
    }

    return segments;
}

void recursiveXYCut(
    const std::vector<LayoutBox>& boxes,
    const std::vector<int>& indices,
    int pageWidth,
    int pageHeight,
    const XYCutConfig& config,
    std::vector<int>& result)
{
    if (indices.empty()) return;

    if (indices.size() == 1) {
        result.push_back(indices[0]);
        return;
    }

    // Gather boxes for current indices
    std::vector<LayoutBox> subBoxes;
    for (int idx : indices) {
        subBoxes.push_back(boxes[idx]);
    }

    int minGapX = std::max(1, static_cast<int>(pageWidth * config.minGapRatio));
    int minGapY = std::max(1, static_cast<int>(pageHeight * config.minGapRatio));
    int minVal = static_cast<int>(config.minValueRatio);

    // Try X-axis split first
    auto xProj = projectionByBboxes(subBoxes, 0, pageWidth);
    auto xSegments = splitProjectionProfile(xProj, minVal, minGapX);

    if (xSegments.size() > 1) {
        // Split into columns (left to right)
        for (const auto& seg : xSegments) {
            std::vector<int> group;
            for (size_t i = 0; i < indices.size(); i++) {
                float cx = subBoxes[i].center().x;
                if (cx >= seg.first && cx < seg.second) {
                    group.push_back(indices[i]);
                }
            }
            recursiveXYCut(boxes, group, pageWidth, pageHeight, config, result);
        }
        return;
    }

    // Try Y-axis split
    auto yProj = projectionByBboxes(subBoxes, 1, pageHeight);
    auto ySegments = splitProjectionProfile(yProj, minVal, minGapY);

    if (ySegments.size() > 1) {
        // Split into rows (top to bottom)
        for (const auto& seg : ySegments) {
            std::vector<int> group;
            for (size_t i = 0; i < indices.size(); i++) {
                float cy = subBoxes[i].center().y;
                if (cy >= seg.first && cy < seg.second) {
                    group.push_back(indices[i]);
                }
            }
            recursiveXYCut(boxes, group, pageWidth, pageHeight, config, result);
        }
        return;
    }

    // No split possible — add all indices sorted by position (top-to-bottom, left-to-right)
    std::vector<int> sorted = indices;
    std::sort(sorted.begin(), sorted.end(), [&boxes](int a, int b) {
        float ya = boxes[a].center().y;
        float yb = boxes[b].center().y;
        float threshold = std::min(boxes[a].height(), boxes[b].height()) * 0.5f;
        if (std::abs(ya - yb) < threshold) {
            return boxes[a].center().x < boxes[b].center().x;
        }
        return ya < yb;
    });
    result.insert(result.end(), sorted.begin(), sorted.end());
}

void recursiveYXCut(
    const std::vector<LayoutBox>& boxes,
    const std::vector<int>& indices,
    int pageWidth,
    int pageHeight,
    const XYCutConfig& config,
    std::vector<int>& result)
{
    if (indices.empty()) return;

    if (indices.size() == 1) {
        result.push_back(indices[0]);
        return;
    }

    std::vector<LayoutBox> subBoxes;
    for (int idx : indices) {
        subBoxes.push_back(boxes[idx]);
    }

    int minGapX = std::max(1, static_cast<int>(pageWidth * config.minGapRatio));
    int minGapY = std::max(1, static_cast<int>(pageHeight * config.minGapRatio));
    int minVal = static_cast<int>(config.minValueRatio);

    // Try Y-axis split first (for vertical text)
    auto yProj = projectionByBboxes(subBoxes, 1, pageHeight);
    auto ySegments = splitProjectionProfile(yProj, minVal, minGapY);

    if (ySegments.size() > 1) {
        for (const auto& seg : ySegments) {
            std::vector<int> group;
            for (size_t i = 0; i < indices.size(); i++) {
                float cy = subBoxes[i].center().y;
                if (cy >= seg.first && cy < seg.second) {
                    group.push_back(indices[i]);
                }
            }
            recursiveYXCut(boxes, group, pageWidth, pageHeight, config, result);
        }
        return;
    }

    // Try X-axis split
    auto xProj = projectionByBboxes(subBoxes, 0, pageWidth);
    auto xSegments = splitProjectionProfile(xProj, minVal, minGapX);

    if (xSegments.size() > 1) {
        for (const auto& seg : xSegments) {
            std::vector<int> group;
            for (size_t i = 0; i < indices.size(); i++) {
                float cx = subBoxes[i].center().x;
                if (cx >= seg.first && cx < seg.second) {
                    group.push_back(indices[i]);
                }
            }
            recursiveYXCut(boxes, group, pageWidth, pageHeight, config, result);
        }
        return;
    }

    // No split — sort by position
    std::vector<int> sorted = indices;
    std::sort(sorted.begin(), sorted.end(), [&boxes](int a, int b) {
        float xa = boxes[a].center().x;
        float xb = boxes[b].center().x;
        float threshold = std::min(boxes[a].width(), boxes[b].width()) * 0.5f;
        if (std::abs(xa - xb) < threshold) {
            return boxes[a].center().y < boxes[b].center().y;
        }
        return xa > xb;  // Right to left for vertical text
    });
    result.insert(result.end(), sorted.begin(), sorted.end());
}

} // namespace detail

std::vector<int> xycutPlusSort(
    const std::vector<LayoutBox>& boxes,
    int pageWidth,
    int pageHeight,
    const XYCutConfig& config)
{
    if (boxes.empty()) return {};

    LOG_DEBUG("XY-Cut sorting {} boxes on {}x{} page", boxes.size(), pageWidth, pageHeight);

    // Build initial index list
    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);

    // Determine direction
    TextDirection dir = config.direction;
    if (dir == TextDirection::AUTO) {
        dir = detectTextDirection(boxes);
    }

    LOG_DEBUG("Text direction: {}", dir == TextDirection::HORIZONTAL ? "horizontal" : "vertical");

    std::vector<int> result;
    result.reserve(boxes.size());

    if (dir == TextDirection::HORIZONTAL) {
        detail::recursiveXYCut(boxes, indices, pageWidth, pageHeight, config, result);
    } else {
        detail::recursiveYXCut(boxes, indices, pageWidth, pageHeight, config, result);
    }

    return result;
}

} // namespace rapid_doc
