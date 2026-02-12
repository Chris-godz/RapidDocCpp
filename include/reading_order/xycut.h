#pragma once

/**
 * @file xycut.h
 * @brief XY-Cut++ reading order algorithm
 * 
 * Pure geometric algorithm — no model inference required.
 * Ported from Python: rapid_doc/model/reading_order/xycut_plus.py
 * 
 * Algorithm:
 *   1. Project bounding boxes onto X and Y axes to create 1D histograms
 *   2. Find gaps (valleys) in the projections to split regions
 *   3. Recursively split until no more splits are possible
 *   4. The leaf order gives the reading order
 * 
 * Supports two directions:
 *   - Horizontal text: XY-Cut (split X first, then Y within each column)
 *   - Vertical text: YX-Cut (split Y first, then X within each row)
 *   - Auto-detect based on bbox aspect ratios
 */

#include "common/types.h"
#include <vector>
#include <string>

namespace rapid_doc {

/**
 * @brief Text direction for reading order
 */
enum class TextDirection {
    HORIZONTAL,  // Left-to-right, top-to-bottom (most languages)
    VERTICAL,    // Top-to-bottom, right-to-left (CJK vertical)
    AUTO,        // Auto-detect from bbox aspect ratios
};

/**
 * @brief XY-Cut configuration
 */
struct XYCutConfig {
    TextDirection direction = TextDirection::AUTO;
    float minGapRatio = 0.05f;    // Minimum gap size relative to page dimension
    float minValueRatio = 0.0f;   // Minimum projection value for gap detection
};

/**
 * @brief Sort bounding boxes by reading order using XY-Cut++ algorithm
 * 
 * @param boxes Layout boxes to sort
 * @param pageWidth Page width in pixels
 * @param pageHeight Page height in pixels
 * @param config Algorithm configuration
 * @return Sorted indices (index into original boxes vector)
 */
std::vector<int> xycutPlusSort(
    const std::vector<LayoutBox>& boxes,
    int pageWidth,
    int pageHeight,
    const XYCutConfig& config = {}
);

/**
 * @brief Detect text direction from bounding box aspect ratios
 * 
 * If most boxes are wider than tall (w >= h * 1.5), direction is HORIZONTAL.
 * Otherwise VERTICAL.
 * 
 * @param boxes Layout boxes
 * @return Detected text direction
 */
TextDirection detectTextDirection(const std::vector<LayoutBox>& boxes);

// ========================================
// Internal functions (exposed for testing)
// ========================================
namespace detail {

/**
 * @brief Create 1D projection histogram from bounding boxes
 * @param boxes Bounding boxes
 * @param axis 0 for X-axis, 1 for Y-axis  
 * @param size Projection length (page width or height)
 * @return Projection array
 */
std::vector<int> projectionByBboxes(
    const std::vector<LayoutBox>& boxes,
    int axis,
    int size
);

/**
 * @brief Split projection profile at gaps
 * @param values Projection values
 * @param minValue Minimum value to consider "occupied"
 * @param minGap Minimum gap width to split
 * @return Pairs of (start, end) for each segment
 */
std::vector<std::pair<int, int>> splitProjectionProfile(
    const std::vector<int>& values,
    int minValue,
    int minGap
);

/**
 * @brief Recursive XY-Cut (horizontal text: X first, then Y)
 */
void recursiveXYCut(
    const std::vector<LayoutBox>& boxes,
    const std::vector<int>& indices,
    int pageWidth,
    int pageHeight,
    const XYCutConfig& config,
    std::vector<int>& result
);

/**
 * @brief Recursive YX-Cut (vertical text: Y first, then X)
 */
void recursiveYXCut(
    const std::vector<LayoutBox>& boxes,
    const std::vector<int>& indices,
    int pageWidth,
    int pageHeight,
    const XYCutConfig& config,
    std::vector<int>& result
);

} // namespace detail
} // namespace rapid_doc
