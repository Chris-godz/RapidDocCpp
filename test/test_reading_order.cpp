/**
 * @file test_reading_order.cpp
 * @brief Tests for XY-Cut++ reading order module
 *
 * Unit tests for the reading order algorithm (no model needed).
 * Tests cover projection, splitting, direction detection, and sorting.
 */

#include <gtest/gtest.h>
#include "reading_order/xycut.h"
#include <vector>
#include <cmath>

namespace {

rapid_doc::LayoutBox makeBox(float x0, float y0, float x1, float y1, rapid_doc::LayoutCategory cat = rapid_doc::LayoutCategory::TEXT) {
    return {x0, y0, x1, y1, cat, 0.9f, 0};
}

} // anonymous namespace

// ========================================
// Direction Detection Tests
// ========================================

TEST(ReadingOrder, DetectTextDirection_Horizontal) {
    std::vector<rapid_doc::LayoutBox> boxes;
    boxes.push_back(makeBox(10, 10, 100, 30));   // wide
    boxes.push_back(makeBox(10, 40, 120, 60));   // wide
    boxes.push_back(makeBox(10, 70, 90, 90));    // wide

    rapid_doc::TextDirection dir = rapid_doc::detectTextDirection(boxes);
    EXPECT_EQ(dir, rapid_doc::TextDirection::HORIZONTAL);
}

TEST(ReadingOrder, DetectTextDirection_Vertical) {
    std::vector<rapid_doc::LayoutBox> boxes;
    boxes.push_back(makeBox(10, 10, 30, 100));   // tall
    boxes.push_back(makeBox(40, 10, 60, 120));   // tall
    boxes.push_back(makeBox(70, 10, 90, 90));    // tall

    rapid_doc::TextDirection dir = rapid_doc::detectTextDirection(boxes);
    EXPECT_EQ(dir, rapid_doc::TextDirection::VERTICAL);
}

TEST(ReadingOrder, DetectTextDirection_Empty) {
    std::vector<rapid_doc::LayoutBox> boxes;
    rapid_doc::TextDirection dir = rapid_doc::detectTextDirection(boxes);
    EXPECT_EQ(dir, rapid_doc::TextDirection::HORIZONTAL);  // default
}

TEST(ReadingOrder, DetectTextDirection_Mixed) {
    std::vector<rapid_doc::LayoutBox> boxes;
    boxes.push_back(makeBox(10, 10, 100, 30));   // wide
    boxes.push_back(makeBox(10, 40, 30, 120));   // tall (not wide enough)

    rapid_doc::TextDirection dir = rapid_doc::detectTextDirection(boxes);
    EXPECT_EQ(dir, rapid_doc::TextDirection::HORIZONTAL);  // 50% threshold
}

// ========================================
// Projection Tests
// ========================================

TEST(ReadingOrder, ProjectionByBboxes_XAxis) {
    std::vector<rapid_doc::LayoutBox> boxes;
    boxes.push_back(makeBox(10, 0, 30, 100));
    boxes.push_back(makeBox(50, 0, 70, 100));

    auto proj = rapid_doc::detail::projectionByBboxes(boxes, 0, 100);

    EXPECT_EQ(proj.size(), 100u);
    EXPECT_EQ(proj[0], 0);
    EXPECT_EQ(proj[15], 1);   // in first box
    EXPECT_EQ(proj[60], 1);   // in second box
    EXPECT_EQ(proj[40], 0);   // gap
}

TEST(ReadingOrder, ProjectionByBboxes_YAxis) {
    std::vector<rapid_doc::LayoutBox> boxes;
    boxes.push_back(makeBox(0, 10, 100, 30));
    boxes.push_back(makeBox(0, 50, 100, 70));

    auto proj = rapid_doc::detail::projectionByBboxes(boxes, 1, 100);

    EXPECT_EQ(proj.size(), 100u);
    EXPECT_EQ(proj[20], 1);
    EXPECT_EQ(proj[60], 1);
    EXPECT_EQ(proj[40], 0);
}

// ========================================
// Split Projection Tests
// ========================================

TEST(ReadingOrder, SplitProjectionProfile_NoGaps) {
    std::vector<int> values(100, 1);
    auto segments = rapid_doc::detail::splitProjectionProfile(values, 0, 10);

    EXPECT_EQ(segments.size(), 1u);
    EXPECT_EQ(segments[0].first, 0);
    EXPECT_EQ(segments[0].second, 100);
}

TEST(ReadingOrder, SplitProjectionProfile_WithGaps) {
    std::vector<int> values(100, 0);
    values[10] = 1; values[11] = 1; values[12] = 1;
    values[30] = 1; values[31] = 1; values[32] = 1;
    values[60] = 1; values[61] = 1; values[62] = 1;

    auto segments = rapid_doc::detail::splitProjectionProfile(values, 0, 5);

    EXPECT_GE(segments.size(), 2u);
}

TEST(ReadingOrder, SplitProjectionProfile_SmallGapBelowMin) {
    std::vector<int> values(100, 0);
    values[10] = 1; values[11] = 1; values[12] = 1;
    values[15] = 1; values[16] = 1;  // gap too small

    auto segments = rapid_doc::detail::splitProjectionProfile(values, 0, 5);

    EXPECT_EQ(segments.size(), 1u);  // stays as one segment
}

// ========================================
// XY-Cut Sorting Tests
// ========================================

TEST(ReadingOrder, XycutSort_Empty) {
    std::vector<rapid_doc::LayoutBox> boxes;
    auto result = rapid_doc::xycutPlusSort(boxes, 800, 1200);
    EXPECT_TRUE(result.empty());
}

TEST(ReadingOrder, XycutSort_Single) {
    std::vector<rapid_doc::LayoutBox> boxes;
    boxes.push_back(makeBox(10, 10, 100, 50));

    auto result = rapid_doc::xycutPlusSort(boxes, 800, 1200);

    EXPECT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], 0);
}

TEST(ReadingOrder, XycutSort_TwoColumns) {
    std::vector<rapid_doc::LayoutBox> boxes;
    boxes.push_back(makeBox(400, 100, 600, 200));  // right column
    boxes.push_back(makeBox(50, 100, 250, 200));    // left column

    auto result = rapid_doc::xycutPlusSort(boxes, 800, 1200);

    EXPECT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], 1);  // left column first
    EXPECT_EQ(result[1], 0);  // right column second
}

TEST(ReadingOrder, XycutSort_TwoRows) {
    std::vector<rapid_doc::LayoutBox> boxes;
    boxes.push_back(makeBox(50, 400, 250, 500));   // bottom row
    boxes.push_back(makeBox(50, 50, 250, 150));     // top row

    auto result = rapid_doc::xycutPlusSort(boxes, 800, 1200);

    EXPECT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], 1);  // top row first
    EXPECT_EQ(result[1], 0);  // bottom row second
}

TEST(ReadingOrder, XycutSort_MultipleElements) {
    std::vector<rapid_doc::LayoutBox> boxes;
    boxes.push_back(makeBox(400, 400, 600, 500));   // right-bottom
    boxes.push_back(makeBox(50, 50, 200, 150));    // left-top
    boxes.push_back(makeBox(400, 50, 600, 150));   // right-top
    boxes.push_back(makeBox(50, 400, 200, 500));   // left-bottom

    rapid_doc::XYCutConfig config;
    config.direction = rapid_doc::TextDirection::HORIZONTAL;
    auto result = rapid_doc::xycutPlusSort(boxes, 800, 1200, config);

    EXPECT_EQ(result.size(), 4u);
    // For horizontal text: sort top-to-bottom, then left-to-right within each row
    // Boxes 1 (left-top) and 2 (right-top) are in same row, should be ordered left-to-right
    // Boxes 3 (left-bottom) and 0 (right-bottom) are in same row, should be ordered left-to-right
}

// ========================================
// YX-Cut Sorting Tests (Vertical Text)
// ========================================

TEST(ReadingOrder, YxcutSort_TwoRows) {
    std::vector<rapid_doc::LayoutBox> boxes;
    boxes.push_back(makeBox(400, 400, 600, 500));  // bottom
    boxes.push_back(makeBox(50, 50, 150, 150));    // top

    rapid_doc::XYCutConfig config;
    config.direction = rapid_doc::TextDirection::VERTICAL;
    auto result = rapid_doc::xycutPlusSort(boxes, 800, 1200, config);

    EXPECT_EQ(result.size(), 2u);
    EXPECT_EQ(result[0], 1);  // top first (for vertical: read top-down)
    EXPECT_EQ(result[1], 0);  // bottom second
}

// ========================================
// Boundary Tests
// ========================================

TEST(ReadingOrder, XycutSort_BoxOutOfBounds) {
    std::vector<rapid_doc::LayoutBox> boxes;
    boxes.push_back(makeBox(-10, -10, 50, 50));    // out of bounds
    boxes.push_back(makeBox(50, 50, 200, 200));    // normal

    auto result = rapid_doc::xycutPlusSort(boxes, 800, 1200);

    EXPECT_EQ(result.size(), 2u);
}

TEST(ReadingOrder, XycutSort_VerySmallPage) {
    std::vector<rapid_doc::LayoutBox> boxes;
    boxes.push_back(makeBox(1, 1, 5, 5));

    auto result = rapid_doc::xycutPlusSort(boxes, 10, 10);

    EXPECT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0], 0);
}

TEST(ReadingOrder, XycutSort_ConfigDefaults) {
    rapid_doc::XYCutConfig config;
    EXPECT_EQ(config.direction, rapid_doc::TextDirection::AUTO);
    EXPECT_FLOAT_EQ(config.minGapRatio, 0.05f);
    EXPECT_FLOAT_EQ(config.minValueRatio, 0.0f);
}

TEST(ReadingOrder, XycutSort_CustomConfig) {
    rapid_doc::XYCutConfig config;
    config.direction = rapid_doc::TextDirection::HORIZONTAL;
    config.minGapRatio = 0.1f;
    config.minValueRatio = 0.5f;

    EXPECT_EQ(config.direction, rapid_doc::TextDirection::HORIZONTAL);
    EXPECT_FLOAT_EQ(config.minGapRatio, 0.1f);
    EXPECT_FLOAT_EQ(config.minValueRatio, 0.5f);
}
