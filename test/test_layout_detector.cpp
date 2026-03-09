/**
 * @file test_layout_detector.cpp
 * @brief Tests for Layout detection module
 *
 * Two-level testing:
 *   1. Unit tests: config defaults, type helpers, guard conditions (no model needed)
 *   2. End-to-end: DXNN + ONNX inference on synthetic image (SKIP if models absent)
 *
 * Tests are SKIPPED when model files are not present, allowing CI
 * to run without NPU hardware.
 */

#include <gtest/gtest.h>
#include "layout/layout_detector.h"
#include "test_utils/tensor_compare.h"
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

namespace {

const char* LAYOUT_DXNN_MODEL = PROJECT_ROOT_DIR
    "/engine/model_files/layout/pp_doclayout_l_part1.dxnn";
const char* LAYOUT_ONNX_POST = PROJECT_ROOT_DIR
    "/engine/model_files/layout/pp_doclayout_l_part2.onnx";

bool modelsExist() {
    return fs::exists(LAYOUT_DXNN_MODEL) && fs::exists(LAYOUT_ONNX_POST);
}

// Create a synthetic test image (colored blocks simulating a document page)
cv::Mat createTestImage(int width = 800, int height = 1200) {
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
    // Title block
    cv::rectangle(img, {50, 30}, {750, 80}, cv::Scalar(0, 0, 0), -1);
    // Text block 1
    for (int y = 120; y < 400; y += 20) {
        cv::line(img, {50, y}, {750, y}, cv::Scalar(100, 100, 100), 1);
    }
    // Table-like block
    cv::rectangle(img, {50, 450}, {750, 700}, cv::Scalar(0, 0, 0), 2);
    for (int x = 50; x <= 750; x += 175) {
        cv::line(img, {x, 450}, {x, 700}, cv::Scalar(0, 0, 0), 1);
    }
    for (int y = 450; y <= 700; y += 50) {
        cv::line(img, {50, y}, {750, y}, cv::Scalar(0, 0, 0), 1);
    }
    // Image placeholder
    cv::rectangle(img, {200, 750}, {600, 1050}, cv::Scalar(200, 200, 200), -1);
    return img;
}

}  // anonymous namespace

// ========================================
// Default Config Test
// ========================================

TEST(LayoutDetector, DefaultConfigValues) {
    rapid_doc::LayoutDetectorConfig config;

    // PP-DocLayout-L model uses 640x640 input
    EXPECT_EQ(config.inputSize, 640);
    EXPECT_FLOAT_EQ(config.confThreshold, 0.5f);
    EXPECT_FALSE(config.useAsync);
    EXPECT_TRUE(config.dxnnModelPath.empty());
    EXPECT_TRUE(config.onnxSubModelPath.empty());
}

// ========================================
// Config Assignment Test
// ========================================

TEST(LayoutDetector, ConfigValues) {
    rapid_doc::LayoutDetectorConfig config;
    config.inputSize = 800;
    config.confThreshold = 0.3f;
    config.dxnnModelPath = "test.dxnn";
    config.onnxSubModelPath = "test.onnx";
    config.useAsync = true;

    EXPECT_EQ(config.inputSize, 800);
    EXPECT_FLOAT_EQ(config.confThreshold, 0.3f);
    EXPECT_EQ(config.dxnnModelPath, "test.dxnn");
    EXPECT_EQ(config.onnxSubModelPath, "test.onnx");
    EXPECT_TRUE(config.useAsync);
}

// ========================================
// Module Initialization Test
// ========================================

TEST(LayoutDetector, InitializeWithoutModel) {
    rapid_doc::LayoutDetectorConfig cfg;
    cfg.dxnnModelPath = "/nonexistent.dxnn";
    cfg.onnxSubModelPath = "/nonexistent.onnx";

    rapid_doc::LayoutDetector detector(cfg);
    // initialize() must return false when model files don't exist
    EXPECT_FALSE(detector.initialize());
    EXPECT_FALSE(detector.isInitialized());
}

// ========================================
// Guard Conditions (no model needed)
// ========================================

TEST(LayoutDetector, DetectWithoutInit) {
    rapid_doc::LayoutDetectorConfig cfg;
    rapid_doc::LayoutDetector detector(cfg);
    // detect() on uninitialized detector should return empty result, no crash
    cv::Mat img = createTestImage();
    auto result = detector.detect(img);
    EXPECT_TRUE(result.boxes.empty());
    EXPECT_DOUBLE_EQ(result.inferenceTimeMs, 0.0);
}

TEST(LayoutDetector, DetectEmptyImage) {
    rapid_doc::LayoutDetectorConfig cfg;
    rapid_doc::LayoutDetector detector(cfg);
    cv::Mat emptyImg;
    auto result = detector.detect(emptyImg);
    EXPECT_TRUE(result.boxes.empty());
}

// ========================================
// Layout Type Helper Tests
// ========================================

TEST(LayoutTypes, CategorySupport) {
    using rapid_doc::LayoutCategory;
    using rapid_doc::isCategorySupported;

    // Supported categories
    EXPECT_TRUE(isCategorySupported(LayoutCategory::TEXT));
    EXPECT_TRUE(isCategorySupported(LayoutCategory::TABLE));
    EXPECT_TRUE(isCategorySupported(LayoutCategory::IMAGE));
    EXPECT_TRUE(isCategorySupported(LayoutCategory::PARAGRAPH_TITLE));
    EXPECT_TRUE(isCategorySupported(LayoutCategory::DOC_TITLE));
    EXPECT_TRUE(isCategorySupported(LayoutCategory::ABSTRACT));
    EXPECT_TRUE(isCategorySupported(LayoutCategory::CONTENT));
    EXPECT_TRUE(isCategorySupported(LayoutCategory::FIGURE_TITLE));
    EXPECT_TRUE(isCategorySupported(LayoutCategory::TABLE_TITLE));
    EXPECT_TRUE(isCategorySupported(LayoutCategory::CHART_TITLE));
    EXPECT_TRUE(isCategorySupported(LayoutCategory::CHART));
    EXPECT_TRUE(isCategorySupported(LayoutCategory::REFERENCE));
    EXPECT_TRUE(isCategorySupported(LayoutCategory::ALGORITHM));
    EXPECT_TRUE(isCategorySupported(LayoutCategory::ASIDE_TEXT));

    // Unsupported categories (formula + abandon)
    EXPECT_FALSE(isCategorySupported(LayoutCategory::FORMULA));
    EXPECT_FALSE(isCategorySupported(LayoutCategory::FORMULA_NUMBER));
    EXPECT_FALSE(isCategorySupported(LayoutCategory::HEADER_IMAGE));
    EXPECT_FALSE(isCategorySupported(LayoutCategory::FOOTER_IMAGE));
    EXPECT_FALSE(isCategorySupported(LayoutCategory::SEAL));
    EXPECT_FALSE(isCategorySupported(LayoutCategory::NUMBER));
    EXPECT_FALSE(isCategorySupported(LayoutCategory::FOOTNOTE));
    EXPECT_FALSE(isCategorySupported(LayoutCategory::HEADER));
    EXPECT_FALSE(isCategorySupported(LayoutCategory::FOOTER));
}

TEST(LayoutTypes, TextBoxesIncludeCaptions) {
    rapid_doc::LayoutResult result;
    // Text-like categories
    result.boxes.push_back({0, 0, 100, 50, rapid_doc::LayoutCategory::TEXT, 0.9f, 0});
    result.boxes.push_back({0, 50, 100, 100, rapid_doc::LayoutCategory::PARAGRAPH_TITLE, 0.9f, 1});
    result.boxes.push_back({0, 100, 100, 150, rapid_doc::LayoutCategory::DOC_TITLE, 0.9f, 2});
    // Caption categories (should now be included in text boxes)
    result.boxes.push_back({0, 150, 100, 200, rapid_doc::LayoutCategory::FIGURE_TITLE, 0.9f, 3});
    result.boxes.push_back({0, 200, 100, 250, rapid_doc::LayoutCategory::TABLE_TITLE, 0.9f, 4});
    result.boxes.push_back({0, 250, 100, 300, rapid_doc::LayoutCategory::CHART_TITLE, 0.9f, 5});
    // Non-text categories
    result.boxes.push_back({0, 300, 100, 400, rapid_doc::LayoutCategory::TABLE, 0.9f, 6});
    result.boxes.push_back({0, 400, 100, 500, rapid_doc::LayoutCategory::IMAGE, 0.9f, 7});
    result.boxes.push_back({0, 500, 100, 600, rapid_doc::LayoutCategory::FORMULA, 0.8f, 8});

    auto textBoxes = result.getTextBoxes();
    // TEXT + PARAGRAPH_TITLE + DOC_TITLE + FIGURE_TITLE + TABLE_TITLE + CHART_TITLE = 6
    EXPECT_EQ(textBoxes.size(), 6u);

    auto tableBoxes = result.getTableBoxes();
    EXPECT_EQ(tableBoxes.size(), 1u);

    auto supportedBoxes = result.getSupportedBoxes();
    // All except FORMULA = 8
    EXPECT_EQ(supportedBoxes.size(), 8u);

    auto unsupportedBoxes = result.getUnsupportedBoxes();
    // Only FORMULA = 1
    EXPECT_EQ(unsupportedBoxes.size(), 1u);
}

TEST(LayoutTypes, BoxConvenience) {
    rapid_doc::LayoutBox box{10.0f, 20.0f, 50.0f, 80.0f,
                             rapid_doc::LayoutCategory::TEXT, 0.95f, 0};
    EXPECT_FLOAT_EQ(box.width(), 40.0f);
    EXPECT_FLOAT_EQ(box.height(), 60.0f);
    EXPECT_FLOAT_EQ(box.area(), 2400.0f);

    auto center = box.center();
    EXPECT_FLOAT_EQ(center.x, 30.0f);
    EXPECT_FLOAT_EQ(center.y, 50.0f);

    auto rect = box.toRect();
    EXPECT_EQ(rect.x, 10);
    EXPECT_EQ(rect.y, 20);
    EXPECT_EQ(rect.width, 40);
    EXPECT_EQ(rect.height, 60);
}

TEST(LayoutTypes, CategoryToString) {
    using rapid_doc::LayoutCategory;
    using rapid_doc::layoutCategoryToString;

    EXPECT_STREQ(layoutCategoryToString(LayoutCategory::TEXT), "text");
    EXPECT_STREQ(layoutCategoryToString(LayoutCategory::TABLE), "table");
    EXPECT_STREQ(layoutCategoryToString(LayoutCategory::FORMULA), "formula");
    EXPECT_STREQ(layoutCategoryToString(LayoutCategory::UNKNOWN), "unknown");
    EXPECT_STREQ(layoutCategoryToString(LayoutCategory::PARAGRAPH_TITLE), "paragraph_title");
    EXPECT_STREQ(layoutCategoryToString(LayoutCategory::ASIDE_TEXT), "aside_text");
}

TEST(LayoutTypes, EmptyResult) {
    rapid_doc::LayoutResult result;
    EXPECT_TRUE(result.boxes.empty());
    EXPECT_TRUE(result.getTextBoxes().empty());
    EXPECT_TRUE(result.getTableBoxes().empty());
    EXPECT_TRUE(result.getSupportedBoxes().empty());
    EXPECT_TRUE(result.getUnsupportedBoxes().empty());
    EXPECT_DOUBLE_EQ(result.inferenceTimeMs, 0.0);
}

// ========================================
// End-to-End with Model (SKIP if no model)
// ========================================

TEST(LayoutDetector, EndToEnd_DxnnVsOnnx) {
    if (!modelsExist()) GTEST_SKIP() << "Models not found";

    cv::Mat img = createTestImage();

    rapid_doc::LayoutDetectorConfig cfg;
    cfg.dxnnModelPath = LAYOUT_DXNN_MODEL;
    cfg.onnxSubModelPath = LAYOUT_ONNX_POST;
    cfg.confThreshold = 0.5f;

    rapid_doc::LayoutDetector detector(cfg);
    ASSERT_TRUE(detector.initialize());
    EXPECT_TRUE(detector.isInitialized());

    auto result = detector.detect(img);

    // Verify detection produced results
    EXPECT_FALSE(result.boxes.empty()) << "Expected layout detections";
    EXPECT_GT(result.inferenceTimeMs, 0.0);

    // Verify box properties
    for (const auto& box : result.boxes) {
        EXPECT_GE(box.confidence, cfg.confThreshold);
        EXPECT_GE(box.x0, 0.0f);
        EXPECT_GE(box.y0, 0.0f);
        EXPECT_LT(box.x0, static_cast<float>(img.cols));
        EXPECT_LT(box.y0, static_cast<float>(img.rows));
        EXPECT_GT(box.width(), 0.0f);
        EXPECT_GT(box.height(), 0.0f);
        EXPECT_NE(box.category, rapid_doc::LayoutCategory::UNKNOWN);
    }
}
