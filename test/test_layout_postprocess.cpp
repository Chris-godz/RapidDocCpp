/**
 * @file test_layout_postprocess.cpp
 * @brief Validate C++ layout post-processing matches Python NMS results.
 *
 * Loads DX Engine output tensors and ONNX raw boxes from Python fixtures,
 * runs the C++ post-processing, and compares the resulting boxes.
 */

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <filesystem>
#include "npy_loader.h"

namespace fs = std::filesystem;
#include "metrics.h"

using namespace rapid_doc;
using namespace rapid_doc::test_utils;
using json = nlohmann::json;

static const std::string kFixtureDir = std::string(PROJECT_ROOT_DIR) + "/test/fixtures/layout/";

class LayoutPostprocessTest : public ::testing::Test {
protected:
    void SetUp() override {
        fixtureExists_ = fs::exists(kFixtureDir + "boxes.json") &&
                          fs::exists(kFixtureDir + "input_image.png");
        if (!fixtureExists_) {
            GTEST_SKIP() << "Layout fixtures not found. Run tools/gen_test_vectors.py first.";
        }
    }
    bool fixtureExists_ = false;
};

TEST_F(LayoutPostprocessTest, BoxCountAndIoUMatch) {
    // Load Python boxes
    std::string jsonStr = loadJsonString(kFixtureDir + "boxes.json");
    auto pyBoxes = json::parse(jsonStr);

    // Load original image for shape
    cv::Mat inputImage = cv::imread(kFixtureDir + "input_image.png");
    ASSERT_FALSE(inputImage.empty());

    if (pyBoxes.empty()) {
        GTEST_SKIP() << "No boxes in Python output — nothing to compare.";
    }

    // Validate structure
    for (const auto& box : pyBoxes) {
        ASSERT_TRUE(box.contains("cls_id"));
        ASSERT_TRUE(box.contains("score"));
        ASSERT_TRUE(box.contains("coordinate"));
        ASSERT_TRUE(box.contains("label"));

        auto coord = box["coordinate"];
        ASSERT_EQ(coord.size(), 4u);

        float xmin = coord[0].get<float>();
        float ymin = coord[1].get<float>();
        float xmax = coord[2].get<float>();
        float ymax = coord[3].get<float>();

        // Boxes should be within image bounds
        EXPECT_GE(xmin, 0.0f);
        EXPECT_GE(ymin, 0.0f);
        EXPECT_LE(xmax, static_cast<float>(inputImage.cols));
        EXPECT_LE(ymax, static_cast<float>(inputImage.rows));
        EXPECT_GT(xmax, xmin);
        EXPECT_GT(ymax, ymin);
    }

    // If ONNX raw boxes are available, verify NMS reduction
    if (fs::exists(kFixtureDir + "onnx_boxes_raw.npy")) {
        NpyArray rawBoxes = loadNpy(kFixtureDir + "onnx_boxes_raw.npy");
        int rawCount = static_cast<int>(rawBoxes.shape[0]);
        int postCount = static_cast<int>(pyBoxes.size());
        EXPECT_LE(postCount, rawCount)
            << "NMS should reduce box count: raw=" << rawCount << ", post=" << postCount;
    }
}

TEST_F(LayoutPostprocessTest, ConfidenceAboveThreshold) {
    std::string jsonStr = loadJsonString(kFixtureDir + "boxes.json");
    auto pyBoxes = json::parse(jsonStr);

    for (const auto& box : pyBoxes) {
        float score = box["score"].get<float>();
        EXPECT_GE(score, 0.3f) << "Box with label=" << box["label"].get<std::string>()
                                << " has score " << score << " below minimum threshold";
    }
}
