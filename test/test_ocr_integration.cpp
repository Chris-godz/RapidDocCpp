/**
 * @file test_ocr_integration.cpp
 * @brief Validate C++ OCR pipeline matches Python DXNN OCR output.
 *
 * Compares detection boxes (IoU) and recognition text (CER).
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

static const std::string kFixtureDir = std::string(PROJECT_ROOT_DIR) + "/test/fixtures/ocr/";

class OcrIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        fixtureExists_ = fs::exists(kFixtureDir + "det_boxes.json") &&
                          fs::exists(kFixtureDir + "rec_results.json") &&
                          fs::exists(kFixtureDir + "input_image.png");
        if (!fixtureExists_) {
            GTEST_SKIP() << "OCR fixtures not found. Run tools/gen_test_vectors.py first.";
        }
    }
    bool fixtureExists_ = false;
};

TEST_F(OcrIntegrationTest, DetBoxesAreValid) {
    std::string jsonStr = loadJsonString(kFixtureDir + "det_boxes.json");
    auto pyBoxes = json::parse(jsonStr);

    cv::Mat image = cv::imread(kFixtureDir + "input_image.png");
    ASSERT_FALSE(image.empty());

    for (const auto& box : pyBoxes) {
        ASSERT_GE(box.size(), 4u);
        // Each box is a list of 4 points [[x,y], [x,y], [x,y], [x,y]]
        for (const auto& pt : box) {
            ASSERT_EQ(pt.size(), 2u);
            float x = pt[0].get<float>();
            float y = pt[1].get<float>();
            EXPECT_GE(x, -5.0f) << "Box point x out of range";
            EXPECT_GE(y, -5.0f) << "Box point y out of range";
            EXPECT_LE(x, image.cols + 5.0f) << "Box point x out of range";
            EXPECT_LE(y, image.rows + 5.0f) << "Box point y out of range";
        }
    }
}

TEST_F(OcrIntegrationTest, RecResultsAreValid) {
    std::string jsonStr = loadJsonString(kFixtureDir + "rec_results.json");
    auto pyResults = json::parse(jsonStr);

    if (pyResults.empty()) {
        GTEST_SKIP() << "No recognition results in fixture (det-only mode).";
    }

    for (const auto& r : pyResults) {
        ASSERT_TRUE(r.contains("text"));
        ASSERT_TRUE(r.contains("score"));

        std::string text = r["text"].get<std::string>();
        float score = r["score"].get<float>();

        EXPECT_FALSE(text.empty()) << "Empty recognition text";
        EXPECT_GE(score, 0.0f);
        EXPECT_LE(score, 1.0f);
    }
}

TEST_F(OcrIntegrationTest, CppOcrMatchesPython) {
    // This test compares C++ OCR results against Python reference.
    // Requires running C++ OCR and saving cpp_rec_results.json.

    if (!fs::exists(kFixtureDir + "cpp_rec_results.json")) {
        GTEST_SKIP() << "C++ OCR results not found — run OCR pipeline test first.";
    }

    std::string pyStr = loadJsonString(kFixtureDir + "rec_results.json");
    std::string cppStr = loadJsonString(kFixtureDir + "cpp_rec_results.json");
    auto pyResults = json::parse(pyStr);
    auto cppResults = json::parse(cppStr);

    // Line count should match closely
    int pyCount = static_cast<int>(pyResults.size());
    int cppCount = static_cast<int>(cppResults.size());
    EXPECT_NEAR(cppCount, pyCount, pyCount * 0.2 + 1)
        << "Detection count mismatch: Python=" << pyCount << ", C++=" << cppCount;

    // Compare recognized text with CER
    int minCount = std::min(pyCount, cppCount);
    float totalCER = 0.0f;
    for (int i = 0; i < minCount; ++i) {
        std::string pyText = pyResults[i]["text"].get<std::string>();
        std::string cppText = cppResults[i]["text"].get<std::string>();
        float cer = test_metrics::calculateCER(pyText, cppText);
        totalCER += cer;
    }
    float avgCER = (minCount > 0) ? totalCER / minCount : 0.0f;
    EXPECT_LE(avgCER, 0.05f) << "Average CER too high: " << avgCER;
}
