/**
 * @file test_table_preprocess.cpp
 * @brief Validate C++ table preprocessing matches Python DXNN output.
 *
 * Tests resize_with_padding(768) + BGR→RGB pipeline.
 */

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <filesystem>
#include "npy_loader.h"

namespace fs = std::filesystem;
#include "table/table_recognizer.h"

using namespace rapid_doc;
using namespace rapid_doc::test_utils;
using json = nlohmann::json;

static const std::string kFixtureDir = std::string(PROJECT_ROOT_DIR) + "/test/fixtures/table/";

class TablePreprocessTest : public ::testing::Test {
protected:
    void SetUp() override {
        fixtureExists_ = fs::exists(kFixtureDir + "input_image.png") &&
                          fs::exists(kFixtureDir + "preprocessed.npy");
        if (!fixtureExists_) {
            GTEST_SKIP() << "Table fixtures not found. Run tools/gen_test_vectors.py first.";
        }
    }
    bool fixtureExists_ = false;
};

TEST_F(TablePreprocessTest, OutputMatchesPython) {
    cv::Mat inputImage = cv::imread(kFixtureDir + "input_image.png");
    ASSERT_FALSE(inputImage.empty());

    NpyArray pyResult = loadNpy(kFixtureDir + "preprocessed.npy");
    ASSERT_EQ(pyResult.shape.size(), 4u);  // (1, 768, 768, 3)
    int targetSize = static_cast<int>(pyResult.shape[1]);

    // C++ preprocess: resize_with_padding + BGR→RGB
    int h = inputImage.rows, w = inputImage.cols;
    float scale = static_cast<float>(targetSize) / std::max(h, w);
    int newH = static_cast<int>(h * scale);
    int newW = static_cast<int>(w * scale);

    int interpolation = (scale < 1.0f) ? cv::INTER_AREA : cv::INTER_CUBIC;
    cv::Mat resized;
    cv::resize(inputImage, resized, cv::Size(newW, newH), 0, 0, interpolation);

    int padH = targetSize - newH;
    int padW = targetSize - newW;
    int padTop = padH / 2;
    int padBottom = padH - padTop;
    int padLeft = padW / 2;
    int padRight = padW - padLeft;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, padTop, padBottom, padLeft, padRight,
                       cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    ASSERT_EQ(rgb.rows, targetSize);
    ASSERT_EQ(rgb.cols, targetSize);
    ASSERT_EQ(rgb.channels(), 3);

    // Compare with Python output
    const uint8_t* pyData = pyResult.asUint8();
    int totalPixels = targetSize * targetSize * 3;
    int mismatchCount = 0;
    double l2Sum = 0.0;

    for (int i = 0; i < totalPixels; ++i) {
        int diff = static_cast<int>(rgb.data[i]) - static_cast<int>(pyData[i]);
        l2Sum += diff * diff;
        if (diff != 0) mismatchCount++;
    }

    double l2 = std::sqrt(l2Sum / totalPixels);
    EXPECT_EQ(mismatchCount, 0) << "Pixel mismatches: " << mismatchCount
                                 << "/" << totalPixels << ", L2=" << l2;
}

TEST_F(TablePreprocessTest, PaddingInfoMatchesPython) {
    if (!fs::exists(kFixtureDir + "preprocess_info.json")) {
        GTEST_SKIP() << "preprocess_info.json not found.";
    }

    cv::Mat inputImage = cv::imread(kFixtureDir + "input_image.png");
    ASSERT_FALSE(inputImage.empty());

    std::string jsonStr = loadJsonString(kFixtureDir + "preprocess_info.json");
    auto info = json::parse(jsonStr);

    int h = inputImage.rows, w = inputImage.cols;
    int targetSize = 768;
    float scale = static_cast<float>(targetSize) / std::max(h, w);
    int newH = static_cast<int>(h * scale);
    int newW = static_cast<int>(w * scale);
    int padTop = (targetSize - newH) / 2;
    int padLeft = (targetSize - newW) / 2;

    EXPECT_NEAR(scale, info["scale"].get<float>(), 1e-4f);
    EXPECT_EQ(padTop, info["pad_top"].get<int>());
    EXPECT_EQ(padLeft, info["pad_left"].get<int>());
    EXPECT_EQ(h, info["original_h"].get<int>());
    EXPECT_EQ(w, info["original_w"].get<int>());
}
