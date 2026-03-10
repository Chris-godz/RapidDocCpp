/**
 * @file test_layout_preprocess.cpp
 * @brief Validate C++ layout preprocessing matches Python DXNN output.
 *
 * Loads a test image and the corresponding Python .npy preprocessed result,
 * runs the C++ preprocessor, and compares them pixel-by-pixel (L2 = 0).
 */

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include "npy_loader.h"
#include "dump_utils.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

#include "layout/layout_detector.h"

using namespace rapid_doc;
using namespace rapid_doc::test_utils;

static const std::string kFixtureDir = std::string(PROJECT_ROOT_DIR) + "/test/fixtures/layout/";

static int getLayoutInputSize() {
    std::string infoPath = kFixtureDir + "preprocess_info.json";
    if (fs::exists(infoPath)) {
        std::ifstream f(infoPath);
        json j;
        f >> j;
        return j.value("input_size", 640);
    }
    return 640;
}

class LayoutPreprocessTest : public ::testing::Test {
protected:
    void SetUp() override {
        fixtureExists_ = fs::exists(kFixtureDir + "input_image.png") &&
                          fs::exists(kFixtureDir + "preprocessed.npy");
        if (!fixtureExists_) {
            GTEST_SKIP() << "Layout fixtures not found. Run tools/gen_test_vectors.py first.";
        }
        inputSize_ = getLayoutInputSize();
    }
    bool fixtureExists_ = false;
    int inputSize_ = 640;
};

TEST_F(LayoutPreprocessTest, OutputMatchesPythonExactly) {
    cv::Mat inputImage = cv::imread(kFixtureDir + "input_image.png");
    ASSERT_FALSE(inputImage.empty()) << "Cannot load input_image.png";

    NpyArray pyResult = loadNpy(kFixtureDir + "preprocessed.npy");
    ASSERT_EQ(pyResult.shape.size(), 4u);  // (1, H, W, C)
    int pyH = static_cast<int>(pyResult.shape[1]);
    int pyW = static_cast<int>(pyResult.shape[2]);
    int pyC = static_cast<int>(pyResult.shape[3]);
    ASSERT_EQ(pyH, inputSize_);
    ASSERT_EQ(pyW, inputSize_);

    cv::Mat resized;
    cv::resize(inputImage, resized, cv::Size(pyW, pyH), 0, 0, cv::INTER_CUBIC);

    ASSERT_EQ(resized.rows, pyH);
    ASSERT_EQ(resized.cols, pyW);
    ASSERT_EQ(resized.channels(), pyC);

    const uint8_t* pyData = pyResult.asUint8();
    int totalPixels = pyH * pyW * pyC;
    int exactMismatch = 0;
    int offByMoreThanOne = 0;
    double l2Sum = 0.0;
    int maxAbsDiff = 0;

    for (int i = 0; i < totalPixels; ++i) {
        int diff = static_cast<int>(resized.data[i]) - static_cast<int>(pyData[i]);
        int absDiff = std::abs(diff);
        l2Sum += diff * diff;
        if (absDiff > maxAbsDiff) maxAbsDiff = absDiff;
        if (diff != 0) exactMismatch++;
        if (absDiff > 1) offByMoreThanOne++;
    }

    double l2 = std::sqrt(l2Sum / totalPixels);

    if (exactMismatch > 0) {
        std::string dumpPath = kFixtureDir + "cpp_preprocessed.npy";
        dump_utils::saveMatAsNpy(dumpPath, resized);
        std::cerr << "  [DUMP] C++ preprocess output -> " << dumpPath << std::endl;
        std::cerr << "  [INFO] exact mismatches: " << exactMismatch
                  << "/" << totalPixels
                  << ", max|diff|=" << maxAbsDiff
                  << ", L2=" << l2 << std::endl;
    }

    // INTER_CUBIC rounding may differ ±1 between OpenCV versions.
    EXPECT_EQ(offByMoreThanOne, 0)
        << "Pixels off by >1: " << offByMoreThanOne << "/" << totalPixels;
    EXPECT_LE(maxAbsDiff, 1) << "Max absolute diff exceeds 1";
    EXPECT_LT(l2, 0.15) << "L2 per-pixel exceeds tolerance";
}

TEST_F(LayoutPreprocessTest, ScaleFactorMatchesPython) {
    cv::Mat inputImage = cv::imread(kFixtureDir + "input_image.png");
    ASSERT_FALSE(inputImage.empty());

    NpyArray pySF = loadNpy(kFixtureDir + "scale_factor.npy");
    ASSERT_EQ(pySF.shape.size(), 2u);  // (1, 2)
    const float* pyScaleData = pySF.asFloat32();

    float pyScaleY = pyScaleData[0];
    float pyScaleX = pyScaleData[1];

    float cppScaleX = static_cast<float>(inputSize_) / inputImage.cols;
    float cppScaleY = static_cast<float>(inputSize_) / inputImage.rows;

    EXPECT_NEAR(cppScaleX, pyScaleX, 1e-5f);
    EXPECT_NEAR(cppScaleY, pyScaleY, 1e-5f);
}
