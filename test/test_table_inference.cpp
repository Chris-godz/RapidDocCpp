/**
 * @file test_table_inference.cpp
 * @brief Validate C++ table UNet inference output matches Python.
 *
 * Compares segmentation masks using cosine similarity and pixel accuracy.
 */

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "npy_loader.h"

namespace fs = std::filesystem;
#include "metrics.h"

using namespace rapid_doc;
using namespace rapid_doc::test_utils;

static const std::string kFixtureDir = std::string(PROJECT_ROOT_DIR) + "/test/fixtures/table/";

class TableInferenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        fixtureExists_ = fs::exists(kFixtureDir + "seg_mask.npy");
        if (!fixtureExists_) {
            GTEST_SKIP() << "Table seg_mask fixture not found. Run tools/gen_test_vectors.py first.";
        }
    }
    bool fixtureExists_ = false;
};

TEST_F(TableInferenceTest, SegMaskStructureIsValid) {
    NpyArray pyMask = loadNpy(kFixtureDir + "seg_mask.npy");
    ASSERT_GE(pyMask.shape.size(), 2u);

    int h = static_cast<int>(pyMask.shape[pyMask.shape.size() - 2]);
    int w = static_cast<int>(pyMask.shape[pyMask.shape.size() - 1]);

    EXPECT_GT(h, 0);
    EXPECT_GT(w, 0);

    // Values should be 0, 1, or 2
    const uint8_t* data = pyMask.asUint8();
    int totalPixels = h * w;
    for (int i = 0; i < totalPixels; ++i) {
        EXPECT_LE(data[i], 2) << "Unexpected segmentation value " << static_cast<int>(data[i])
                               << " at pixel " << i;
    }
}

TEST_F(TableInferenceTest, SegMaskHasLineClasses) {
    NpyArray pyMask = loadNpy(kFixtureDir + "seg_mask.npy");
    const uint8_t* data = pyMask.asUint8();
    size_t total = pyMask.elementCount();

    int bgCount = 0, hLineCount = 0, vLineCount = 0;
    for (size_t i = 0; i < total; ++i) {
        if (data[i] == 0) bgCount++;
        else if (data[i] == 1) hLineCount++;
        else if (data[i] == 2) vLineCount++;
    }

    // For a wired table, expect some horizontal and vertical lines
    EXPECT_GT(hLineCount, 0) << "No horizontal lines detected in segmentation mask";
    EXPECT_GT(vLineCount, 0) << "No vertical lines detected in segmentation mask";
    EXPECT_GT(bgCount, hLineCount + vLineCount) << "Background should be majority class";
}

TEST_F(TableInferenceTest, CosineSimilarityWithCppOutput) {
    // This test requires running C++ inference first and saving the result.
    // It serves as a template for when DX Engine is available at test time.

    if (!fs::exists(kFixtureDir + "cpp_seg_mask.npy")) {
        GTEST_SKIP() << "C++ seg_mask output not found — run inference test first.";
    }

    NpyArray pyMask = loadNpy(kFixtureDir + "seg_mask.npy");
    NpyArray cppMask = loadNpy(kFixtureDir + "cpp_seg_mask.npy");

    ASSERT_EQ(pyMask.elementCount(), cppMask.elementCount());

    std::vector<float> pyVec = pyMask.toFloatVector();
    std::vector<float> cppVec = cppMask.toFloatVector();

    float cosSim = test_metrics::calculateCosineSimilarity(pyVec, cppVec);
    EXPECT_GE(cosSim, 0.99f) << "Cosine similarity between Python and C++ seg masks: " << cosSim;

    // Also check pixel accuracy
    const uint8_t* pyData = pyMask.asUint8();
    const uint8_t* cppData = cppMask.asUint8();
    size_t total = pyMask.elementCount();
    int match = 0;
    for (size_t i = 0; i < total; ++i) {
        if (pyData[i] == cppData[i]) match++;
    }
    float accuracy = static_cast<float>(match) / total;
    EXPECT_GE(accuracy, 0.95f) << "Pixel accuracy: " << accuracy;
}
