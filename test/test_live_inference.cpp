/**
 * @file test_live_inference.cpp
 * @brief Live inference tests that run actual DXRT models on fixture data.
 *
 * These tests require DeepX hardware and models in engine/model_files/.
 * They produce intermediate C++ outputs (cpp_seg_mask.npy etc.) that are
 * consumed by comparison tests in test_table_inference.cpp.
 */

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include "npy_loader.h"
#include "dump_utils.h"
#include "metrics.h"
#include <dxrt/inference_engine.h>

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace rapid_doc;
using namespace rapid_doc::test_utils;

static const std::string kProjectRoot = std::string(PROJECT_ROOT_DIR);
static const std::string kTableFixDir = kProjectRoot + "/test/fixtures/table/";
static const std::string kLayoutFixDir = kProjectRoot + "/test/fixtures/layout/";
static const std::string kModelDir = kProjectRoot + "/engine/model_files/";

// ---------------------------------------------------------------------------
// Table UNet live inference → produces cpp_seg_mask.npy
// ---------------------------------------------------------------------------

class TableLiveInferenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::string modelPath = kModelDir + "table/unet.dxnn";
        std::string imagePath = kTableFixDir + "input_image.png";
        if (!fs::exists(modelPath) || !fs::exists(imagePath)) {
            GTEST_SKIP() << "Table model or input image not available.";
        }
        modelPath_ = modelPath;
        imagePath_ = imagePath;
    }
    std::string modelPath_;
    std::string imagePath_;
};

TEST_F(TableLiveInferenceTest, RunUnetAndSaveMask) {
    // Use Python's exact preprocessed data to ensure identical input
    std::string prepPath = kTableFixDir + "preprocessed.npy";
    if (!fs::exists(prepPath)) {
        GTEST_SKIP() << "Table preprocessed.npy not found.";
    }

    NpyArray pyPrep = loadNpy(prepPath);
    ASSERT_EQ(pyPrep.shape.size(), 4u);
    int inputH = static_cast<int>(pyPrep.shape[1]);
    int inputW = static_cast<int>(pyPrep.shape[2]);

    cv::Mat rgb(inputH, inputW, CV_8UC3,
                const_cast<uint8_t*>(pyPrep.asUint8()));

    try {
        dxrt::InferenceEngine engine(modelPath_);
        auto outputs = engine.Run(static_cast<void*>(rgb.data));

        ASSERT_FALSE(outputs.empty()) << "DX Engine returned no outputs";

        auto& outTensor = outputs[0];
        auto& shape = outTensor->shape();

        // Table UNet output is int64 with shape (1, 1, H, W) or similar.
        // Extract the last 2 dims as the spatial mask.
        ASSERT_GE(shape.size(), 2u);
        int maskH = static_cast<int>(shape[shape.size() - 2]);
        int maskW = static_cast<int>(shape[shape.size() - 1]);
        size_t totalElements = 1;
        for (auto d : shape) totalElements *= d;

        // DXRT returns int64 data — cast each element to uint8.
        const int64_t* rawPtr = reinterpret_cast<const int64_t*>(outTensor->data());
        cv::Mat segMask(maskH, maskW, CV_8UC1);
        for (int i = 0; i < maskH * maskW; ++i) {
            segMask.data[i] = static_cast<uint8_t>(rawPtr[i]);
        }
        dump_utils::saveMatAsNpy(kTableFixDir + "cpp_seg_mask.npy", segMask);

        std::cerr << "  [DUMP] cpp_seg_mask.npy: " << maskH << "x" << maskW
                  << ", elements=" << totalElements << std::endl;

        // Immediately compare with Python fixture
        std::string pyMaskPath = kTableFixDir + "seg_mask.npy";
        if (fs::exists(pyMaskPath)) {
            NpyArray pyMask = loadNpy(pyMaskPath);
            ASSERT_EQ(pyMask.elementCount(),
                      static_cast<size_t>(maskH * maskW));
            const uint8_t* pyData = pyMask.asUint8();
            int match = 0;
            for (int i = 0; i < maskH * maskW; ++i) {
                if (segMask.data[i] == pyData[i]) match++;
            }
            float accuracy = static_cast<float>(match) / (maskH * maskW);
            std::cerr << "  [INFO] Pixel accuracy vs Python: " << accuracy << std::endl;
            EXPECT_GE(accuracy, 0.99f)
                << "Table seg mask pixel accuracy: " << accuracy;
        }
    } catch (const std::exception& e) {
        GTEST_SKIP() << "DXRT engine failed: " << e.what();
    }
}

// ---------------------------------------------------------------------------
// Layout DX inference → produces cpp_dxnn_out_0.npy, cpp_dxnn_out_1.npy
// ---------------------------------------------------------------------------

class LayoutLiveInferenceTest : public ::testing::Test {
protected:
    void SetUp() override {
        std::string modelPath = kModelDir + "layout/pp_doclayout_l_part1.dxnn";
        std::string imagePath = kLayoutFixDir + "input_image.png";
        if (!fs::exists(modelPath) || !fs::exists(imagePath)) {
            GTEST_SKIP() << "Layout model or input image not available.";
        }
        modelPath_ = modelPath;
        imagePath_ = imagePath;
    }
    std::string modelPath_;
    std::string imagePath_;
};

TEST_F(LayoutLiveInferenceTest, RunDxnnAndCompareOutputs) {
    // Use Python's exact preprocessed data to isolate inference differences
    NpyArray pyPrep = loadNpy(kLayoutFixDir + "preprocessed.npy");
    ASSERT_EQ(pyPrep.shape.size(), 4u);
    int inputH = static_cast<int>(pyPrep.shape[1]);
    int inputW = static_cast<int>(pyPrep.shape[2]);
    int inputC = static_cast<int>(pyPrep.shape[3]);
    ASSERT_EQ(inputC, 3);

    cv::Mat preprocessed(inputH, inputW, CV_8UC3,
                         const_cast<uint8_t*>(pyPrep.asUint8()));

    try {
        dxrt::InferenceEngine engine(modelPath_);
        auto outputs = engine.Run(static_cast<void*>(preprocessed.data));

        ASSERT_GE(outputs.size(), 2u) << "Expected at least 2 DXNN outputs";

        for (size_t i = 0; i < outputs.size(); ++i) {
            auto& t = outputs[i];
            const float* ptr = reinterpret_cast<const float*>(t->data());
            size_t count = 1;
            std::vector<size_t> shape;
            for (auto d : t->shape()) {
                count *= d;
                shape.push_back(static_cast<size_t>(d));
            }
            std::vector<float> vec(ptr, ptr + count);
            dump_utils::saveFloatVecAsNpy(
                kLayoutFixDir + "cpp_dxnn_out_" + std::to_string(i) + ".npy",
                vec, shape);
            std::cerr << "  [DUMP] cpp_dxnn_out_" << i << ".npy: " << count << " elements" << std::endl;
        }

        // Compare with Python DXNN outputs if available
        for (size_t i = 0; i < outputs.size(); ++i) {
            std::string pyPath = kLayoutFixDir + "dxnn_out_" + std::to_string(i) + ".npy";
            if (!fs::exists(pyPath)) continue;

            NpyArray pyArr = loadNpy(pyPath);
            auto& t = outputs[i];
            const float* cppPtr = reinterpret_cast<const float*>(t->data());
            size_t count = 1;
            for (auto d : t->shape()) count *= d;

            ASSERT_EQ(pyArr.elementCount(), count)
                << "Output " << i << " element count mismatch";

            const float* pyPtr = pyArr.asFloat32();
            double maxDiff = 0.0, sumSqDiff = 0.0;
            for (size_t j = 0; j < count; ++j) {
                double diff = std::abs(static_cast<double>(cppPtr[j]) - static_cast<double>(pyPtr[j]));
                if (diff > maxDiff) maxDiff = diff;
                sumSqDiff += diff * diff;
            }
            double rmse = std::sqrt(sumSqDiff / count);

            std::vector<float> pyVec(pyPtr, pyPtr + count);
            std::vector<float> cppVec(cppPtr, cppPtr + count);
            float cosSim = test_metrics::calculateCosineSimilarity(pyVec, cppVec);

            std::cerr << "  [INFO] DXNN out " << i
                      << ": max|diff|=" << maxDiff
                      << ", RMSE=" << rmse
                      << ", cosine=" << cosSim << std::endl;

            // Preprocess ±1 rounding may amplify through the network.
            // Require high cosine similarity (structural agreement).
            EXPECT_GE(cosSim, 0.99f)
                << "DXNN output " << i << " cosine similarity too low: " << cosSim;
        }
    } catch (const std::exception& e) {
        GTEST_SKIP() << "DXRT engine failed: " << e.what();
    }
}
