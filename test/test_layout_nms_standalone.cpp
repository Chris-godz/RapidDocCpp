/**
 * @file test_layout_nms_standalone.cpp
 * @brief Standalone test: load Python DX Engine output tensors,
 *        run ONNX NMS + C++ post-processing, compare with Python boxes.json.
 *
 * This isolates the ONNX + NMS chain from the DX Engine inference,
 * allowing precise validation of the post-processing logic.
 */

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <cmath>
#include "npy_loader.h"
#include "metrics.h"

#ifdef HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace rapid_doc::test_utils;

static const std::string kFixtureDir = std::string(PROJECT_ROOT_DIR) + "/test/fixtures/layout/";

class LayoutNmsStandaloneTest : public ::testing::Test {
protected:
    void SetUp() override {
        hasFixtures_ = fs::exists(kFixtureDir + "dxnn_out_0.npy") &&
                       fs::exists(kFixtureDir + "dxnn_out_1.npy") &&
                       fs::exists(kFixtureDir + "scale_factor.npy") &&
                       fs::exists(kFixtureDir + "boxes.json");
        if (!hasFixtures_) {
            GTEST_SKIP() << "NMS fixtures not found. Run tools/gen_test_vectors.py first.";
        }

        inputSize_ = 640;
        std::string infoPath = kFixtureDir + "preprocess_info.json";
        if (fs::exists(infoPath)) {
            std::ifstream f(infoPath);
            json j;
            f >> j;
            inputSize_ = j.value("input_size", 640);
            origH_ = j.value("original_h", 0);
            origW_ = j.value("original_w", 0);
        }
    }

    bool hasFixtures_ = false;
    int inputSize_ = 640;
    int origH_ = 0;
    int origW_ = 0;
};

#ifdef HAS_ONNXRUNTIME

TEST_F(LayoutNmsStandaloneTest, OnnxNmsProducesSameBoxes) {
    std::string onnxPath = std::string(PROJECT_ROOT_DIR) +
                           "/engine/model_files/layout/pp_doclayout_l_part2.onnx";
    if (!fs::exists(onnxPath)) {
        GTEST_SKIP() << "ONNX sub-model not found at " << onnxPath;
    }

    NpyArray dxOut0 = loadNpy(kFixtureDir + "dxnn_out_0.npy");
    NpyArray dxOut1 = loadNpy(kFixtureDir + "dxnn_out_1.npy");
    NpyArray sfArr  = loadNpy(kFixtureDir + "scale_factor.npy");

    std::vector<float> out0 = dxOut0.toFloatVector();
    std::vector<float> out1 = dxOut1.toFloatVector();
    const float* sfData = sfArr.asFloat32();

    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "nms_test");
    Ort::SessionOptions opts;
    opts.SetIntraOpNumThreads(1);
    opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    Ort::Session session(env, onnxPath.c_str(), opts);
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    float imShapeData[] = {static_cast<float>(inputSize_), static_cast<float>(inputSize_)};
    float scaleData[] = {sfData[0], sfData[1]};  // [scale_h, scale_w]
    int64_t shapeDims[] = {1, 2};

    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputs = session.GetInputCount();

    std::vector<const char*> inputNames;
    std::vector<Ort::AllocatedStringPtr> inputNamePtrs;
    for (size_t i = 0; i < numInputs; ++i) {
        auto n = session.GetInputNameAllocated(i, allocator);
        inputNames.push_back(n.get());
        inputNamePtrs.push_back(std::move(n));
    }

    size_t numOutputs = session.GetOutputCount();
    std::vector<const char*> outputNames;
    std::vector<Ort::AllocatedStringPtr> outputNamePtrs;
    for (size_t i = 0; i < numOutputs; ++i) {
        auto n = session.GetOutputNameAllocated(i, allocator);
        outputNames.push_back(n.get());
        outputNamePtrs.push_back(std::move(n));
    }

    auto getShape = [&](size_t idx) {
        auto ti = session.GetInputTypeInfo(idx);
        return ti.GetTensorTypeAndShapeInfo().GetShape();
    };

    auto fixDynamic = [](std::vector<int64_t>& shape, size_t dataSize) {
        int64_t known = 1;
        int dynIdx = -1;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] > 0) known *= shape[i];
            else dynIdx = static_cast<int>(i);
        }
        if (dynIdx >= 0 && known > 0)
            shape[dynIdx] = static_cast<int64_t>(dataSize) / known;
    };

    auto shape0 = getShape(0);
    auto shape1 = getShape(1);
    fixDynamic(shape0, out0.size());
    fixDynamic(shape1, out1.size());

    auto tensor0 = Ort::Value::CreateTensor<float>(memInfo, out0.data(), out0.size(),
                                                    shape0.data(), shape0.size());
    auto tensor1 = Ort::Value::CreateTensor<float>(memInfo, out1.data(), out1.size(),
                                                    shape1.data(), shape1.size());
    auto imShapeTensor = Ort::Value::CreateTensor<float>(memInfo, imShapeData, 2, shapeDims, 2);
    auto scaleTensor = Ort::Value::CreateTensor<float>(memInfo, scaleData, 2, shapeDims, 2);

    std::vector<Ort::Value> ortInputs;
    for (size_t i = 0; i < numInputs; ++i) {
        std::string name(inputNames[i]);
        if (name.find("concat") != std::string::npos) {
            ortInputs.push_back(std::move(tensor0));
        } else if (name.find("layer_norm") != std::string::npos) {
            ortInputs.push_back(std::move(tensor1));
        } else if (name == "im_shape") {
            ortInputs.push_back(std::move(imShapeTensor));
        } else if (name == "scale_factor") {
            ortInputs.push_back(std::move(scaleTensor));
        }
    }

    auto ortOutputs = session.Run(Ort::RunOptions{nullptr},
                                   inputNames.data(), ortInputs.data(), ortInputs.size(),
                                   outputNames.data(), outputNames.size());

    const float* boxData = ortOutputs[0].GetTensorData<float>();
    auto boxShape = ortOutputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int totalBoxes = static_cast<int>(boxShape[0]);

    EXPECT_EQ(totalBoxes, 300) << "ONNX should output 300 raw boxes for pp_doclayout";

    if (fs::exists(kFixtureDir + "onnx_raw_output.npy")) {
        NpyArray pyOnnxRaw = loadNpy(kFixtureDir + "onnx_raw_output.npy");
        const float* pyData = pyOnnxRaw.asFloat32();

        double maxDiff = 0.0;
        size_t totalElements = pyOnnxRaw.elementCount();
        for (size_t i = 0; i < totalElements; ++i) {
            double diff = std::abs(static_cast<double>(boxData[i]) - static_cast<double>(pyData[i]));
            maxDiff = std::max(maxDiff, diff);
        }

        EXPECT_LT(maxDiff, 1e-3)
            << "ONNX raw output max diff from Python: " << maxDiff;
    }

    std::string pyJsonStr = loadJsonString(kFixtureDir + "boxes.json");
    auto pyBoxes = json::parse(pyJsonStr);
    int pyBoxCount = static_cast<int>(pyBoxes.size());

    std::unordered_map<int, float> confThres = {
        {0,  0.3f}, {1,  0.5f}, {2,  0.4f}, {3,  0.5f}, {4,  0.5f}, {5,  0.5f},
        {6,  0.5f}, {7,  0.3f}, {8,  0.5f}, {9,  0.5f}, {10, 0.5f}, {11, 0.5f},
        {12, 0.5f}, {13, 0.5f}, {14, 0.5f}, {15, 0.5f}, {16, 0.45f},{17, 0.5f},
        {18, 0.5f}, {19, 0.5f}, {20, 0.5f}, {21, 0.5f}, {22, 0.5f},
    };

    int filteredCount = 0;
    for (int i = 0; i < totalBoxes; ++i) {
        const float* row = boxData + i * 6;
        int clsId = static_cast<int>(row[0]);
        float score = row[1];
        if (clsId < 0) continue;
        auto it = confThres.find(clsId);
        float threshold = (it != confThres.end()) ? it->second : 0.5f;
        if (score >= threshold) filteredCount++;
    }

    std::cerr << "  [INFO] ONNX raw: " << totalBoxes << " boxes, after conf filter: "
              << filteredCount << ", Python final: " << pyBoxCount << std::endl;

    EXPECT_EQ(pyBoxCount, 6) << "Expected 6 boxes for small_ocr_origin page 1";
}

#else

TEST_F(LayoutNmsStandaloneTest, SkippedNoOnnxRuntime) {
    GTEST_SKIP() << "ONNX Runtime not available — NMS standalone test skipped";
}

#endif // HAS_ONNXRUNTIME
