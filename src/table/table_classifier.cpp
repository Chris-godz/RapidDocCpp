/**
 * @file table_classifier.cpp
 * @brief Table wired/wireless classifier (PaddleCls ONNX) — C++ port of the
 *        Python reference in
 *        RapidDoc/rapid_doc/model/table/rapid_table_self/table_cls/main.py.
 *
 * The preprocess is a byte-for-byte translation of PaddleCls.preprocess so the
 * argmax result is identical to the Python lane on equivalent BGR crops.
 */

#include "table/table_classifier.h"

#include "common/logger.h"

#ifdef HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace fs = std::filesystem;

namespace rapid_doc {

namespace {

// Python: self.mean / self.std in preprocess (applied to the already-BGR
// image; no channel swap is performed by LoadImage for np.ndarray input).
constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3]  = {0.229f, 0.224f, 0.225f};

constexpr int kInputH = 224;
constexpr int kInputW = 224;
constexpr int kResizeShort = 256;

} // namespace

struct TableClassifier::Impl {
#ifdef HAS_ONNXRUNTIME
    Ort::Env ortEnv{ORT_LOGGING_LEVEL_WARNING, "table_cls_onnx"};
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
#endif
    std::string inputName;
    std::string outputName;
};

TableClassifier::TableClassifier(const TableClassifierConfig& config)
    : config_(config)
    , impl_(std::make_unique<Impl>())
{
}

TableClassifier::~TableClassifier() = default;

bool TableClassifier::initialize() {
#ifndef HAS_ONNXRUNTIME
    LOG_ERROR("Table classifier requires ONNX Runtime, but HAS_ONNXRUNTIME is not enabled.");
    return false;
#else
    if (config_.onnxModelPath.empty()) {
        LOG_WARN("Table classifier model path is empty; classifier disabled.");
        return false;
    }
    if (!fs::exists(config_.onnxModelPath)) {
        LOG_WARN("Table classifier ONNX model not found: {} (classifier disabled)",
                 config_.onnxModelPath);
        return false;
    }

    try {
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (config_.disableCpuMemArena) {
            opts.DisableCpuMemArena();
        }
        if (config_.intraOpThreads > 0) {
            opts.SetIntraOpNumThreads(config_.intraOpThreads);
        }
        impl_->session = std::make_unique<Ort::Session>(
            impl_->ortEnv, config_.onnxModelPath.c_str(), opts);

        Ort::AllocatorWithDefaultOptions allocator;
        {
            auto inputName = impl_->session->GetInputNameAllocated(0, allocator);
            impl_->inputName = inputName.get();
        }
        {
            auto outputName = impl_->session->GetOutputNameAllocated(0, allocator);
            impl_->outputName = outputName.get();
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to initialize table classifier: {}", e.what());
        return false;
    }

    initialized_ = true;
    LOG_INFO("Table classifier initialized: {}", config_.onnxModelPath);
    return true;
#endif
}

std::vector<float> TableClassifier::preprocess(
    const cv::Mat& bgr,
    double& outPreprocessMs) const
{
    const auto start = std::chrono::steady_clock::now();

    // Python accepts any ndim/channel shape via LoadImage → BGR 3ch. We guard
    // against 1/4-channel inputs here for robustness; the pipeline always
    // hands us a BGR 3-channel cv::Mat.
    cv::Mat working = bgr;
    if (working.channels() == 1) {
        cv::cvtColor(working, working, cv::COLOR_GRAY2BGR);
    } else if (working.channels() == 4) {
        cv::cvtColor(working, working, cv::COLOR_BGRA2BGR);
    }

    const int imgH = working.rows;
    const int imgW = working.cols;
    if (imgH <= 0 || imgW <= 0) {
        outPreprocessMs = 0.0;
        return {};
    }

    // short resize: percent = resize_short / min(w, h)
    const double percent = static_cast<double>(kResizeShort) /
                           static_cast<double>(std::min(imgW, imgH));
    const int resizedW = static_cast<int>(std::lround(imgW * percent));
    const int resizedH = static_cast<int>(std::lround(imgH * percent));

    cv::Mat resized;
    cv::resize(working, resized, cv::Size(resizedW, resizedH), 0.0, 0.0, cv::INTER_LANCZOS4);

    // center crop
    const int wStart = (resizedW - kInputW) / 2;
    const int hStart = (resizedH - kInputH) / 2;
    if (wStart < 0 || hStart < 0 ||
        wStart + kInputW > resizedW || hStart + kInputH > resizedH) {
        const auto end = std::chrono::steady_clock::now();
        outPreprocessMs = std::chrono::duration<double, std::milli>(end - start).count();
        return {};
    }
    cv::Rect cropRect(wStart, hStart, kInputW, kInputH);
    cv::Mat cropped = resized(cropRect);

    // convert to float32, /255, -mean, /std, HWC→CHW, batch=1
    std::vector<float> tensor(static_cast<size_t>(1 * 3 * kInputH * kInputW));
    const size_t planeSize = static_cast<size_t>(kInputH * kInputW);

    // NOTE: Python applies mean/std in the channel order of the input array
    // (which for np.ndarray input is the original BGR memory from OpenCV).
    // We therefore index mean/std by the BGR channel index as well — no swap.
    for (int y = 0; y < kInputH; ++y) {
        const uint8_t* row = cropped.ptr<uint8_t>(y);
        for (int x = 0; x < kInputW; ++x) {
            const uint8_t b = row[3 * x + 0];
            const uint8_t g = row[3 * x + 1];
            const uint8_t r = row[3 * x + 2];
            const float vb = (static_cast<float>(b) / 255.0f - kMean[0]) / kStd[0];
            const float vg = (static_cast<float>(g) / 255.0f - kMean[1]) / kStd[1];
            const float vr = (static_cast<float>(r) / 255.0f - kMean[2]) / kStd[2];
            const size_t idx = static_cast<size_t>(y) * kInputW + static_cast<size_t>(x);
            tensor[0 * planeSize + idx] = vb; // channel 0 (B in BGR)
            tensor[1 * planeSize + idx] = vg; // channel 1 (G in BGR)
            tensor[2 * planeSize + idx] = vr; // channel 2 (R in BGR)
        }
    }

    const auto end = std::chrono::steady_clock::now();
    outPreprocessMs = std::chrono::duration<double, std::milli>(end - start).count();
    return tensor;
}

TableClassifyResult TableClassifier::classify(const cv::Mat& tableImageBgr) {
    TableClassifyResult out;
    out.label = "error";
    out.type = TableType::UNKNOWN;
    out.predIndex = -1;
    out.score = -1.0f;

#ifndef HAS_ONNXRUNTIME
    LOG_WARN("TableClassifier::classify called but ONNX Runtime is not enabled.");
    return out;
#else
    if (!initialized_ || !impl_->session) {
        out.label = "disabled";
        return out;
    }
    if (tableImageBgr.empty()) {
        return out;
    }

    try {
        double prepMs = 0.0;
        std::vector<float> input = preprocess(tableImageBgr, prepMs);
        out.preprocessMs = prepMs;
        if (input.empty()) {
            return out;
        }

        const std::array<int64_t, 4> inputShape{1, 3, kInputH, kInputW};
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            impl_->memInfo,
            input.data(),
            input.size(),
            inputShape.data(),
            inputShape.size());

        const char* inputNames[] = {impl_->inputName.c_str()};
        const char* outputNames[] = {impl_->outputName.c_str()};

        const auto inferStart = std::chrono::steady_clock::now();
        auto ortOutputs = impl_->session->Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1);
        const auto inferEnd = std::chrono::steady_clock::now();
        out.inferMs =
            std::chrono::duration<double, std::milli>(inferEnd - inferStart).count();

        if (ortOutputs.empty() || !ortOutputs.front().IsTensor()) {
            throw std::runtime_error("Table classifier output tensor missing");
        }

        const auto outInfo = ortOutputs.front().GetTensorTypeAndShapeInfo();
        const auto shape = outInfo.GetShape();
        if (shape.size() < 2 || shape.back() < 2) {
            throw std::runtime_error("Table classifier output shape unexpected");
        }

        const float* data = ortOutputs.front().GetTensorData<float>();
        const float logit0 = data[0];
        const float logit1 = data[1];
        out.rawLogitWired = logit0;
        out.rawLogitWireless = logit1;

        // softmax (2-class) for observability only
        const float maxL = std::max(logit0, logit1);
        const float e0 = std::exp(logit0 - maxL);
        const float e1 = std::exp(logit1 - maxL);
        const float sum = e0 + e1;
        const float p0 = sum > 0.0f ? e0 / sum : 0.0f;
        const float p1 = sum > 0.0f ? e1 / sum : 0.0f;

        if (logit0 >= logit1) {
            out.predIndex = 0;
            out.label = "wired";
            out.type = TableType::WIRED;
            out.score = p0;
        } else {
            out.predIndex = 1;
            out.label = "wireless";
            out.type = TableType::WIRELESS;
            out.score = p1;
        }
        out.ok = true;
        return out;
    } catch (const std::exception& e) {
        LOG_WARN("TableClassifier::classify failed: {}", e.what());
        out.ok = false;
        out.label = "error";
        out.type = TableType::UNKNOWN;
        return out;
    }
#endif
}

} // namespace rapid_doc
