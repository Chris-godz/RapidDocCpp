#include "formula/formula_recognizer.h"
#include "common/logger.h"

#include <opencv2/opencv.hpp>
#include <fstream>
#include <algorithm>

#ifdef HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace rapid_doc {

struct FormulaRecognizer::Impl {
#ifdef HAS_ONNXRUNTIME
    std::unique_ptr<Ort::Env> ortEnv;
    std::unique_ptr<Ort::Session> session;
    std::vector<std::string> inputNames;
    std::vector<std::string> outputNames;
#endif
    std::vector<std::string> dict;
};

FormulaRecognizer::FormulaRecognizer(const FormulaRecognizerConfig& config)
    : impl_(std::make_unique<Impl>())
    , config_(config)
{
}
FormulaRecognizer::~FormulaRecognizer() = default;

// ============================================================================
// Initialize — load ONNX model and dictionary
// ============================================================================
bool FormulaRecognizer::initialize() {
    LOG_INFO("Initializing Formula Recognizer...");
    LOG_INFO("  ONNX model: {}", config_.onnxModelPath);
    LOG_INFO("  Dict path: {}", config_.dictPath);

#ifdef HAS_ONNXRUNTIME
    try {
        if (config_.onnxModelPath.empty()) {
            LOG_WARN("Formula ONNX model path is empty, recognition will be skipped.");
            return false;
        }

        impl_->ortEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "rapid_doc_formula");
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        opts.SetIntraOpNumThreads(1);

        impl_->session = std::make_unique<Ort::Session>(
            *impl_->ortEnv, config_.onnxModelPath.c_str(), opts);

        Ort::AllocatorWithDefaultOptions allocator;
        size_t numInputNodes = impl_->session->GetInputCount();
        for (size_t i = 0; i < numInputNodes; i++) {
            auto namePtr = impl_->session->GetInputNameAllocated(i, allocator);
            impl_->inputNames.push_back(namePtr.get());
        }
        size_t numOutputNodes = impl_->session->GetOutputCount();
        for (size_t i = 0; i < numOutputNodes; i++) {
            auto namePtr = impl_->session->GetOutputNameAllocated(i, allocator);
            impl_->outputNames.push_back(namePtr.get());
        }

        // Load dictionary
        if (!config_.dictPath.empty()) {
            std::ifstream dictFile(config_.dictPath);
            if (!dictFile.is_open()) {
                LOG_ERROR("Failed to open formula dict file: {}", config_.dictPath);
            } else {
                std::string line;
                while (std::getline(dictFile, line)) {
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    impl_->dict.push_back(line);
                }
                LOG_INFO("Loaded formula dictionary with {} entries", impl_->dict.size());
            }
        }

        initialized_ = true;
        LOG_INFO("Formula ONNX model loaded successfully");
        return true;
    } catch (const std::exception& e) {
        LOG_WARN("ONNX Runtime init failed for FormulaRecognizer: {}", e.what());
    }
#else
    LOG_WARN("ONNX Runtime not available, formula recognition will be skipped.");
#endif

    return false;
}

FormulaResult FormulaRecognizer::recognize(const cv::Mat& image) {
    FormulaResult result;
    result.success = false;

    if (!initialized_) {
        LOG_ERROR("Formula recognizer not initialized");
        return result;
    }

    if (image.empty()) {
        LOG_WARN("Empty formula image received");
        return result;
    }

#ifdef HAS_ONNXRUNTIME
    try {
        int out_h = 0;
        int out_w = 0;
        std::vector<float> inputTensorValues = preprocess(image, out_h, out_w);
        
        std::vector<int64_t> inputDims = {1, 1, out_h, out_w};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorValues.size(),
            inputDims.data(), inputDims.size());

        std::vector<const char*> inputNamesPtrs;
        for (const auto& s : impl_->inputNames) inputNamesPtrs.push_back(s.c_str());
        std::vector<const char*> outputNamesPtrs;
        for (const auto& s : impl_->outputNames) outputNamesPtrs.push_back(s.c_str());

        auto outputTensors = impl_->session->Run(
            Ort::RunOptions{nullptr},
            inputNamesPtrs.data(),
            &inputTensor,
            1,
            outputNamesPtrs.data(),
            outputNamesPtrs.size());

        if (!outputTensors.empty()) {
            // output is usually token IDs
            // Assume first output contains token ids: shape [batch, seq_len]
            const auto& tokenTensor = outputTensors[0];
            auto typeInfo = tokenTensor.GetTensorTypeAndShapeInfo();
            auto shape = typeInfo.GetShape();
            
            size_t numTokens = 1;
            for (auto dim : shape) {
                numTokens *= dim;
            }

            // check type
            auto type = typeInfo.GetElementType();
            std::vector<int64_t> tokens;
            
            if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
                const int64_t* data = tokenTensor.GetTensorData<int64_t>();
                tokens.assign(data, data + numTokens);
            } else if (type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
                const int32_t* data = tokenTensor.GetTensorData<int32_t>();
                tokens.assign(data, data + numTokens);
            }

            result.latex = postprocess(tokens);
            result.success = true;
            result.confidence = 1.0f; // placeholder
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Formula recognition failed: {}", e.what());
    }
#endif

    return result;
}

std::vector<float> FormulaRecognizer::preprocess(const cv::Mat& image, int& out_h, int& out_w) {
    // 1. Grayscale
    cv::Mat gray;
    if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    } else if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = image.clone();
    }

    // 2. Resize maintaining aspect ratio (up to max inputSize)
    int maxLen = config_.inputSize;
    float ratio = std::min(1.0f, static_cast<float>(maxLen) / std::max(gray.rows, gray.cols));
    int target_h = static_cast<int>(gray.rows * ratio);
    int target_w = static_cast<int>(gray.cols * ratio);

    cv::Mat resized;
    if (ratio < 1.0f) {
        cv::resize(gray, resized, cv::Size(target_w, target_h));
    } else {
        resized = gray.clone();
    }

    // 3. Pad to multiple of 16
    out_h = (target_h + 15) / 16 * 16;
    out_w = (target_w + 15) / 16 * 16;

    // Center padding
    int pad_top = (out_h - target_h) / 2;
    int pad_bottom = out_h - target_h - pad_top;
    int pad_left = (out_w - target_w) / 2;
    int pad_right = out_w - target_w - pad_left;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, cv::Scalar(255));

    // 4. Normalize (mean/std)
    padded.convertTo(padded, CV_32FC1, 1.0f / 255.0f);
    
    // using generic mean/std for grayscale
    const float mean = 0.5f;
    const float std_val = 0.5f;
    padded = (padded - mean) / std_val;

    // 5. To flat vector
    std::vector<float> tensorValues(out_h * out_w);
    if (padded.isContinuous()) {
        std::memcpy(tensorValues.data(), padded.ptr<float>(), out_h * out_w * sizeof(float));
    } else {
        for (int i = 0; i < out_h; ++i) {
            std::memcpy(&tensorValues[i * out_w], padded.ptr<float>(i), out_w * sizeof(float));
        }
    }

    return tensorValues;
}

std::string FormulaRecognizer::postprocess(const std::vector<int64_t>& tokens) {
    if (impl_->dict.empty()) {
        LOG_WARN("Dictionary is empty, returning raw tokens");
        std::string res;
        for (auto t : tokens) res += std::to_string(t) + " ";
        return res;
    }

    std::string latex;
    // Simple decoding
    // skip 0 (pad) and possibly bos/eos tokens (often 1 or 2, depends on dict)
    for (int64_t t : tokens) {
        if (t < 0 || static_cast<size_t>(t) >= impl_->dict.size()) continue;
        
        std::string tokenStr = impl_->dict[t];
        // skip special tokens
        if (tokenStr == "<pad>" || tokenStr == "<s>" || tokenStr == "</s>" || tokenStr == "<unk>") {
            continue;
        }
        latex += tokenStr;
    }

    // A basic cleanup of spaces (in Python they often do extra standardizations)
    return latex;
}

} // namespace rapid_doc
