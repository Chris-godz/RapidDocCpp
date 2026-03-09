/**
 * @file layout_detector.cpp
 * @brief Layout detection implementation (DEEPX NPU + ONNX RT post-processing)
 * 
 * Reference: RapidDoc/rapid_doc/model/layout/rapid_layout_self/inference_engine/dxengine.py
 * 
 * Architecture:
 *   1. Preprocess: direct resize to target size → NHWC uint8 tensor (no normalize for DXEngine)
 *   2. DX Engine inference: produces two intermediate tensors
 *   3. ONNX RT post-processing (required): NMS + bbox decode with im_shape and scale_factor
 *   4. Map category IDs to LayoutCategory enum
 */

#include "layout/layout_detector.h"
#include "common/logger.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <thread>
#include <future>

#include <dxrt/inference_engine.h>

#ifdef HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace rapid_doc {

namespace {

static LayoutCategory categoryFromId(int classId) {
    // PP-DocLayout-L DXEngine: 23 categories, indices 0-22
    if (classId >= static_cast<int>(LayoutCategory::PARAGRAPH_TITLE) &&
        classId <= static_cast<int>(LayoutCategory::ASIDE_TEXT)) {
        return static_cast<LayoutCategory>(classId);
    }
    return LayoutCategory::UNKNOWN;
}

static size_t elementCount(const std::vector<int64_t>& shape) {
    if (shape.empty()) {
        return 0;
    }
    size_t count = 1;
    for (int64_t dim : shape) {
        if (dim <= 0) {
            return 0;
        }
        count *= static_cast<size_t>(dim);
    }
    return count;
}

} // namespace

struct LayoutDetector::Impl {
    std::unique_ptr<dxrt::InferenceEngine> engine;

#ifdef HAS_ONNXRUNTIME
    std::unique_ptr<Ort::Env> ortEnv;
    std::unique_ptr<Ort::Session> ortSession;
    std::vector<std::string> ortInputNames;
    std::vector<std::string> ortOutputNames;
#endif
};

LayoutDetector::LayoutDetector(const LayoutDetectorConfig& config)
    : impl_(std::make_unique<Impl>())
    , config_(config)
{
}

LayoutDetector::~LayoutDetector() = default;

bool LayoutDetector::initialize() {
    LOG_INFO("Initializing Layout detector...");
    LOG_INFO("  DXNN model: {}", config_.dxnnModelPath);
    LOG_INFO("  ONNX sub-model: {}", config_.onnxSubModelPath);

    if (config_.dxnnModelPath.empty() || !std::filesystem::exists(config_.dxnnModelPath)) {
        LOG_ERROR("DXNN model not found: {}", config_.dxnnModelPath);
        initialized_ = false;
        return false;
    }

    try {
        impl_->engine = std::make_unique<dxrt::InferenceEngine>(config_.dxnnModelPath);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to initialize DXRT inference engine: {}", e.what());
        initialized_ = false;
        return false;
    }

#ifdef HAS_ONNXRUNTIME
    if (config_.onnxSubModelPath.empty() || !std::filesystem::exists(config_.onnxSubModelPath)) {
        LOG_ERROR("ONNX post model not found: {}", config_.onnxSubModelPath);
        initialized_ = false;
        return false;
    }

    try {
        impl_->ortEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "rapid_doc_layout");
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        opts.SetIntraOpNumThreads(1);

        impl_->ortSession = std::make_unique<Ort::Session>(
            *impl_->ortEnv,
            config_.onnxSubModelPath.c_str(),
            opts
        );

        Ort::AllocatorWithDefaultOptions allocator;
        size_t inputCount = impl_->ortSession->GetInputCount();
        size_t outputCount = impl_->ortSession->GetOutputCount();

        impl_->ortInputNames.clear();
        impl_->ortOutputNames.clear();
        impl_->ortInputNames.reserve(inputCount);
        impl_->ortOutputNames.reserve(outputCount);

        for (size_t i = 0; i < inputCount; ++i) {
            auto name = impl_->ortSession->GetInputNameAllocated(i, allocator);
            impl_->ortInputNames.emplace_back(name.get());
        }
        for (size_t i = 0; i < outputCount; ++i) {
            auto name = impl_->ortSession->GetOutputNameAllocated(i, allocator);
            impl_->ortOutputNames.emplace_back(name.get());
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to initialize ONNX post-processing session: {}", e.what());
        initialized_ = false;
        return false;
    }
#else
    LOG_WARN("ONNX Runtime not enabled at build time; layout post-process unavailable");
#endif

    initialized_ = true;
    return true;
}

LayoutResult LayoutDetector::detect(const cv::Mat& image) {
    LayoutResult result;

    if (!initialized_) {
        LOG_ERROR("Layout detector not initialized");
        return result;
    }

    if (image.empty()) {
        LOG_WARN("Layout detection received empty image");
        return result;
    }

    if (!impl_->engine) {
        LOG_ERROR("DXRT engine is null");
        return result;
    }

    LOG_INFO("Layout detection: image {}x{}", image.cols, image.rows);

    auto t0 = std::chrono::steady_clock::now();

    cv::Point2f scaleFactor(1.0f, 1.0f);
    cv::Mat inputTensor = preprocess(image, scaleFactor);
    if (inputTensor.empty()) {
        LOG_ERROR("Layout preprocess failed");
        return result;
    }

    if (!inputTensor.isContinuous()) {
        inputTensor = inputTensor.clone();
    }

    std::vector<std::vector<float>> dxOutputs;
    std::vector<std::vector<int64_t>> dxOutputShapes;

    try {
        dxrt::TensorPtrs outputs = impl_->engine->Run(inputTensor.data);
        dxOutputs.reserve(outputs.size());
        dxOutputShapes.reserve(outputs.size());

        for (const auto& tensor : outputs) {
            if (!tensor) {
                continue;
            }

            const std::vector<int64_t>& shape = tensor->shape();
            size_t count = elementCount(shape);
            if (count == 0 || tensor->data() == nullptr) {
                continue;
            }

            if (tensor->elem_size() != sizeof(float)) {
                LOG_WARN("Unexpected DX output elem_size={} (expected 4), skipping tensor {}",
                         tensor->elem_size(), tensor->name());
                continue;
            }

            const float* ptr = static_cast<const float*>(tensor->data());
            dxOutputs.emplace_back(ptr, ptr + count);
            dxOutputShapes.push_back(shape);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("DXRT inference failed: {}", e.what());
        return result;
    }

    if (dxOutputs.empty()) {
        LOG_WARN("DXRT produced no usable outputs");
        return result;
    }

    result.boxes = postprocess(dxOutputs, dxOutputShapes, image.size(), scaleFactor);

    auto t1 = std::chrono::steady_clock::now();
    result.inferenceTimeMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

    LOG_INFO("Layout detection done: {} boxes, {:.2f} ms", result.boxes.size(), result.inferenceTimeMs);
    return result;
}

void LayoutDetector::detectAsync(const cv::Mat& image, DetectionCallback callback) {
    if (!callback) {
        return;
    }

    if (!config_.useAsync) {
        callback(detect(image));
        return;
    }

    cv::Mat imageCopy = image.clone();
    std::thread([this, imageCopy = std::move(imageCopy), callback = std::move(callback)]() mutable {
        callback(detect(imageCopy));
    }).detach();
}

cv::Mat LayoutDetector::preprocess(const cv::Mat& image, cv::Point2f& scaleFactor) {
    if (image.empty() || config_.inputSize <= 0) {
        scaleFactor = cv::Point2f(1.0f, 1.0f);
        return {};
    }

    int srcW = image.cols;
    int srcH = image.rows;
    int dstSize = config_.inputSize;

    // Direct resize (stretch to target).
    // No aspect-ratio preservation; the model was trained on directly resized inputs.
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(dstSize, dstSize), 0, 0, cv::INTER_CUBIC);

    // Independent per-axis scale factors: (W_scale, H_scale)
    scaleFactor = cv::Point2f(
        static_cast<float>(dstSize) / static_cast<float>(std::max(srcW, 1)),
        static_cast<float>(dstSize) / static_cast<float>(std::max(srcH, 1))
    );

    return resized;
}

std::vector<LayoutBox> LayoutDetector::postprocess(
    const std::vector<std::vector<float>>& dxOutputs,
    const std::vector<std::vector<int64_t>>& dxOutputShapes,
    const cv::Size& imShape,
    const cv::Point2f& scaleFactor)
{
    // Parse decoded bounding boxes from a flat [N, stride] tensor.
    // scaleToOriginal: if true, rescale coordinates from model-input space to original-image space.
    //   - ONNX post-model already decodes to original coords → pass false
    //   - Raw/debug path (not normally used) → pass true
    auto parseBoxes = [&](const float* data, size_t elemCount, int64_t stride, bool scaleToOriginal) {
        std::vector<LayoutBox> boxes;
        if (data == nullptr || elemCount == 0) {
            return boxes;
        }

        int64_t step = stride >= 6 ? stride : 6;
        size_t rows = elemCount / static_cast<size_t>(step);

        boxes.reserve(rows);
        for (size_t i = 0; i < rows; ++i) {
            const float* row = data + i * step;
            int classId = static_cast<int>(std::round(row[0]));
            float conf = row[1];
            if (conf < config_.confThreshold) {
                continue;
            }

            float x0 = row[2];
            float y0 = row[3];
            float x1 = row[4];
            float y1 = row[5];

            if (scaleToOriginal && scaleFactor.x > 0.0f && scaleFactor.y > 0.0f) {
                x0 /= scaleFactor.x;
                x1 /= scaleFactor.x;
                y0 /= scaleFactor.y;
                y1 /= scaleFactor.y;
            }

            x0 = std::clamp(x0, 0.0f, static_cast<float>(imShape.width - 1));
            y0 = std::clamp(y0, 0.0f, static_cast<float>(imShape.height - 1));
            x1 = std::clamp(x1, 0.0f, static_cast<float>(imShape.width - 1));
            y1 = std::clamp(y1, 0.0f, static_cast<float>(imShape.height - 1));

            if (x1 <= x0 || y1 <= y0) {
                continue;
            }

            boxes.push_back(LayoutBox{
                x0, y0, x1, y1,
                categoryFromId(classId),
                conf,
                static_cast<int>(i)
            });
        }
        return boxes;
    };

#ifdef HAS_ONNXRUNTIME
    if (impl_->ortSession && !impl_->ortInputNames.empty() && !impl_->ortOutputNames.empty()) {
        try {
            Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            // im_shape must be the resized (model input) dimensions, not the original image.
            std::vector<float> imShapeTensor = {
                static_cast<float>(config_.inputSize),
                static_cast<float>(config_.inputSize)
            };
            // scale_factor: [H_scale, W_scale] = [dstSize/srcH, dstSize/srcW]
            std::vector<float> scaleTensor = {
                scaleFactor.y,  // H_scale = dstSize / srcH
                scaleFactor.x   // W_scale = dstSize / srcW
            };

            std::vector<Ort::Value> inputValues;
            inputValues.reserve(impl_->ortInputNames.size());

            size_t dxIndex = 0;
            for (const std::string& inputName : impl_->ortInputNames) {
                if (inputName.find("im_shape") != std::string::npos) {
                    const int64_t shape[2] = {1, 2};
                    inputValues.emplace_back(Ort::Value::CreateTensor<float>(
                        memInfo,
                        imShapeTensor.data(),
                        imShapeTensor.size(),
                        shape,
                        2
                    ));
                    continue;
                }

                if (inputName.find("scale_factor") != std::string::npos) {
                    const int64_t shape[2] = {1, 2};
                    inputValues.emplace_back(Ort::Value::CreateTensor<float>(
                        memInfo,
                        scaleTensor.data(),
                        scaleTensor.size(),
                        shape,
                        2
                    ));
                    continue;
                }

                if (dxIndex >= dxOutputs.size()) {
                    LOG_WARN("Not enough DX outputs for ONNX input {}", inputName);
                    return {};
                }

                const auto& tensor = dxOutputs[dxIndex];
                std::vector<int64_t> shape;
                if (dxIndex < dxOutputShapes.size() && !dxOutputShapes[dxIndex].empty()) {
                    shape = dxOutputShapes[dxIndex];
                } else {
                    shape = {1, static_cast<int64_t>(tensor.size())};
                }

                inputValues.emplace_back(Ort::Value::CreateTensor<float>(
                    memInfo,
                    const_cast<float*>(tensor.data()),
                    tensor.size(),
                    shape.data(),
                    shape.size()
                ));
                ++dxIndex;
            }

            std::vector<const char*> inputNames;
            std::vector<const char*> outputNames;
            inputNames.reserve(impl_->ortInputNames.size());
            outputNames.reserve(impl_->ortOutputNames.size());

            for (const auto& n : impl_->ortInputNames) inputNames.push_back(n.c_str());
            for (const auto& n : impl_->ortOutputNames) outputNames.push_back(n.c_str());

            auto outputs = impl_->ortSession->Run(
                Ort::RunOptions{nullptr},
                inputNames.data(),
                inputValues.data(),
                inputValues.size(),
                outputNames.data(),
                outputNames.size()
            );

            if (!outputs.empty() && outputs[0].IsTensor()) {
                auto info = outputs[0].GetTensorTypeAndShapeInfo();
                auto shape = info.GetShape();
                size_t count = info.GetElementCount();
                const float* data = outputs[0].GetTensorData<float>();

                int64_t stride = 6;
                if (!shape.empty() && shape.back() >= 6) {
                    stride = shape.back();
                }

                // ONNX post-model outputs boxes in original-image coordinates
                // (it uses scale_factor internally to rescale), so no additional
                // scale inverse is needed.
                return parseBoxes(data, count, stride, /*scaleToOriginal=*/false);
            }
        } catch (const std::exception& e) {
            LOG_ERROR("ONNX postprocess failed: {}", e.what());
        }
    }
#endif

    // The DX engine outputs are intermediate feature tensors (e.g. p2o.pd_op.concat,
    // p2o.pd_op.layer_norm), NOT decoded detection boxes.  They MUST go through the
    // ONNX post-processing sub-model to produce valid bounding boxes.
    // Without ONNX Runtime, layout detection cannot produce results.
    LOG_ERROR("ONNX post-processing is required for layout detection but is unavailable");
    return {};
}

} // namespace rapid_doc
