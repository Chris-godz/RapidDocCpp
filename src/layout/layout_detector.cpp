/**
 * @file layout_detector.cpp
 * @brief Layout detection implementation (DEEPX NPU + ONNX RT post-processing)
 *
 * Aligned with Python:
 *   RapidDoc/rapid_doc/model/layout/rapid_layout_self/inference_engine/dxengine.py
 *   RapidDoc/rapid_doc/model/layout/rapid_layout_self/model_handler/pp_doclayout/pre_process.py
 *   RapidDoc/rapid_doc/model/layout/rapid_layout_self/model_handler/pp_doclayout/post_process.py
 */

#include "layout/layout_detector.h"
#include "common/logger.h"

#include <dxrt/inference_engine.h>

#ifdef HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include <algorithm>
#include <numeric>
#include <cmath>
#include <chrono>

namespace rapid_doc {

// ---------------------------------------------------------------------------
// DXEngine label list (matches Python PPDocLayoutModelHandler for DXENGINE)
// ---------------------------------------------------------------------------
static const std::vector<std::string> kDxEngineLabels = {
    "paragraph_title", "image", "text", "number", "abstract", "content",
    "figure_title", "formula", "table", "table_title", "reference",
    "doc_title", "footnote", "header", "algorithm", "footer", "seal",
    "chart_title", "chart", "formula_number", "header_image",
    "footer_image", "aside_text"
};

// Per-category confidence thresholds (Python DXENGINE branch)
static const std::unordered_map<int, float> kPerCategoryConfThres = {
    {0,  0.3f},  // paragraph_title
    {1,  0.5f},  // image
    {2,  0.4f},  // text
    {3,  0.5f},  // number
    {4,  0.5f},  // abstract
    {5,  0.5f},  // content
    {6,  0.5f},  // figure_title
    {7,  0.3f},  // formula
    {8,  0.5f},  // table
    {9,  0.5f},  // table_title
    {10, 0.5f},  // reference
    {11, 0.5f},  // doc_title
    {12, 0.5f},  // footnote
    {13, 0.5f},  // header
    {14, 0.5f},  // algorithm
    {15, 0.5f},  // footer
    {16, 0.45f}, // seal
    {17, 0.5f},  // chart_title
    {18, 0.5f},  // chart
    {19, 0.5f},  // formula_number
    {20, 0.5f},  // header_image
    {21, 0.5f},  // footer_image
    {22, 0.5f},  // aside_text
};

// Map DXEngine label string -> LayoutCategory enum
static LayoutCategory labelToCategory(const std::string& label) {
    if (label == "text" || label == "content" || label == "reference" ||
        label == "footnote" || label == "number" || label == "abstract" ||
        label == "aside_text")
        return LayoutCategory::TEXT;
    if (label == "paragraph_title" || label == "doc_title" || label == "chart_title")
        return LayoutCategory::TITLE;
    if (label == "image" || label == "header_image" || label == "footer_image" || label == "chart")
        return LayoutCategory::FIGURE;
    if (label == "figure_title")
        return LayoutCategory::FIGURE_CAPTION;
    if (label == "table")
        return LayoutCategory::TABLE;
    if (label == "table_title")
        return LayoutCategory::TABLE_CAPTION;
    if (label == "header")
        return LayoutCategory::HEADER;
    if (label == "footer")
        return LayoutCategory::FOOTER;
    if (label == "formula" || label == "formula_number")
        return LayoutCategory::EQUATION;
    if (label == "algorithm" || label == "seal")
        return LayoutCategory::STAMP;
    if (label == "code")
        return LayoutCategory::CODE;
    return LayoutCategory::UNKNOWN;
}

// ---------------------------------------------------------------------------
// IoU helpers (matches Python post_process.py iou())
// ---------------------------------------------------------------------------
static float computeIoU(const float* a, const float* b) {
    float x1 = std::max(a[0], b[0]);
    float y1 = std::max(a[1], b[1]);
    float x2 = std::min(a[2], b[2]);
    float y2 = std::min(a[3], b[3]);
    float interArea = std::max(0.0f, x2 - x1 + 1) * std::max(0.0f, y2 - y1 + 1);
    float area1 = (a[2] - a[0] + 1) * (a[3] - a[1] + 1);
    float area2 = (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
    return interArea / (area1 + area2 - interArea);
}

// NMS with different IoU thresholds for same/different class (Python nms())
static std::vector<int> nms(const std::vector<std::vector<float>>& boxes,
                            float iouSame = 0.6f, float iouDiff = 0.98f) {
    std::vector<int> indices(boxes.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        return boxes[a][1] > boxes[b][1];
    });

    std::vector<int> selected;
    while (!indices.empty()) {
        int current = indices.front();
        selected.push_back(current);
        const auto& curBox = boxes[current];
        float curClass = curBox[0];
        const float* curCoords = curBox.data() + 2;

        std::vector<int> remaining;
        for (size_t i = 1; i < indices.size(); ++i) {
            int idx = indices[i];
            const auto& box = boxes[idx];
            float threshold = (box[0] == curClass) ? iouSame : iouDiff;
            if (computeIoU(curCoords, box.data() + 2) < threshold) {
                remaining.push_back(idx);
            }
        }
        indices = std::move(remaining);
    }
    return selected;
}

// ---------------------------------------------------------------------------
// Pimpl
// ---------------------------------------------------------------------------
struct LayoutDetector::Impl {
    std::unique_ptr<dxrt::InferenceEngine> dxEngine;

#ifdef HAS_ONNXRUNTIME
    Ort::Env ortEnv{ORT_LOGGING_LEVEL_WARNING, "layout_nms"};
    std::unique_ptr<Ort::Session> ortSession;
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
#endif
};

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
LayoutDetector::LayoutDetector(const LayoutDetectorConfig& config)
    : config_(config)
    , impl_(std::make_unique<Impl>())
{
}

LayoutDetector::~LayoutDetector() = default;

// ---------------------------------------------------------------------------
// Initialization — load DX Engine + ONNX Runtime session
// ---------------------------------------------------------------------------
bool LayoutDetector::initialize() {
    LOG_INFO("Initializing Layout detector...");
    LOG_INFO("  DXNN model: {}", config_.dxnnModelPath);
    LOG_INFO("  ONNX sub-model: {}", config_.onnxSubModelPath);

    try {
        impl_->dxEngine = std::make_unique<dxrt::InferenceEngine>(config_.dxnnModelPath);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to load DX Engine model: {}", e.what());
        return false;
    }

#ifdef HAS_ONNXRUNTIME
    try {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        impl_->ortSession = std::make_unique<Ort::Session>(
            impl_->ortEnv, config_.onnxSubModelPath.c_str(), opts);
    } catch (const Ort::Exception& e) {
        LOG_ERROR("Failed to load ONNX sub-model: {}", e.what());
        return false;
    }
#else
    LOG_WARN("ONNX Runtime not available — post-processing will be skipped");
#endif

    initialized_ = true;
    LOG_INFO("Layout detector initialized successfully");
    return true;
}

// ---------------------------------------------------------------------------
// Preprocessing — matches Python PPPreProcess for DXENGINE path
//   resize to (inputSize, inputSize) with INTER_CUBIC
//   keep uint8, NHWC, no normalization
// ---------------------------------------------------------------------------
cv::Mat LayoutDetector::preprocess(const cv::Mat& image, cv::Point2f& scaleFactor) {
    int targetH = config_.inputSize;
    int targetW = config_.inputSize;

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(targetW, targetH), 0, 0, cv::INTER_CUBIC);

    // scale_factor = [target_h / orig_h, target_w / orig_w]
    scaleFactor.x = static_cast<float>(targetW) / image.cols;  // w_scale
    scaleFactor.y = static_cast<float>(targetH) / image.rows;  // h_scale

    return resized;
}

// ---------------------------------------------------------------------------
// Post-processing — matches Python PPPostProcess + PPDocLayoutModelHandler
//   1. Per-category confidence filter
//   2. NMS (iou_same=0.6, iou_diff=0.98)
//   3. Large image-box filter (area_thres 0.82/0.93)
//   4. Coordinate clamping & LayoutCategory mapping
// ---------------------------------------------------------------------------
std::vector<LayoutBox> LayoutDetector::postprocess(
    const std::vector<std::vector<float>>& dxOutputs,
    const cv::Size& imShape,
    const cv::Point2f& scaleFactor)
{
#ifndef HAS_ONNXRUNTIME
    LOG_WARN("ONNX Runtime not available — cannot run NMS post-processing");
    return {};
#else
    if (!impl_->ortSession) return {};

    // ---- Run ONNX sub-model ----
    // dxOutputs[0] and dxOutputs[1] are the two DX Engine output tensors.
    // We build the ONNX feed dict with 4 inputs.
    int64_t inputH = config_.inputSize;
    int64_t inputW = config_.inputSize;

    // im_shape = [[inputH, inputW]]  float32
    std::vector<float> imShapeData = {static_cast<float>(inputH),
                                      static_cast<float>(inputW)};
    std::vector<int64_t> imShapeDims = {1, 2};

    // scale_factor = [[h_scale, w_scale]]
    std::vector<float> scaleData = {scaleFactor.y, scaleFactor.x};
    std::vector<int64_t> scaleDims = {1, 2};

    auto imShapeTensor = Ort::Value::CreateTensor<float>(
        impl_->memInfo, imShapeData.data(), imShapeData.size(),
        imShapeDims.data(), imShapeDims.size());
    auto scaleTensor = Ort::Value::CreateTensor<float>(
        impl_->memInfo, scaleData.data(), scaleData.size(),
        scaleDims.data(), scaleDims.size());

    // Determine DX output tensor shapes from the ONNX model input metadata
    Ort::AllocatorWithDefaultOptions allocator;
    size_t numInputs = impl_->ortSession->GetInputCount();
    std::vector<const char*> inputNames;
    std::vector<Ort::AllocatedStringPtr> inputNamePtrs;
    for (size_t i = 0; i < numInputs; ++i) {
        auto namePtr = impl_->ortSession->GetInputNameAllocated(i, allocator);
        inputNames.push_back(namePtr.get());
        inputNamePtrs.push_back(std::move(namePtr));
    }

    size_t numOutputs = impl_->ortSession->GetOutputCount();
    std::vector<const char*> outputNames;
    std::vector<Ort::AllocatedStringPtr> outputNamePtrs;
    for (size_t i = 0; i < numOutputs; ++i) {
        auto namePtr = impl_->ortSession->GetOutputNameAllocated(i, allocator);
        outputNames.push_back(namePtr.get());
        outputNamePtrs.push_back(std::move(namePtr));
    }

    // Build tensors for DX outputs, matching the ONNX input names
    // We need mutable copies because CreateTensor takes non-const pointers
    std::vector<float> dxOut0 = dxOutputs[0];
    std::vector<float> dxOut1 = dxOutputs[1];

    // Get expected shapes from the ONNX model
    auto getInputShape = [&](size_t idx) -> std::vector<int64_t> {
        auto typeInfo = impl_->ortSession->GetInputTypeInfo(idx);
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        return tensorInfo.GetShape();
    };

    std::vector<int64_t> shape0 = getInputShape(0);
    std::vector<int64_t> shape1 = getInputShape(1);

    // Fix dynamic dimensions (-1) based on actual data size
    auto fixDynamic = [](std::vector<int64_t>& shape, size_t dataSize) {
        int64_t known = 1;
        int dynIdx = -1;
        for (size_t i = 0; i < shape.size(); ++i) {
            if (shape[i] > 0) known *= shape[i];
            else dynIdx = static_cast<int>(i);
        }
        if (dynIdx >= 0 && known > 0) {
            shape[dynIdx] = static_cast<int64_t>(dataSize) / known;
        }
    };

    fixDynamic(shape0, dxOut0.size());
    fixDynamic(shape1, dxOut1.size());

    auto dxTensor0 = Ort::Value::CreateTensor<float>(
        impl_->memInfo, dxOut0.data(), dxOut0.size(),
        shape0.data(), shape0.size());
    auto dxTensor1 = Ort::Value::CreateTensor<float>(
        impl_->memInfo, dxOut1.data(), dxOut1.size(),
        shape1.data(), shape1.size());

    // Arrange tensors in the order expected by the ONNX model
    std::vector<Ort::Value> ortInputs;
    for (size_t i = 0; i < numInputs; ++i) {
        std::string name(inputNames[i]);
        if (name.find("concat") != std::string::npos) {
            ortInputs.push_back(std::move(dxTensor0));
        } else if (name.find("layer_norm") != std::string::npos) {
            ortInputs.push_back(std::move(dxTensor1));
        } else if (name == "im_shape") {
            ortInputs.push_back(std::move(imShapeTensor));
        } else if (name == "scale_factor") {
            ortInputs.push_back(std::move(scaleTensor));
        }
    }

    auto ortOutputs = impl_->ortSession->Run(
        Ort::RunOptions{nullptr},
        inputNames.data(), ortInputs.data(), ortInputs.size(),
        outputNames.data(), outputNames.size());

    // ---- Parse ONNX output: boxes and box_num ----
    // Output format follows PaddleDetection: pred[0] = boxes, pred[1] = box_nums
    // boxes: [N, 6]  each row = [cls_id, score, xmin, ymin, xmax, ymax]
    const float* boxData = ortOutputs[0].GetTensorData<float>();
    auto boxShape = ortOutputs[0].GetTensorTypeAndShapeInfo().GetShape();
    int totalBoxes = static_cast<int>(boxShape[0]);

    // Collect raw boxes as vector of 6-element vectors
    std::vector<std::vector<float>> rawBoxes;
    rawBoxes.reserve(totalBoxes);
    for (int i = 0; i < totalBoxes; ++i) {
        const float* row = boxData + i * 6;
        int clsId = static_cast<int>(row[0]);
        float score = row[1];
        if (clsId < 0) continue;

        auto it = kPerCategoryConfThres.find(clsId);
        float threshold = (it != kPerCategoryConfThres.end()) ? it->second : 0.5f;
        if (score < threshold) continue;

        rawBoxes.push_back({row[0], row[1], row[2], row[3], row[4], row[5]});
    }

    // ---- NMS (same-class IoU=0.6, diff-class IoU=0.98) ----
    auto selected = nms(rawBoxes, 0.6f, 0.98f);

    // ---- Large image box filter ----
    float imgW = static_cast<float>(imShape.width);
    float imgH = static_cast<float>(imShape.height);
    float imgArea = imgW * imgH;
    float areaThreshold = (imgH > imgW) ? 0.82f : 0.93f;

    int imageClsIdx = -1;
    for (size_t i = 0; i < kDxEngineLabels.size(); ++i) {
        if (kDxEngineLabels[i] == "image") {
            imageClsIdx = static_cast<int>(i);
            break;
        }
    }

    std::vector<std::vector<float>> filteredBoxes;
    for (int idx : selected) {
        const auto& box = rawBoxes[idx];
        int clsId = static_cast<int>(box[0]);
        if (clsId == imageClsIdx && selected.size() > 1) {
            float xmin = std::max(0.0f, box[2]);
            float ymin = std::max(0.0f, box[3]);
            float xmax = std::min(imgW, box[4]);
            float ymax = std::min(imgH, box[5]);
            float boxArea = (xmax - xmin) * (ymax - ymin);
            if (boxArea > areaThreshold * imgArea) continue;
        }
        filteredBoxes.push_back(box);
    }
    if (filteredBoxes.empty() && !selected.empty()) {
        for (int idx : selected)
            filteredBoxes.push_back(rawBoxes[idx]);
    }

    // ---- Map to LayoutBox ----
    std::vector<LayoutBox> result;
    result.reserve(filteredBoxes.size());
    for (size_t i = 0; i < filteredBoxes.size(); ++i) {
        const auto& box = filteredBoxes[i];
        int clsId = static_cast<int>(box[0]);
        float score = box[1];
        float xmin = std::max(0.0f, box[2]);
        float ymin = std::max(0.0f, box[3]);
        float xmax = std::min(imgW, box[4]);
        float ymax = std::min(imgH, box[5]);
        if (xmax <= xmin || ymax <= ymin) continue;

        std::string label = (clsId >= 0 && clsId < static_cast<int>(kDxEngineLabels.size()))
                            ? kDxEngineLabels[clsId]
                            : "unknown";

        LayoutBox lb;
        lb.x0 = xmin;
        lb.y0 = ymin;
        lb.x1 = xmax;
        lb.y1 = ymax;
        lb.category = labelToCategory(label);
        lb.confidence = score;
        lb.index = static_cast<int>(i);
        lb.clsId = clsId;
        lb.label = label;
        result.push_back(lb);
    }

    return result;
#endif // HAS_ONNXRUNTIME
}

// ---------------------------------------------------------------------------
// Detect — full pipeline
// ---------------------------------------------------------------------------
LayoutResult LayoutDetector::detect(const cv::Mat& image) {
    LayoutResult result;

    if (!initialized_) {
        LOG_ERROR("Layout detector not initialized");
        return result;
    }

    auto tStart = std::chrono::steady_clock::now();

    // 1. Preprocess
    cv::Point2f scaleFactor;
    cv::Mat preprocessed = preprocess(image, scaleFactor);

    // 2. DX Engine inference — input is NHWC uint8
    // DX Engine Run() takes void* pointing to the raw data buffer.
    auto dxRawOutputs = impl_->dxEngine->Run(
        static_cast<void*>(preprocessed.data));

    // Convert raw DX outputs to float vectors
    std::vector<std::vector<float>> dxOutputs;
    for (auto& outPtr : dxRawOutputs) {
        const float* ptr = reinterpret_cast<const float*>(outPtr->data());
        size_t count = 1;
        for (auto d : outPtr->shape()) count *= d;
        dxOutputs.emplace_back(ptr, ptr + count);
    }

    // 3. Post-process (ONNX NMS + category mapping)
    cv::Size origShape(image.cols, image.rows);
    result.boxes = postprocess(dxOutputs, origShape, scaleFactor);

    auto tEnd = std::chrono::steady_clock::now();
    result.inferenceTimeMs =
        std::chrono::duration<double, std::milli>(tEnd - tStart).count();

    LOG_INFO("Layout detection: {} boxes in {:.1f}ms",
             result.boxes.size(), result.inferenceTimeMs);

    return result;
}

// ---------------------------------------------------------------------------
// Async detect — runs synchronously for now, async DX Engine integration later
// ---------------------------------------------------------------------------
void LayoutDetector::detectAsync(const cv::Mat& image, DetectionCallback callback) {
    auto result = detect(image);
    if (callback) {
        callback(result);
    }
}

} // namespace rapid_doc
