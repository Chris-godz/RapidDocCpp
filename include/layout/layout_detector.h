#pragma once

/**
 * @file layout_detector.h
 * @brief Layout detection using DEEPX NPU + ONNX Runtime post-processing
 * 
 * Architecture (matching Python dxengine.py):
 *   1. Main inference: dxrt::InferenceEngine loads .dxnn model (NPU-accelerated)
 *   2. Post-processing: ONNX Runtime Session loads .onnx sub-model (CPU, for NMS/bbox decode)
 * 
 * The DX Engine outputs two intermediate tensors which are fed into the ONNX
 * sub-model along with image shape and scale factor to produce final bounding boxes.
 * 
 * Input:  NHWC image tensor (uint8 or float32)
 * Output: LayoutResult with detected boxes and categories
 */

#include "common/types.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <functional>

namespace rapid_doc {

/**
 * @brief Layout detector configuration
 */
struct LayoutDetectorConfig {
    std::string dxnnModelPath;      // .dxnn model for DEEPX NPU
    std::string onnxSubModelPath;   // .onnx sub-model for post-processing (NMS)
    int inputSize = 800;            // Model input size (resize target)
    float confThreshold = 0.5f;     // Detection confidence threshold
    bool useAsync = false;          // Enable async inference
};

/**
 * @brief Layout detector using DEEPX NPU engine
 * 
 * Dual-engine design:
 *   - DX Engine: runs the main detection backbone on NPU
 *   - ONNX Runtime: runs NMS post-processing on CPU
 */
class LayoutDetector {
public:
    explicit LayoutDetector(const LayoutDetectorConfig& config);
    ~LayoutDetector();

    /**
     * @brief Initialize models (load DX engine + ONNX session)
     * @return true if successful
     */
    bool initialize();

    /**
     * @brief Detect layout elements in a page image
     * @param image Input page image (BGR, full resolution)
     * @return Layout detection result
     */
    LayoutResult detect(const cv::Mat& image);

    /**
     * @brief Async detection with callback
     * @param image Input page image
     * @param callback Called when detection completes
     */
    using DetectionCallback = std::function<void(const LayoutResult&)>;
    void detectAsync(const cv::Mat& image, DetectionCallback callback);

    /**
     * @brief Check if detector is initialized
     */
    bool isInitialized() const { return initialized_; }

private:
    /**
     * @brief Preprocess image for layout model
     * @param image Input BGR image
     * @param scaleFactor Output scale factor for post-processing
     * @return Preprocessed image tensor
     */
    cv::Mat preprocess(const cv::Mat& image, cv::Point2f& scaleFactor);

    /**
     * @brief Run ONNX post-processing (NMS + bbox decode)
     * @param dxOutputs Raw outputs from DX engine
     * @param imShape Original image shape (H, W)
     * @param scaleFactor Scale factor from preprocessing
     * @return Decoded layout boxes
     */
    std::vector<LayoutBox> postprocess(
        const std::vector<std::vector<float>>& dxOutputs,
        const cv::Size& imShape,
        const cv::Point2f& scaleFactor
    );

    struct Impl;
    std::unique_ptr<Impl> impl_;
    LayoutDetectorConfig config_;
    bool initialized_ = false;
};

} // namespace rapid_doc
