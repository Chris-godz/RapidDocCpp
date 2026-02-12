/**
 * @file ocr_pipeline.h
 * @brief OCR Pipeline bridge header for DXNN-OCR-cpp integration
 * 
 * This header provides the interface between RapidDoc and DXNN-OCR-cpp.
 * The actual implementation is in the submodule at 3rd-party/DXNN-OCR-cpp.
 */

#pragma once

// Forward declarations and re-exports from DXNN-OCR-cpp
// When building, include the DXNN-OCR-cpp headers via the CMake include path

#include <opencv2/core.hpp>
#include <string>
#include <vector>
#include <memory>

namespace ocr {

/**
 * @brief OCR Detection result (single text box)
 */
struct TextBox {
    std::vector<cv::Point2f> points;  // 4 corner points
    float score;
};

/**
 * @brief OCR Recognition result (text + confidence)
 */
struct TextRecResult {
    std::string text;
    float confidence;
};

/**
 * @brief Combined OCR result for a text region
 */
struct OCRResult {
    TextBox box;
    TextRecResult recognition;
};

/**
 * @brief OCR Detector configuration
 */
struct DetectorConfig {
    std::string modelPath640;
    std::string modelPath960;
    int inputSize = 640;
    float boxThreshold = 0.5f;
    float boxScoreThreshold = 0.3f;
};

/**
 * @brief OCR Recognizer configuration
 */
struct RecognizerConfig {
    std::string modelDir;
    std::string dictPath;
    int maxTextLength = 32;
};

/**
 * @brief OCR Pipeline configuration
 */
struct OCRPipelineConfig {
    DetectorConfig detectorConfig;
    RecognizerConfig recognizerConfig;
    bool enableAngleClassifier = false;  // Not supported on DEEPX NPU
};

/**
 * @class OCRPipeline
 * @brief Combined detection + recognition pipeline
 * 
 * This is a forward declaration. The actual implementation comes from
 * DXNN-OCR-cpp via the submodule.
 */
class OCRPipeline {
public:
    explicit OCRPipeline(const OCRPipelineConfig& config);
    ~OCRPipeline();

    /**
     * @brief Initialize models and engines
     * @return true if successful
     */
    bool initialize();

    /**
     * @brief Process an image and return all detected text
     */
    std::vector<OCRResult> process(const cv::Mat& image);

    /**
     * @brief Process a single cropped text region (recognition only)
     */
    TextRecResult recognize(const cv::Mat& croppedText);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace ocr
