#pragma once

/**
 * @file formula_recognizer.h
 * @brief Formula recognition using ONNX Runtime
 * 
 * Implements pp_formulanet_plus_m.onnx integration for converting cropped 
 * formula images into LaTeX representations.
 */

#include "common/types.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

namespace rapid_doc {

/**
 * @brief Formula recognizer configuration
 */
struct FormulaRecognizerConfig {
    std::string onnxModelPath;      // .onnx model path
    std::string dictPath;           // Path to formula token dictionary
    int inputSize = 384;            // Expected input max size (PP-FormulaNet-M uses 384)
};

/**
 * @brief Result of formula recognition
 */
struct FormulaResult {
    bool success = false;
    std::string latex;
    float confidence = 0.0f;
};

/**
 * @brief Formula Recognizer using ONNX Runtime
 */
class FormulaRecognizer {
public:
    explicit FormulaRecognizer(const FormulaRecognizerConfig& config);
    ~FormulaRecognizer();

    /**
     * @brief Initialize model and load dictionary
     * @return true if successful
     */
    bool initialize();

    /**
     * @brief Recognize a formula from a cropped image
     * @param image Input cropped BGR image of the formula
     * @return FormulaResult containing the recognized LaTeX string
     */
    FormulaResult recognize(const cv::Mat& image);

    /**
     * @brief Check if recognizer is initialized
     */
    bool isInitialized() const { return initialized_; }

private:
    /**
     * @brief Preprocess formula image
     * @param image Input BGR image
     * @return Preprocessed float tensor (NCHW format, 1x1xHxW for grayscale model, or 1x3xHxW)
     */
    std::vector<float> preprocess(const cv::Mat& image, int& out_h, int& out_w);

    /**
     * @brief Postprocess output token sequence to LaTeX string
     * @param tokens Output token IDs from the model
     * @return Decoded LaTeX string
     */
    std::string postprocess(const std::vector<int64_t>& tokens);

    struct Impl;
    std::unique_ptr<Impl> impl_;
    FormulaRecognizerConfig config_;
    bool initialized_ = false;
};

} // namespace rapid_doc
