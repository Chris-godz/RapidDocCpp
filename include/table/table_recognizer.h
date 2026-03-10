#pragma once

/**
 * @file table_recognizer.h
 * @brief Table recognition using DEEPX NPU (UNET model — wired tables only)
 * 
 * Architecture (matching Python dx_infer_session.py):
 *   - Single DX Engine: dxrt::InferenceEngine loads .dxnn UNET model
 *   - No ONNX sub-model needed (unlike Layout)
 *   - Post-processing: extract cell boundaries from segmentation mask (C++)
 * 
 * IMPORTANT: Only wired tables (tables with visible borders) are supported.
 * Wireless table recognition (SLANet/SLANeXt) is NOT supported on DEEPX NPU.
 * Pipeline should skip wireless tables or output raw cropped images as fallback.
 */

#include "common/types.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>

namespace rapid_doc {

/**
 * @brief Table recognizer configuration
 */
struct TableRecognizerConfig {
    std::string unetDxnnModelPath;   // .dxnn UNET model
    int inputSize = 768;             // Model input size (matches Python DXNN target_size=768)
    float threshold = 0.5f;          // Segmentation threshold
    bool useAsync = false;           // Enable async inference
};

/**
 * @brief Table recognizer using DEEPX NPU (wired tables only)
 */
class TableRecognizer {
public:
    explicit TableRecognizer(const TableRecognizerConfig& config);
    ~TableRecognizer();

    /**
     * @brief Initialize UNET model
     * @return true if successful
     */
    bool initialize();

    /**
     * @brief Recognize table structure from cropped table image
     * @param tableImage Cropped table region (BGR)
     * @return Table recognition result
     * 
     * NOTE: Caller should ensure this is a wired table.
     *       For wireless tables, this will return result with supported=false.
     */
    TableResult recognize(const cv::Mat& tableImage);

    /**
     * @brief Check if a table is likely wired (has visible borders)
     * @param tableImage Cropped table image
     * @return Estimated table type
     * 
     * Simple heuristic-based check (edge detection), since Table Cls model
     * is not supported on DEEPX NPU. This is a rough estimate only.
     */
    static TableType estimateTableType(const cv::Mat& tableImage);

    bool isInitialized() const { return initialized_; }

    /**
     * @brief Generate HTML from recognized cells (public for re-generation after OCR fill)
     */
    std::string generateHtml(const std::vector<TableCell>& cells);

private:
    cv::Mat preprocess(const cv::Mat& image);
    std::vector<TableCell> extractCells(const cv::Mat& mask, const cv::Size& originalSize);

    struct Impl;
    std::unique_ptr<Impl> impl_;
    TableRecognizerConfig config_;
    bool initialized_ = false;
};

} // namespace rapid_doc
