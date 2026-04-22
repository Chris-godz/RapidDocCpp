#pragma once

/**
 * @file table_classifier.h
 * @brief Table wired/wireless classifier (PaddleCls ONNX).
 *
 * Mirrors Python reference in
 *   RapidDoc/rapid_doc/model/table/rapid_table_self/table_cls/main.py
 * which uses the PaddleClas paddle_cls.onnx model:
 *   - input:  float32 NCHW [1, 3, 224, 224]
 *   - preprocess: resize_short=256 (LANCZOS4), center-crop 224, /255,
 *                 mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225],
 *                 HWC -> CHW, batch=1
 *   - output: float logits [1, 2]; argmax -> 0=wired, 1=wireless
 *
 * Used by TableRecognizer as the primary router before the legacy
 * estimateTableType() geometric heuristic (which now acts as a fallback
 * when the classifier is disabled or fails to load).
 */

#include "common/types.h"

#include <opencv2/opencv.hpp>

#include <memory>
#include <string>

namespace rapid_doc {

struct TableClassifierConfig {
    std::string onnxModelPath;   // paddle_cls.onnx
    // -1 mirrors Python OrtInferSession default (do not set intra_op_num_threads).
    int intraOpThreads = -1;
    bool disableCpuMemArena = true;
};

struct TableClassifyResult {
    TableType type = TableType::UNKNOWN; // WIRED / WIRELESS / UNKNOWN
    std::string label;                   // "wired" | "wireless" | "disabled" | "error"
    int predIndex = -1;                  // argmax index (0=wired, 1=wireless); -1 if unknown
    float score = -1.0f;                 // softmax max; -1 if unknown
    float rawLogitWired = 0.0f;
    float rawLogitWireless = 0.0f;
    double preprocessMs = 0.0;
    double inferMs = 0.0;
    bool ok = false;                     // true iff argmax produced a valid label
};

class TableClassifier {
public:
    explicit TableClassifier(const TableClassifierConfig& config);
    ~TableClassifier();

    // Load the ONNX session. Returns false on model-missing / init failure.
    bool initialize();
    bool isInitialized() const { return initialized_; }

    // Classify a table crop. Input is BGR (OpenCV default; matches the Python
    // call site rapid_table.py L88 which hands `bgr_image` to TableCls).
    TableClassifyResult classify(const cv::Mat& tableImageBgr);

private:
    // Exact Python parity preprocess. Output is a contiguous float32 buffer
    // of length 1*3*224*224, stored in NCHW order (R/G/B first channel plane
    // follows Python: `np.asarray(img, float32) / 255.0 - mean / std` applied
    // in original channel order — we feed the classifier the same BGR order
    // as Python does).
    std::vector<float> preprocess(const cv::Mat& bgr, double& outPreprocessMs) const;

    struct Impl;
    TableClassifierConfig config_;
    std::unique_ptr<Impl> impl_;
    bool initialized_ = false;
};

} // namespace rapid_doc
