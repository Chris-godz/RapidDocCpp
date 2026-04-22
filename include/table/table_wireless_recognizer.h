#pragma once

/**
 * @file table_wireless_recognizer.h
 * @brief Wireless table recognition (SLANet+ ONNX) — C++ port of the Python
 *        reference in
 *          RapidDoc/rapid_doc/model/table/rapid_table_self/
 *            - main.py (RapidTable orchestration)
 *            - table_structure/pp_structure/{pre,post}_process.py
 *            - table_structure/pp_structure/main.py (PPTableStructurer)
 *            - table_matcher/main.py (TableMatch)
 *
 * Model (ONNX):
 *   - input  x:              float32 NCHW [1, 3, 488, 488]
 *   - output bbox_preds:     float32      [1, T, 8]   (per-step cell polygon)
 *   - output struct_probs:   float32      [1, T, 50]  (sos + 48 tokens + eos)
 *   - metadata "character":  48 newline-separated HTML structure tokens
 *     (<thead>, </thead>, <tbody>, </tbody>, <tr>, </tr>, <td, >, </td>,
 *      colspan=..., rowspan=..., <td></td>)
 *
 * Output:
 *   recognize() returns a TableResult with
 *     supported = true
 *     type      = WIRELESS
 *     html      = fully assembled HTML with OCR-matched text content
 *     cells     = empty (SLANet+ produces an HTML stream, not a grid)
 */

#include "common/types.h"

#include <opencv2/opencv.hpp>

#include <memory>
#include <string>
#include <vector>

namespace rapid_doc {

struct TableWirelessRecognizerConfig {
    std::string onnxModelPath;   // slanet-plus.onnx
    int inputSize = 488;         // SLANet+ canvas; SLANext would be 512
    int intraOpThreads = -1;     // -1 = leave unset (mirror Python)
    bool disableCpuMemArena = true;
};

// Simple OCR box input for the wireless matcher. The recognizer does not
// depend on DXNN-OCR-cpp headers directly — the caller is responsible for
// converting ocr::PipelineOCRResult → WirelessOcrBox. This keeps doc_table
// build-self-contained.
struct WirelessOcrBox {
    cv::Rect aabb;      // crop-local axis-aligned box (NOT page coordinates)
    std::string text;   // recognized text (may contain HTML tags like <b>)
};

class TableWirelessRecognizer {
public:
    explicit TableWirelessRecognizer(const TableWirelessRecognizerConfig& config);
    ~TableWirelessRecognizer();

    bool initialize();
    bool isInitialized() const { return initialized_; }

    // bgrImage:  cropped table region (BGR, OpenCV native)
    // ocrBoxes:  OCR results already computed for the same crop. Empty is
    //            allowed and produces structure-only HTML (no cell text).
    TableResult recognize(
        const cv::Mat& bgrImage,
        const std::vector<WirelessOcrBox>& ocrBoxes);

private:
    struct Impl;
    TableWirelessRecognizerConfig config_;
    std::unique_ptr<Impl> impl_;
    bool initialized_ = false;
};

} // namespace rapid_doc
