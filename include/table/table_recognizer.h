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
    struct NpuStageResult {
        TableType type = TableType::UNKNOWN;
        bool supported = false;
        cv::Mat mask;
        float scale = 1.0f;
        int padTop = 0;
        int padLeft = 0;
        int origH = 0;
        int origW = 0;
        double estimateTableTypeMs = 0.0;
        double preprocessMs = 0.0;
        double dxRunMs = 0.0;
        double maskDecodeMs = 0.0;
        double npuStageTimeMs = 0.0;
    };

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
     * @brief Run only the NPU-dependent phase (estimate/preprocess/dx_run/mask_decode).
     * @param tableImage Cropped table region (BGR)
     * @return Intermediate result for lock-safe postprocess split.
     */
    NpuStageResult recognizeNpuStage(const cv::Mat& tableImage);

    /**
     * @brief Finish table recognition by running CPU postprocess on NPU artifacts.
     * @param tableImage Original cropped table image (BGR)
     * @param npuStage NPU stage output from recognizeNpuStage()
     * @return Final table result with cells populated
     */
    TableResult finalizeRecognizePostprocess(
        const cv::Mat& tableImage,
        const NpuStageResult& npuStage);

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

    // Line extraction (aligned with Python get_table_line + adjust/extend)
    struct LineSeg { float x1, y1, x2, y2; };
    std::vector<LineSeg> getTableLine(const cv::Mat& binImg, int axis, int lineW);
    std::vector<LineSeg> adjustLines(const std::vector<LineSeg>& lines, float alph, float angle);
    void finalAdjustLines(std::vector<LineSeg>& rowboxes, std::vector<LineSeg>& colboxes);
    static LineSeg lineToLine(LineSeg pts1, const LineSeg& pts2, float alpha, float angle);

    // Cell polygon extraction (aligned with Python cal_region_boxes + min_area_rect_box)
    struct Polygon8 { float pts[8]; };
    std::vector<Polygon8> extractCellPolygons(const cv::Mat& lineImg, int W, int H);

    // TableRecover (aligned with Python table_recover.py)
    struct LogicPoint { int rowStart, rowEnd, colStart, colEnd; };
    void recoverTableStructure(
        std::vector<Polygon8>& polygons,
        std::vector<LogicPoint>& logicPoints,
        float rowThresh = 10.0f, float colThresh = 15.0f);

    // Full postprocess (dxengine path)
    std::vector<TableCell> postprocessDxEngine(
        const cv::Mat& img, const cv::Mat& pred,
        float scale, int padTop, int padLeft, int origH, int origW);

    struct Impl;
    TableRecognizerConfig config_;
    std::unique_ptr<Impl> impl_;
    bool initialized_ = false;
};

} // namespace rapid_doc
