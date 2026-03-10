/**
 * @file table_recognizer.cpp
 * @brief Table recognition implementation (DEEPX NPU UNET — wired tables only)
 *
 * Aligned with Python:
 *   RapidDoc/rapid_doc/model/table/rapid_table_self/wired_table_rec/table_structure_unet.py
 *   RapidDoc/rapid_doc/model/table/rapid_table_self/wired_table_rec/utils/dx_infer_session.py
 *   RapidDoc/rapid_doc/model/table/rapid_table_self/wired_table_rec/utils/utils.py  (resize_with_padding)
 */

#include "table/table_recognizer.h"
#include "common/logger.h"

#include <dxrt/inference_engine.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <numeric>
#include <map>

namespace rapid_doc {

// ---------------------------------------------------------------------------
// Pimpl
// ---------------------------------------------------------------------------
struct TableRecognizer::Impl {
    std::unique_ptr<dxrt::InferenceEngine> dxEngine;
};

// ---------------------------------------------------------------------------
// Constructor / Destructor
// ---------------------------------------------------------------------------
TableRecognizer::TableRecognizer(const TableRecognizerConfig& config)
    : config_(config)
    , impl_(std::make_unique<Impl>())
{
}

TableRecognizer::~TableRecognizer() = default;

// ---------------------------------------------------------------------------
// Initialize
// ---------------------------------------------------------------------------
bool TableRecognizer::initialize() {
    LOG_INFO("Initializing Table recognizer (wired tables only)...");
    LOG_INFO("  UNET DXNN model: {}", config_.unetDxnnModelPath);

    try {
        impl_->dxEngine = std::make_unique<dxrt::InferenceEngine>(config_.unetDxnnModelPath);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to load UNET DX Engine model: {}", e.what());
        return false;
    }

    initialized_ = true;
    LOG_INFO("Table recognizer initialized successfully");
    return true;
}

// ---------------------------------------------------------------------------
// Preprocess — matches Python TSRUnet.preprocess() for dxengine path
//   resize_with_padding(img, target_size=768) → BGR→RGB → NHWC uint8
// ---------------------------------------------------------------------------
cv::Mat TableRecognizer::preprocess(const cv::Mat& image) {
    int targetSize = config_.inputSize;  // 768
    int h = image.rows;
    int w = image.cols;

    float scale = static_cast<float>(targetSize) / std::max(h, w);
    int newH = static_cast<int>(h * scale);
    int newW = static_cast<int>(w * scale);

    int interpolation = (scale < 1.0f) ? cv::INTER_AREA : cv::INTER_CUBIC;
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(newW, newH), 0, 0, interpolation);

    // Center padding with white (255,255,255)
    int padH = targetSize - newH;
    int padW = targetSize - newW;
    int padTop = padH / 2;
    int padBottom = padH - padTop;
    int padLeft = padW / 2;
    int padRight = padW - padLeft;

    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, padTop, padBottom, padLeft, padRight,
                       cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));

    // BGR → RGB (Python does cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img))
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);

    return rgb;
}

// ---------------------------------------------------------------------------
// extractCells — matches Python _postprocess_dxengine()
//   1. Separate h/v predictions from segmentation mask
//   2. Remove padding, resize back to original size
//   3. Morphological close to enhance lines
//   4. Connected component analysis → cell bounding boxes
// ---------------------------------------------------------------------------
std::vector<TableCell> TableRecognizer::extractCells(
    const cv::Mat& mask, const cv::Size& originalSize)
{
    int origH = originalSize.height;
    int origW = originalSize.width;
    int targetSize = config_.inputSize;

    float scale = static_cast<float>(targetSize) / std::max(origH, origW);
    int padTop = (targetSize - static_cast<int>(origH * scale)) / 2;
    int padLeft = (targetSize - static_cast<int>(origW * scale)) / 2;
    int scaledH = static_cast<int>(origH * scale);
    int scaledW = static_cast<int>(origW * scale);

    // Separate: 0=background, 1=horizontal line, 2=vertical line
    cv::Mat hpred = (mask == 1);
    cv::Mat vpred = (mask == 2);

    // Remove padding region
    int hEnd = std::min(padTop + scaledH, mask.rows);
    int wEnd = std::min(padLeft + scaledW, mask.cols);
    hpred = hpred(cv::Range(padTop, hEnd), cv::Range(padLeft, wEnd));
    vpred = vpred(cv::Range(padTop, hEnd), cv::Range(padLeft, wEnd));

    // Morphological kernel sizes (matches Python: sqrt(w)*1.2, sqrt(h)*1.2)
    int hKernW = std::max(1, static_cast<int>(std::sqrt(static_cast<float>(hpred.cols)) * 1.2f));
    int vKernH = std::max(1, static_cast<int>(std::sqrt(static_cast<float>(vpred.rows)) * 1.2f));

    // Resize to original image size
    cv::resize(hpred, hpred, cv::Size(origW, origH));
    cv::resize(vpred, vpred, cv::Size(origW, origH));

    // Morphological CLOSE to enhance lines
    cv::Mat hKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(hKernW, 1));
    cv::Mat vKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, vKernH));
    cv::morphologyEx(vpred, vpred, cv::MORPH_CLOSE, vKernel, cv::Point(-1, -1), 1);
    cv::morphologyEx(hpred, hpred, cv::MORPH_CLOSE, hKernel, cv::Point(-1, -1), 1);

    // Draw lines on blank canvas
    cv::Mat lineImg = cv::Mat::zeros(origH, origW, CV_8UC1);

    // Extract horizontal line contours and record their y-positions (row separators)
    std::vector<std::vector<cv::Point>> hContours;
    cv::findContours(hpred, hContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<float> rowSepY;   // y-positions of horizontal lines (row boundaries)
    for (const auto& contour : hContours) {
        cv::Rect bbox = cv::boundingRect(contour);
        if (bbox.width >= 30) {
            int cy = bbox.y + bbox.height / 2;
            cv::line(lineImg,
                     cv::Point(bbox.x, cy),
                     cv::Point(bbox.x + bbox.width, cy),
                     255, 2);
            rowSepY.push_back(static_cast<float>(cy));
        }
    }

    // Extract vertical line contours and record their x-positions (column separators)
    std::vector<std::vector<cv::Point>> vContours;
    cv::findContours(vpred, vContours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    std::vector<float> colSepX;   // x-positions of vertical lines (column boundaries)
    for (const auto& contour : vContours) {
        cv::Rect bbox = cv::boundingRect(contour);
        if (bbox.height >= 20) {
            int cx = bbox.x + bbox.width / 2;
            cv::line(lineImg,
                     cv::Point(cx, bbox.y),
                     cv::Point(cx, bbox.y + bbox.height),
                     255, 2);
            colSepX.push_back(static_cast<float>(cx));
        }
    }

    // Connected component analysis on inverted line image → cell regions
    // (Python: measure.label(line_img < 255, connectivity=2))
    cv::Mat inverted;
    cv::threshold(lineImg, inverted, 200, 255, cv::THRESH_BINARY_INV);

    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(inverted, labels, stats, centroids, 8);

    // Collect cells, filtering only the large background-spanning region.
    // Python uses: filter if bbox_area > H * W * 3/4  (only drop over-large regions)
    // DO NOT use a minimum size — small cells are legitimate in dense tables.
    std::vector<TableCell> cells;
    float minCellArea = 4.0f;          // just filter single/few pixels
    float maxCellArea = origH * origW * 0.75f;  // match Python: skip region > 75% total

    // Build row/col assignment by sorting cell centers
    struct CellInfo {
        int labelId;
        float cx, cy;
        cv::Rect bbox;
    };
    std::vector<CellInfo> cellInfos;

    for (int i = 1; i < numLabels; ++i) {
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        if (area < minCellArea) continue;
        // Filter out large background-spanning regions (matches Python: bbox_area > H*W*3/4)
        if (static_cast<float>(w * h) > maxCellArea) continue;

        CellInfo ci;
        ci.labelId = i;
        ci.cx = static_cast<float>(centroids.at<double>(i, 0));
        ci.cy = static_cast<float>(centroids.at<double>(i, 1));
        ci.bbox = cv::Rect(x, y, w, h);
        cellInfos.push_back(ci);
    }

    if (cellInfos.empty()) return cells;

    // Sort by y first, then x
    std::sort(cellInfos.begin(), cellInfos.end(), [](const CellInfo& a, const CellInfo& b) {
        if (std::abs(a.cy - b.cy) > 10.0f)
            return a.cy < b.cy;
        return a.cx < b.cx;
    });

    // Compute adaptive row threshold based on median cell height
    std::vector<float> heights;
    for (const auto& ci : cellInfos)
        heights.push_back(static_cast<float>(ci.bbox.height));
    std::sort(heights.begin(), heights.end());
    float medianH = heights[heights.size() / 2];
    float rowThreshold = std::max(10.0f, medianH * 0.4f);

    int currentRow = 0;
    float lastY = cellInfos[0].cy;
    int colInRow = 0;

    for (auto& ci : cellInfos) {
        if (std::abs(ci.cy - lastY) > rowThreshold) {
            currentRow++;
            colInRow = 0;
            lastY = ci.cy;
        }

        TableCell cell;
        cell.row = currentRow;
        cell.col = colInRow++;
        cell.rowSpan = 1;
        cell.colSpan = 1;
        cell.x0 = static_cast<float>(ci.bbox.x);
        cell.y0 = static_cast<float>(ci.bbox.y);
        cell.x1 = static_cast<float>(ci.bbox.x + ci.bbox.width);
        cell.y1 = static_cast<float>(ci.bbox.y + ci.bbox.height);
        cells.push_back(cell);
    }

    // Compute rowspan/colspan from LINE POSITIONS (column/row separators),
    // NOT from cell extents — cell bbox edges are noisy and create inflated spans.
    // Column separators = vertical line x-positions, row separators = horizontal line y-positions.

    // Cluster separator positions with adaptive tolerance
    auto clusterPositions = [](std::vector<float>& pos, float tol) {
        if (pos.empty()) return;
        std::sort(pos.begin(), pos.end());
        std::vector<float> clustered;
        float groupSum = pos[0];
        int groupCount = 1;
        for (size_t i = 1; i < pos.size(); ++i) {
            if (pos[i] - pos[i-1] <= tol) {
                groupSum += pos[i];
                groupCount++;
            } else {
                clustered.push_back(groupSum / groupCount);
                groupSum = pos[i];
                groupCount = 1;
            }
        }
        clustered.push_back(groupSum / groupCount);
        pos = clustered;
    };

    // Tolerance: 5% of average cell size, min 8px
    float avgCellW = (cells.empty()) ? 20.0f :
        static_cast<float>(origW) / std::max(1.0f, static_cast<float>(colSepX.size()));
    float avgCellH = (cells.empty()) ? 20.0f :
        static_cast<float>(origH) / std::max(1.0f, static_cast<float>(rowSepY.size()));
    clusterPositions(colSepX, std::max(8.0f, avgCellW * 0.05f));
    clusterPositions(rowSepY, std::max(8.0f, avgCellH * 0.05f));

    // Add table edges as boundaries
    colSepX.insert(colSepX.begin(), 0.0f);
    colSepX.push_back(static_cast<float>(origW));
    rowSepY.insert(rowSepY.begin(), 0.0f);
    rowSepY.push_back(static_cast<float>(origH));
    std::sort(colSepX.begin(), colSepX.end());
    std::sort(rowSepY.begin(), rowSepY.end());

    // For each cell, count how many column/row bands it spans by checking how many
    // separators fall within [cellStart+margin, cellEnd-margin]
    if (colSepX.size() >= 2 && rowSepY.size() >= 2) {
        for (auto& c : cells) {
            // Column span: count vertical separators strictly inside cell x-range
            float margin = 5.0f;
            int colSpan = 1;
            for (float xp : colSepX) {
                if (xp > c.x0 + margin && xp < c.x1 - margin)
                    colSpan++;
            }
            // Row span: count horizontal separators strictly inside cell y-range
            int rowSpan = 1;
            for (float yp : rowSepY) {
                if (yp > c.y0 + margin && yp < c.y1 - margin)
                    rowSpan++;
            }
            c.colSpan = colSpan;
            c.rowSpan = rowSpan;
        }
    }

    return cells;
}

// ---------------------------------------------------------------------------
// Generate HTML from recognized cells
// ---------------------------------------------------------------------------
std::string TableRecognizer::generateHtml(const std::vector<TableCell>& cells) {
    if (cells.empty()) return "";

    int maxRow = 0;
    for (const auto& c : cells) maxRow = std::max(maxRow, c.row);

    // Group cells by row
    std::map<int, std::vector<const TableCell*>> rowMap;
    for (const auto& c : cells) {
        rowMap[c.row].push_back(&c);
    }

    std::string html = "<table border=\"1\">\n";
    for (int r = 0; r <= maxRow; ++r) {
        html += "  <tr>\n";
        auto it = rowMap.find(r);
        if (it != rowMap.end()) {
            auto& rowCells = it->second;
            for (const auto* cell : rowCells) {
                html += "    <td";
                if (cell->rowSpan > 1)
                    html += " rowspan=\"" + std::to_string(cell->rowSpan) + "\"";
                if (cell->colSpan > 1)
                    html += " colspan=\"" + std::to_string(cell->colSpan) + "\"";
                html += ">";
                html += cell->content.empty() ? "&nbsp;" : cell->content;
                html += "</td>\n";
            }
        }
        html += "  </tr>\n";
    }
    html += "</table>";

    return html;
}

// ---------------------------------------------------------------------------
// Recognize — full pipeline
// ---------------------------------------------------------------------------
TableResult TableRecognizer::recognize(const cv::Mat& tableImage) {
    TableResult result;
    result.type = TableType::WIRED;

    if (!initialized_) {
        LOG_ERROR("Table recognizer not initialized");
        result.supported = false;
        return result;
    }

    auto tStart = std::chrono::steady_clock::now();

    // 1. Preprocess: resize_with_padding → BGR→RGB → NHWC uint8
    cv::Mat preprocessed = preprocess(tableImage);

    // 2. DX Engine inference
    auto dxOutputs = impl_->dxEngine->Run(
        static_cast<void*>(preprocessed.data));

    // 3. Parse segmentation mask
    //    Python: result = session(input["img"][None,...])[0][0]
    //            result = result[0].astype(np.uint8)
    //    Output shape: (1, 1, H, W), dtype int64, values 0/1/2
    auto& outTensor = dxOutputs[0];
    int maskH = config_.inputSize;
    int maskW = config_.inputSize;
    auto& outShape = outTensor->shape();
    if (outShape.size() >= 2) {
        maskH = static_cast<int>(outShape[outShape.size() - 2]);
        maskW = static_cast<int>(outShape[outShape.size() - 1]);
    }

    // DXRT returns int64 per element — cast to uint8 for OpenCV processing.
    const int64_t* rawPtr = reinterpret_cast<const int64_t*>(outTensor->data());
    cv::Mat mask(maskH, maskW, CV_8UC1);
    for (int i = 0; i < maskH * maskW; ++i) {
        mask.data[i] = static_cast<uint8_t>(rawPtr[i]);
    }

    // 4. Extract cells from mask
    cv::Size origSize(tableImage.cols, tableImage.rows);
    result.cells = extractCells(mask, origSize);

    // 5. Generate HTML
    result.html = generateHtml(result.cells);
    result.supported = true;

    auto tEnd = std::chrono::steady_clock::now();
    result.inferenceTimeMs =
        std::chrono::duration<double, std::milli>(tEnd - tStart).count();

    LOG_INFO("Table recognition: {} cells in {:.1f}ms",
             result.cells.size(), result.inferenceTimeMs);

    return result;
}

// ---------------------------------------------------------------------------
// estimateTableType — heuristic (unchanged, already implemented)
// ---------------------------------------------------------------------------
TableType TableRecognizer::estimateTableType(const cv::Mat& tableImage) {
    if (tableImage.empty()) return TableType::UNKNOWN;

    cv::Mat gray, edges;
    cv::cvtColor(tableImage, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 50, 150);

    cv::Mat horizontal;
    cv::Mat horizontalKernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(tableImage.cols / 4, 1));
    cv::morphologyEx(edges, horizontal, cv::MORPH_OPEN, horizontalKernel);

    cv::Mat vertical;
    cv::Mat verticalKernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(1, tableImage.rows / 4));
    cv::morphologyEx(edges, vertical, cv::MORPH_OPEN, verticalKernel);

    int hLinePixels = cv::countNonZero(horizontal);
    int vLinePixels = cv::countNonZero(vertical);
    int totalPixels = tableImage.cols * tableImage.rows;

    float lineRatio = static_cast<float>(hLinePixels + vLinePixels) / totalPixels;

    if (lineRatio > 0.01f) {
        return TableType::WIRED;
    } else {
        return TableType::WIRELESS;
    }
}

} // namespace rapid_doc
