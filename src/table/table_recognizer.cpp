/**
 * @file table_recognizer.cpp
 * @brief Table recognition aligned with Python wired_table_rec pipeline
 *
 * Python sources:
 *   table_structure_unet.py  → _postprocess_dxengine()
 *   utils_table_line_rec.py  → get_table_line, adjust_lines, final_adjust_lines, draw_lines
 *   utils_table_line_rec.py  → min_area_rect_box
 *   table_recover.py         → TableRecover (get_rows, get_benchmark_cols, get_merge_cells)
 *   utils_table_recover.py   → plot_html_table, sorted_ocr_boxes
 */

#include "table/table_recognizer.h"
#include "table/table_classifier.h"
#include "common/logger.h"

#include <dxrt/inference_engine.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <map>
#include <set>
#include <stdexcept>
#include <tuple>

namespace fs = std::filesystem;

namespace rapid_doc {

// ============================================================================
// Pimpl
// ============================================================================
struct TableRecognizer::Impl {
    std::unique_ptr<dxrt::InferenceEngine> dxEngine;
    std::unique_ptr<TableClassifier> classifier;
};

// ============================================================================
// Constructor / Destructor / Initialize
// ============================================================================
TableRecognizer::TableRecognizer(const TableRecognizerConfig& config)
    : config_(config), impl_(std::make_unique<Impl>())
{
}

TableRecognizer::~TableRecognizer() = default;

bool TableRecognizer::initialize() {
    LOG_INFO("Initializing Table recognizer (wired tables only)...");
    LOG_INFO("  UNET DXNN model: {}", config_.unetDxnnModelPath);
    LOG_INFO("  Device ID: {}", config_.deviceId);
    try {
        if (config_.deviceId >= 0) {
            dxrt::InferenceOption option;
            option.devices.push_back(config_.deviceId);
            impl_->dxEngine = std::make_unique<dxrt::InferenceEngine>(
                config_.unetDxnnModelPath,
                option);
        } else {
            impl_->dxEngine = std::make_unique<dxrt::InferenceEngine>(config_.unetDxnnModelPath);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to load UNET DX Engine model: {}", e.what());
        return false;
    }

    // Optional: load the PaddleCls wired/wireless classifier. Failure is
    // non-fatal — the geometric lineRatio heuristic acts as a safe fallback.
    if (config_.enableTableCls && !config_.tableClsOnnxModelPath.empty()) {
        TableClassifierConfig clsCfg;
        clsCfg.onnxModelPath = config_.tableClsOnnxModelPath;
        // Mirror Python OrtInferSession defaults (num_threads=-1 ⇒ leave alone).
        clsCfg.intraOpThreads = -1;
        clsCfg.disableCpuMemArena = true;
        auto classifier = std::make_unique<TableClassifier>(clsCfg);
        if (classifier->initialize()) {
            impl_->classifier = std::move(classifier);
            LOG_INFO("Table classifier enabled (paddle_cls.onnx)");
        } else {
            LOG_WARN(
                "Table classifier unavailable; falling back to lineRatio heuristic (path={})",
                config_.tableClsOnnxModelPath);
        }
    } else if (!config_.enableTableCls) {
        LOG_INFO("Table classifier disabled by config; using lineRatio heuristic only");
    }

    initialized_ = true;
    LOG_INFO("Table recognizer initialized successfully");
    return true;
}

// ============================================================================
// Preprocess — resize_with_padding → BGR→RGB → NHWC uint8
// ============================================================================
cv::Mat TableRecognizer::preprocess(const cv::Mat& image) {
    int targetSize = config_.inputSize;
    int h = image.rows, w = image.cols;
    float scale = static_cast<float>(targetSize) / std::max(h, w);
    int newH = static_cast<int>(h * scale);
    int newW = static_cast<int>(w * scale);
    int interpolation = (scale < 1.0f) ? cv::INTER_AREA : cv::INTER_CUBIC;
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(newW, newH), 0, 0, interpolation);
    int padH = targetSize - newH, padW = targetSize - newW;
    int padTop = padH / 2, padBottom = padH - padTop;
    int padLeft = padW / 2, padRight = padW - padLeft;
    cv::Mat padded;
    cv::copyMakeBorder(resized, padded, padTop, padBottom, padLeft, padRight,
                       cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
    cv::Mat rgb;
    cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

// ============================================================================
// Geometry helpers
// ============================================================================
namespace {

struct Pt2f { float x, y; };

float dist2(Pt2f a, Pt2f b) {
    float dx = a.x - b.x, dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

// Order 4 points: top-left, top-right, bottom-right, bottom-left
void orderPoints(Pt2f pts[4], Pt2f out[4]) {
    // sort by x
    std::sort(pts, pts + 4, [](const Pt2f& a, const Pt2f& b){ return a.x < b.x; });
    Pt2f leftMost[2] = {pts[0], pts[1]};
    Pt2f rightMost[2] = {pts[2], pts[3]};
    std::sort(leftMost, leftMost + 2, [](const Pt2f& a, const Pt2f& b){ return a.y < b.y; });
    Pt2f tl = leftMost[0], bl = leftMost[1];
    // bottom-right = farthest from tl
    float d0 = dist2(tl, rightMost[0]), d1 = dist2(tl, rightMost[1]);
    Pt2f br, tr;
    if (d0 > d1) { br = rightMost[0]; tr = rightMost[1]; }
    else { br = rightMost[1]; tr = rightMost[0]; }
    out[0] = tl; out[1] = tr; out[2] = br; out[3] = bl;
}

// Calculate center, w, h, angle of a box
struct BoxInfo { float angle, w, h, cx, cy; };

BoxInfo calcBoxInfo(const float box[8]) {
    float x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3];
    float x3 = box[4], y3 = box[5], x4 = box[6], y4 = box[7];
    float cx = (x1 + x2 + x3 + x4) / 4.0f;
    float cy = (y1 + y2 + y3 + y4) / 4.0f;
    float w = (std::sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)) +
               std::sqrt((x3-x4)*(x3-x4)+(y3-y4)*(y3-y4))) / 2.0f;
    float h = (std::sqrt((x2-x3)*(x2-x3)+(y2-y3)*(y2-y3)) +
               std::sqrt((x1-x4)*(x1-x4)+(y1-y4)*(y1-y4))) / 2.0f;
    float denom = h * h + w * w;
    float sinA = 0.0f;
    if (denom > 1e-6f)
        sinA = std::clamp((h * (x1 - cx) - w * (y1 - cy)) * 2.0f / denom, -1.0f, 1.0f);
    float angle = std::asin(sinA);
    return {angle, w, h, cx, cy};
}

} // anonymous namespace

// ============================================================================
// get_table_line (Python utils_table_line_rec.py)
// Uses connected component analysis on the binary h/v pred mask,
// extracts line segments as (x1,y1,x2,y2) from min-area rectangle.
// ============================================================================
std::vector<TableRecognizer::LineSeg> TableRecognizer::getTableLine(
    const cv::Mat& binImg, int axis, int lineW)
{
    // Connected component analysis (Python: measure.label + regionprops)
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(binImg, labels, stats, centroids, 8);

    std::vector<LineSeg> lines;
    for (int i = 1; i < numLabels; ++i) {
        int bx = stats.at<int>(i, cv::CC_STAT_LEFT);
        int by = stats.at<int>(i, cv::CC_STAT_TOP);
        int bw = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int bh = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        // Python filter: axis=1(vertical): bbox height > lineW; axis=0(horizontal): bbox width > lineW
        if (axis == 1 && bh <= lineW) continue;
        if (axis == 0 && bw <= lineW) continue;

        // Get the pixel coordinates of this component for minAreaRect
        std::vector<cv::Point> coords;
        for (int r = by; r < by + bh; ++r) {
            const int* labelRow = labels.ptr<int>(r);
            for (int c = bx; c < bx + bw; ++c) {
                if (labelRow[c] == i) coords.emplace_back(c, r);
            }
        }
        if (coords.empty()) continue;

        // Python: min_area_rect(region.coords[:, ::-1]) — note coords are (row,col), reversed to (x,y)
        cv::RotatedRect rr = cv::minAreaRect(coords);
        cv::Point2f boxPts[4];
        rr.points(boxPts);
        float rawBox[8];
        for (int k = 0; k < 4; ++k) { rawBox[2*k] = boxPts[k].x; rawBox[2*k+1] = boxPts[k].y; }

        // Order points: tl, tr, br, bl
        Pt2f pts4[4] = {{rawBox[0],rawBox[1]},{rawBox[2],rawBox[3]},{rawBox[4],rawBox[5]},{rawBox[6],rawBox[7]}};
        Pt2f ordered[4];
        orderPoints(pts4, ordered);
        float ob[8] = {ordered[0].x,ordered[0].y, ordered[1].x,ordered[1].y,
                       ordered[2].x,ordered[2].y, ordered[3].x,ordered[3].y};

        BoxInfo bi = calcBoxInfo(ob);
        LineSeg seg;
        if (bi.w < bi.h) {
            seg.x1 = (ob[0] + ob[2]) / 2; seg.y1 = (ob[1] + ob[3]) / 2;
            seg.x2 = (ob[4] + ob[6]) / 2; seg.y2 = (ob[5] + ob[7]) / 2;
        } else {
            seg.x1 = (ob[0] + ob[6]) / 2; seg.y1 = (ob[1] + ob[7]) / 2;
            seg.x2 = (ob[2] + ob[4]) / 2; seg.y2 = (ob[3] + ob[5]) / 2;
        }
        // Drop very short segments to reduce spurious grid boundaries (table misalignment)
        float segLen = dist2({seg.x1, seg.y1}, {seg.x2, seg.y2});
        int imgDim = (axis == 1) ? binImg.rows : binImg.cols;
        // Min length for vertical lines: 0.07 to target ~20 cols (Python); 0.06→25, 0.08/0.10→17
        float minRatio = (axis == 1) ? 0.07f : 0.06f;
        float minLen = std::max(static_cast<float>(lineW), minRatio * static_cast<float>(imgDim));
        if (segLen < minLen) continue;
        lines.push_back(seg);
    }
    return lines;
}

// ============================================================================
// adjust_lines (Python utils_table_line_rec.py)
// Finds nearby endpoints between different line segments and adds bridge lines
// ============================================================================
std::vector<TableRecognizer::LineSeg> TableRecognizer::adjustLines(
    const std::vector<LineSeg>& lines, float alph, float angleDeg)
{
    std::vector<LineSeg> newLines;
    int n = static_cast<int>(lines.size());
    for (int i = 0; i < n; ++i) {
        float x1 = lines[i].x1, y1 = lines[i].y1, x2 = lines[i].x2, y2 = lines[i].y2;
        float cx1 = (x1 + x2) / 2, cy1 = (y1 + y2) / 2;
        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            float x3 = lines[j].x1, y3 = lines[j].y1, x4 = lines[j].x2, y4 = lines[j].y2;
            float cx2 = (x3 + x4) / 2, cy2 = (y3 + y4) / 2;
            // Check projection overlap
            if ((x3 < cx1 && cx1 < x4) || (y3 < cy1 && cy1 < y4) ||
                (x1 < cx2 && cx2 < x2) || (y1 < cy2 && cy2 < y2))
                continue;

            auto tryBridge = [&](float ax, float ay, float bx, float by) {
                float r = dist2({ax, ay}, {bx, by});
                float k = std::abs((by - ay) / (bx - ax + 1e-10f));
                float a = std::atan(k) * 180.0f / static_cast<float>(M_PI);
                if (r < alph && a < angleDeg)
                    newLines.push_back({ax, ay, bx, by});
            };
            tryBridge(x1, y1, x3, y3);
            tryBridge(x1, y1, x4, y4);
            tryBridge(x2, y2, x3, y3);
            tryBridge(x2, y2, x4, y4);
        }
    }
    return newLines;
}

// ============================================================================
// lineToLine (Python utils_table_line_rec.py)
// Extends a line segment to reach the intersection with another line
// ============================================================================
TableRecognizer::LineSeg TableRecognizer::lineToLine(
    LineSeg pts1, const LineSeg& pts2, float alpha, float angleDeg)
{
    auto fitLine = [](float x1, float y1, float x2, float y2) -> std::tuple<float,float,float> {
        float A = y2 - y1, B = x1 - x2, C = x2 * y1 - x1 * y2;
        return {A, B, C};
    };
    auto pointLineCor = [](float px, float py, float A, float B, float C) {
        return A * px + B * py + C;
    };

    auto [A1, B1, C1] = fitLine(pts1.x1, pts1.y1, pts1.x2, pts1.y2);
    auto [A2, B2, C2] = fitLine(pts2.x1, pts2.y1, pts2.x2, pts2.y2);
    float flag1 = pointLineCor(pts1.x1, pts1.y1, A2, B2, C2);
    float flag2 = pointLineCor(pts1.x2, pts1.y2, A2, B2, C2);

    if ((flag1 > 0 && flag2 > 0) || (flag1 < 0 && flag2 < 0)) {
        float det = A1 * B2 - A2 * B1;
        if (det != 0) {
            float ix = (B1 * C2 - B2 * C1) / det;
            float iy = (A2 * C1 - A1 * C2) / det;
            float r0 = dist2({ix, iy}, {pts1.x1, pts1.y1});
            float r1 = dist2({ix, iy}, {pts1.x2, pts1.y2});
            if (std::min(r0, r1) < alpha) {
                if (r0 < r1) {
                    float k = std::abs((pts1.y2 - iy) / (pts1.x2 - ix + 1e-10f));
                    float a = std::atan(k) * 180.0f / static_cast<float>(M_PI);
                    if (a < angleDeg || std::abs(90.0f - a) < angleDeg)
                        pts1 = {ix, iy, pts1.x2, pts1.y2};
                } else {
                    float k = std::abs((pts1.y1 - iy) / (pts1.x1 - ix + 1e-10f));
                    float a = std::atan(k) * 180.0f / static_cast<float>(M_PI);
                    if (a < angleDeg || std::abs(90.0f - a) < angleDeg)
                        pts1 = {pts1.x1, pts1.y1, ix, iy};
                }
            }
        }
    }
    return pts1;
}

// ============================================================================
// final_adjust_lines — extends row/col line endpoints to meet each other
// ============================================================================
void TableRecognizer::finalAdjustLines(
    std::vector<LineSeg>& rowboxes, std::vector<LineSeg>& colboxes)
{
    for (auto& rb : rowboxes)
        for (auto& cb : colboxes) {
            rb = lineToLine(rb, cb, 20.0f, 30.0f);
            cb = lineToLine(cb, rb, 20.0f, 30.0f);
        }
}

// ============================================================================
// extractCellPolygons — min_area_rect_box from Python
// Connected components on inverted line image → 8-point polygons
// ============================================================================
std::vector<TableRecognizer::Polygon8> TableRecognizer::extractCellPolygons(
    const cv::Mat& lineImg, int W, int H)
{
    cv::Mat inverted;
    cv::threshold(lineImg, inverted, 200, 255, cv::THRESH_BINARY_INV);

    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(inverted, labels, stats, centroids, 8);

    std::vector<Polygon8> polygons;
    float maxArea = static_cast<float>(H) * W * 0.75f;

    for (int i = 1; i < numLabels; ++i) {
        int bx = stats.at<int>(i, cv::CC_STAT_LEFT);
        int by = stats.at<int>(i, cv::CC_STAT_TOP);
        int bw = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int bh = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        // Python: bbox_area > H * W * 3/4 → skip
        if (static_cast<float>(bw * bh) > maxArea) continue;

        // Collect component pixel coords
        std::vector<cv::Point> coords;
        coords.reserve(area);
        for (int r = by; r < by + bh; ++r) {
            const int* lr = labels.ptr<int>(r);
            for (int c = bx; c < bx + bw; ++c)
                if (lr[c] == i) coords.emplace_back(c, r);
        }
        if (coords.empty()) continue;

        cv::RotatedRect rr = cv::minAreaRect(coords);
        cv::Point2f boxPts[4];
        rr.points(boxPts);

        float rawBox[8];
        for (int k = 0; k < 4; ++k) { rawBox[2*k] = boxPts[k].x; rawBox[2*k+1] = boxPts[k].y; }

        Pt2f pts4[4] = {{rawBox[0],rawBox[1]},{rawBox[2],rawBox[3]},{rawBox[4],rawBox[5]},{rawBox[6],rawBox[7]}};
        Pt2f ordered[4];
        orderPoints(pts4, ordered);
        float ob[8] = {ordered[0].x,ordered[0].y, ordered[1].x,ordered[1].y,
                       ordered[2].x,ordered[2].y, ordered[3].x,ordered[3].y};

        BoxInfo bi = calcBoxInfo(ob);
        // Python filter: w * h < 0.5 * W * H, and filtersmall: w < 15 or h < 15
        if (bi.w * bi.h >= 0.5f * W * H) continue;
        if (bi.w < 15.0f || bi.h < 15.0f) continue;

        Polygon8 p;
        std::copy(ob, ob + 8, p.pts);
        polygons.push_back(p);
    }
    return polygons;
}

// ============================================================================
// recoverTableStructure — Python TableRecover
// ============================================================================
void TableRecognizer::recoverTableStructure(
    std::vector<Polygon8>& polygons,
    std::vector<LogicPoint>& logicPoints,
    float rowThresh, float colThresh)
{
    int N = static_cast<int>(polygons.size());
    if (N == 0) return;

    // Sort polygons by (y of top-left, then x of top-left)
    // Python: sorted_ocr_boxes sorts by (y1, x1) with bubble-sort fix
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](int a, int b) {
        float ya = polygons[a].pts[1], yb = polygons[b].pts[1];
        if (std::abs(ya - yb) > 20.0f) return ya < yb;
        return polygons[a].pts[0] < polygons[b].pts[0];
    });
    std::vector<Polygon8> sorted(N);
    for (int i = 0; i < N; ++i) sorted[i] = polygons[indices[i]];
    polygons = sorted;

    // ---- get_rows: group polygons into rows by y-axis differences ----
    std::map<int, std::vector<int>> rows;  // rowNum -> [polygon indices]
    {
        std::vector<float> yAxis(N);
        for (int i = 0; i < N; ++i) yAxis[i] = polygons[i].pts[1]; // top-left y

        if (N == 1) {
            rows[0] = {0};
        } else {
            std::vector<float> diffs(N - 1);
            for (int i = 0; i < N - 1; ++i) diffs[i] = yAxis[i + 1] - yAxis[i];

            std::vector<int> splitIdxs;
            for (int i = 0; i < (int)diffs.size(); ++i)
                if (std::abs(diffs[i]) > rowThresh) splitIdxs.push_back(i);

            if (splitIdxs.empty()) {
                rows[0].resize(N);
                std::iota(rows[0].begin(), rows[0].end(), 0);
            } else {
                if (splitIdxs.back() != (int)diffs.size())
                    splitIdxs.push_back(static_cast<int>(diffs.size()));
                int start = 0;
                for (int rn = 0; rn < (int)splitIdxs.size(); ++rn) {
                    if (rn != 0) start = splitIdxs[rn - 1] + 1;
                    for (int k = start; k <= splitIdxs[rn]; ++k)
                        rows[rn].push_back(k);
                }
            }
        }
    }

    // ---- get_benchmark_cols ----
    // Find the row with the most cells → its x-starts define column boundaries
    int longestRowIdx = 0;
    size_t longestLen = 0;
    for (auto& [rn, idxs] : rows)
        if (idxs.size() > longestLen) { longestLen = idxs.size(); longestRowIdx = rn; }

    auto& longestCol = rows[longestRowIdx];
    std::vector<float> longestXStart, longestXEnd;
    for (int idx : longestCol) {
        longestXStart.push_back(polygons[idx].pts[0]); // tl.x
        longestXEnd.push_back(polygons[idx].pts[4]);   // br.x
    }
    float minX = longestXStart.front(), maxX = longestXEnd.back();

    // Python: update_longest_col — insert col boundaries from other rows
    auto updateLongestCol = [&](std::vector<float>& colX, float curV, float& mn, float& mx, bool insertLast) {
        for (float v : colX)
            if (curV - colThresh <= v && v <= curV + colThresh) return;

        if (curV < mn) { colX.insert(colX.begin(), curV); mn = curV; return; }
        if (curV > mx) { if (insertLast) colX.push_back(curV); mx = curV; return; }
        for (int i = 0; i < (int)colX.size(); ++i)
            if (curV < colX[i]) { colX.insert(colX.begin() + i, curV); return; }
    };

    for (auto& [rn, idxs] : rows) {
        for (int idx : idxs) {
            float xs = polygons[idx].pts[0]; // tl.x
            float xe = polygons[idx].pts[4]; // br.x
            updateLongestCol(longestXStart, xs, minX, maxX, true);
            updateLongestCol(longestXStart, xe, minX, maxX, false);
        }
    }

    int colNums = static_cast<int>(longestXStart.size());
    std::vector<float> eachColWidths(colNums);
    for (int i = 0; i < colNums - 1; ++i)
        eachColWidths[i] = longestXStart[i + 1] - longestXStart[i];
    eachColWidths[colNums - 1] = maxX - longestXStart[colNums - 1];

    // ---- get_benchmark_rows ----
    std::vector<float> benchmarkY;
    for (auto& [rn, idxs] : rows)
        benchmarkY.push_back(polygons[idxs[0]].pts[1]); // leftmost cell tl.y
    int rowNums = static_cast<int>(benchmarkY.size());

    std::vector<float> eachRowHeights(rowNums);
    for (int i = 0; i < rowNums - 1; ++i)
        eachRowHeights[i] = benchmarkY[i + 1] - benchmarkY[i];
    // Last row: max height of cells in last row
    {
        auto& lastRow = rows.rbegin()->second;
        float maxH = 0;
        for (int idx : lastRow) {
            // height = L2(bl, tl)
            float h = dist2({polygons[idx].pts[6], polygons[idx].pts[7]},
                            {polygons[idx].pts[0], polygons[idx].pts[1]});
            maxH = std::max(maxH, h);
        }
        eachRowHeights[rowNums - 1] = maxH;
    }

    // ---- get_merge_cells: compute colspan/rowspan ----
    logicPoints.resize(N);
    float mergeThresh = 10.0f;

    for (auto& [curRow, colList] : rows) {
        std::map<int, int> oneColResult;
        for (int oneCol : colList) {
            auto& poly = polygons[oneCol];
            // box width = L2(tr, tl) = dist(top-right, top-left)
            // pts layout: [tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y]
            float boxWidth = dist2({poly.pts[2], poly.pts[3]}, {poly.pts[0], poly.pts[1]});

            // Find starting column index
            float cellX = poly.pts[0];
            int locColIdx = 0;
            float minDist = 1e9f;
            for (int ci = 0; ci < colNums; ++ci) {
                float d = std::abs(longestXStart[ci] - cellX);
                if (d < minDist) { minDist = d; locColIdx = ci; }
            }

            int sumPrev = 0;
            for (auto& [k, v] : oneColResult) sumPrev += v;
            int colStart = std::max(sumPrev, locColIdx);

            // Determine colspan
            int colSpan = 1;
            for (int ci = colStart; ci < colNums; ++ci) {
                float colCumSum = 0;
                for (int k = colStart; k <= ci; ++k) colCumSum += eachColWidths[k];
                if (ci == colStart && colCumSum > boxWidth) { colSpan = 1; break; }
                if (std::abs(colCumSum - boxWidth) <= mergeThresh) { colSpan = ci + 1 - colStart; break; }
                if (colCumSum > boxWidth) {
                    float prevCum = colCumSum - eachColWidths[ci];
                    int idx = (std::abs(colCumSum - boxWidth) < std::abs(prevCum - boxWidth)) ? ci : ci - 1;
                    colSpan = idx + 1 - colStart;
                    break;
                }
                if (ci == colNums - 1) colSpan = colNums - colStart;
            }
            oneColResult[oneCol] = colSpan;
            int colEnd = colStart + colSpan - 1;

            // Determine rowspan
            // box height = L2(bl, tl) = dist(bottom-left, top-left)
            float boxHeight = dist2({poly.pts[6], poly.pts[7]}, {poly.pts[0], poly.pts[1]});
            int rowStart = curRow;
            int rowSpan = 1;
            for (int rj = rowStart; rj < rowNums; ++rj) {
                float rowCumSum = 0;
                for (int k = rowStart; k <= rj; ++k) rowCumSum += eachRowHeights[k];
                if (rj == rowStart && rowCumSum > boxHeight) { rowSpan = 1; break; }
                if (std::abs(boxHeight - rowCumSum) <= mergeThresh) { rowSpan = rj + 1 - rowStart; break; }
                if (rowCumSum > boxHeight) {
                    float prevCum = rowCumSum - eachRowHeights[rj];
                    int idx = (std::abs(rowCumSum - boxHeight) < std::abs(prevCum - boxHeight)) ? rj : rj - 1;
                    rowSpan = idx + 1 - rowStart;
                    break;
                }
                if (rj == rowNums - 1) rowSpan = rowNums - rowStart;
            }
            int rowEnd = rowStart + rowSpan - 1;

            logicPoints[oneCol] = {rowStart, rowEnd, colStart, colEnd};
        }
    }
}

// ============================================================================
// postprocessDxEngine — full pipeline matching Python _postprocess_dxengine
// ============================================================================
std::vector<TableCell> TableRecognizer::postprocessDxEngine(
    const cv::Mat& /*img*/, const cv::Mat& pred,
    float scale, int padTop, int padLeft, int origH, int origW)
{
    cv::Mat predU8 = pred.clone();

    // Separate horizontal (1) and vertical (2) predictions
    cv::Mat hpred = (predU8 == 1);
    cv::Mat vpred = (predU8 == 2);

    // Remove padding
    int hEnd = std::min(padTop + static_cast<int>(origH * scale), predU8.rows);
    int wEnd = std::min(padLeft + static_cast<int>(origW * scale), predU8.cols);
    hpred = hpred(cv::Range(padTop, hEnd), cv::Range(padLeft, wEnd));
    vpred = vpred(cv::Range(padTop, hEnd), cv::Range(padLeft, wEnd));

    // Morphological kernel sizes (Python: sqrt(w)*1.2, sqrt(h)*1.2 on pred shape)
    int h = hpred.rows, w = hpred.cols;
    int hors_k = std::max(1, static_cast<int>(std::sqrt(static_cast<float>(w)) * 1.2f));
    int vert_k = std::max(1, static_cast<int>(std::sqrt(static_cast<float>(h)) * 1.2f));

    // Resize to original
    cv::resize(hpred, hpred, cv::Size(origW, origH));
    cv::resize(vpred, vpred, cv::Size(origW, origH));

    // Morphological operations
    cv::Mat hKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(hors_k, 1));
    cv::Mat vKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, vert_k));
    cv::morphologyEx(vpred, vpred, cv::MORPH_CLOSE, vKernel, cv::Point(-1, -1), 1);
    cv::morphologyEx(hpred, hpred, cv::MORPH_CLOSE, hKernel, cv::Point(-1, -1), 1);

    // Extract lines using get_table_line (Python defaults: row=50, col=30)
    auto colboxes = getTableLine(vpred, 1, 30);  // vertical lines
    auto rowboxes = getTableLine(hpred, 0, 50);  // horizontal lines

    // adjust_lines: add bridge segments between nearby endpoints
    auto rboxesRow = adjustLines(rowboxes, 100.0f, 50.0f);
    auto rboxesCol = adjustLines(colboxes, 15.0f, 50.0f);
    rowboxes.insert(rowboxes.end(), rboxesRow.begin(), rboxesRow.end());
    colboxes.insert(colboxes.end(), rboxesCol.begin(), rboxesCol.end());

    // final_adjust_lines: extend endpoints to meet intersections
    finalAdjustLines(rowboxes, colboxes);

    // Draw all lines on blank canvas
    cv::Mat lineImg = cv::Mat::zeros(origH, origW, CV_8UC1);
    auto allLines = rowboxes;
    allLines.insert(allLines.end(), colboxes.begin(), colboxes.end());
    for (auto& seg : allLines) {
        cv::line(lineImg,
                 cv::Point(static_cast<int>(seg.x1), static_cast<int>(seg.y1)),
                 cv::Point(static_cast<int>(seg.x2), static_cast<int>(seg.y2)),
                 255, 2, cv::LINE_AA);
    }

    // Extract cell polygons (Python: cal_region_boxes → min_area_rect_box)
    auto polyVec = extractCellPolygons(lineImg, origW, origH);
    if (polyVec.empty()) return {};

    // Recover table structure (row/col/span)
    std::vector<LogicPoint> logicPoints;
    recoverTableStructure(polyVec, logicPoints);

    // Convert to TableCell format
    std::vector<TableCell> cells(polyVec.size());
    for (int i = 0; i < (int)polyVec.size(); ++i) {
        auto& lp = logicPoints[i];
        auto& p = polyVec[i];
        cells[i].row = lp.rowStart;
        cells[i].col = lp.colStart;
        cells[i].rowSpan = lp.rowEnd - lp.rowStart + 1;
        cells[i].colSpan = lp.colEnd - lp.colStart + 1;
        std::copy(p.pts, p.pts + 8, cells[i].poly);
        // Axis-aligned bbox from polygon
        float xmin = std::min({p.pts[0], p.pts[2], p.pts[4], p.pts[6]});
        float xmax = std::max({p.pts[0], p.pts[2], p.pts[4], p.pts[6]});
        float ymin = std::min({p.pts[1], p.pts[3], p.pts[5], p.pts[7]});
        float ymax = std::max({p.pts[1], p.pts[3], p.pts[5], p.pts[7]});
        cells[i].x0 = xmin; cells[i].y0 = ymin;
        cells[i].x1 = xmax; cells[i].y1 = ymax;
    }
    return cells;
}

// ============================================================================
// generateHtml — aligned with Python plot_html_table
// ============================================================================
namespace {
std::string escapeHtml(const std::string& text) {
    std::string escaped;
    escaped.reserve(text.size());
    for (char ch : text) {
        switch (ch) {
            case '&': escaped += "&amp;"; break;
            case '<': escaped += "&lt;"; break;
            case '>': escaped += "&gt;"; break;
            case '"': escaped += "&quot;"; break;
            default: escaped.push_back(ch); break;
        }
    }
    return escaped;
}
} // namespace

std::string TableRecognizer::generateHtml(const std::vector<TableCell>& cells) {
    if (cells.empty()) return "";

    int maxRow = 0;
    int maxCol = 0;
    for (const auto& c : cells) {
        if (c.row < 0 || c.col < 0 || c.rowSpan <= 0 || c.colSpan <= 0) {
            throw std::runtime_error("invalid table cell span or index");
        }
        maxRow = std::max(maxRow, c.row + c.rowSpan);
        maxCol = std::max(maxCol, c.col + c.colSpan);
    }

    if (maxRow <= 0 || maxCol <= 0) {
        throw std::runtime_error("invalid table dimensions");
    }

    // Occupancy matrix: each slot stores the owning cell index.
    std::vector<std::vector<int>> occupancy(maxRow, std::vector<int>(maxCol, -1));
    for (int i = 0; i < static_cast<int>(cells.size()); ++i) {
        const auto& c = cells[i];
        for (int r = c.row; r < c.row + c.rowSpan; ++r) {
            for (int col = c.col; col < c.col + c.colSpan; ++col) {
                if (occupancy[r][col] != -1) {
                    throw std::runtime_error("overlapping table spans detected");
                }
                occupancy[r][col] = i;
            }
        }
    }

    // Hole / ragged validation: every row must fully cover [0, maxCol).
    for (int r = 0; r < maxRow; ++r) {
        for (int c = 0; c < maxCol; ++c) {
            if (occupancy[r][c] == -1) {
                throw std::runtime_error("table structure has hole/ragged row width");
            }
        }
    }

    std::string html = "<table border=\"1\">\n";
    for (int r = 0; r < maxRow; ++r) {
        html += "  <tr>";
        for (int c = 0; c < maxCol; ++c) {
            const int cellIdx = occupancy[r][c];
            const auto& cell = cells[cellIdx];
            if (cell.row == r && cell.col == c) {
                html += "<td";
                if (cell.rowSpan > 1) html += " rowspan=" + std::to_string(cell.rowSpan);
                if (cell.colSpan > 1) html += " colspan=" + std::to_string(cell.colSpan);
                html += ">";
                html += escapeHtml(cell.content);
                html += "</td>";
            }
        }
        html += "</tr>\n";
    }
    html += "</table>";
    return html;
}

// ============================================================================
// recognize — full pipeline
// ============================================================================
TableResult TableRecognizer::recognize(const cv::Mat& tableImage) {
    const NpuStageResult npuStage = recognizeNpuStage(tableImage);
    return finalizeRecognizePostprocess(tableImage, npuStage);
}

TableRecognizer::NpuStageResult TableRecognizer::recognizeNpuStage(const cv::Mat& tableImage) {
    NpuStageResult npuStage;
    auto tStart = std::chrono::steady_clock::now();

    if (tableImage.empty()) {
        npuStage.type = TableType::UNKNOWN;
        npuStage.supported = false;
        npuStage.npuStageTimeMs = 0.0;
        return npuStage;
    }

    // Routing: classifier (primary) → heuristic (fallback).
    // We always compute the heuristic result for trace observability even
    // when the classifier is active, but it only binds as the final verdict
    // if the classifier produced no usable label.
    auto estimateStart = std::chrono::steady_clock::now();
    const TableType heuristicType = estimateTableType(tableImage);
    auto estimateEnd = std::chrono::steady_clock::now();
    npuStage.estimateTableTypeMs =
        std::chrono::duration<double, std::milli>(estimateEnd - estimateStart).count();

    // F3 routing gate:
    //   - minSide < kClsMinSide (224) → bypass classifier (crop too small for
    //     reliable 224x224 classification; its LANCZOS4 upscale would magnify
    //     noise) and fall back to heuristic;
    //   - classifier score < kClsScoreThresh (0.6) → low-confidence flip risk,
    //     fall back to heuristic;
    //   - classifier error / disabled → fall back to heuristic.
    //   otherwise classifier verdict is authoritative.
    constexpr float kClsScoreThresh = 0.60f;
    constexpr int kClsMinSide = 224;

    TableType resolvedType = TableType::UNKNOWN;
    std::string clsLabel = "disabled";
    std::string clsGate = "disabled"; // pass | small_crop | low_score | error | disabled
    float clsScore = -1.0f;
    double clsInferMs = 0.0;

    const int minSide = std::min(tableImage.cols, tableImage.rows);
    if (impl_->classifier && impl_->classifier->isInitialized()) {
        if (minSide < kClsMinSide) {
            clsGate = "small_crop";
        } else {
            TableClassifyResult cls = impl_->classifier->classify(tableImage);
            clsLabel = cls.label;
            clsScore = cls.score;
            clsInferMs = cls.inferMs;
            if (!cls.ok) {
                clsGate = "error";
            } else if (cls.score >= kClsScoreThresh) {
                clsGate = "pass";
                resolvedType = cls.type;
            } else {
                clsGate = "low_score";
            }
        }
    }
    if (resolvedType == TableType::UNKNOWN) {
        resolvedType = heuristicType;
    }
    npuStage.type = resolvedType;

    // Env-gated route summary: one line per table crop with cls + heuristic +
    // final verdict + gate decision so downstream log diffs are trivial.
    const char* routeTrace = std::getenv("RAPIDDOC_TABLE_TRACE");
    if (routeTrace != nullptr && routeTrace[0] != '\0' && routeTrace[0] != '0') {
        const char* heuristicStr =
            (heuristicType == TableType::WIRED) ? "WIRED" :
            (heuristicType == TableType::WIRELESS) ? "WIRELESS" : "UNKNOWN";
        const char* finalStr =
            (resolvedType == TableType::WIRED) ? "WIRED" :
            (resolvedType == TableType::WIRELESS) ? "WIRELESS" : "UNKNOWN";
        LOG_INFO(
            "TABLE_ROUTE_TRACE route crop_size={}x{} cls_label={} cls_score={:.4f} "
            "cls_gate={} cls_infer_ms={:.2f} heuristic_type={} final_type={}",
            tableImage.cols, tableImage.rows,
            clsLabel.c_str(), clsScore,
            clsGate.c_str(), clsInferMs,
            heuristicStr, finalStr);
    }

    if (npuStage.type == TableType::WIRELESS) {
        npuStage.supported = false;
        auto tEnd = std::chrono::steady_clock::now();
        npuStage.npuStageTimeMs =
            std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        return npuStage;
    }

    if (!initialized_) {
        LOG_ERROR("Table recognizer not initialized");
        npuStage.supported = false;
        auto tEnd = std::chrono::steady_clock::now();
        npuStage.npuStageTimeMs =
            std::chrono::duration<double, std::milli>(tEnd - tStart).count();
        return npuStage;
    }

    npuStage.origH = tableImage.rows;
    npuStage.origW = tableImage.cols;
    int targetSize = config_.inputSize;
    npuStage.scale = static_cast<float>(targetSize) / std::max(npuStage.origH, npuStage.origW);
    npuStage.padTop = (targetSize - static_cast<int>(npuStage.origH * npuStage.scale)) / 2;
    npuStage.padLeft = (targetSize - static_cast<int>(npuStage.origW * npuStage.scale)) / 2;

    auto preprocessStart = std::chrono::steady_clock::now();
    cv::Mat preprocessed = preprocess(tableImage);
    auto preprocessEnd = std::chrono::steady_clock::now();
    npuStage.preprocessMs =
        std::chrono::duration<double, std::milli>(preprocessEnd - preprocessStart).count();

    auto dxRunStart = std::chrono::steady_clock::now();
    auto dxOutputs = impl_->dxEngine->Run(static_cast<void*>(preprocessed.data));
    auto dxRunEnd = std::chrono::steady_clock::now();
    npuStage.dxRunMs =
        std::chrono::duration<double, std::milli>(dxRunEnd - dxRunStart).count();

    auto& outTensor = dxOutputs[0];
    int maskH = targetSize;
    int maskW = targetSize;
    auto& outShape = outTensor->shape();
    if (outShape.size() >= 2) {
        maskH = static_cast<int>(outShape[outShape.size() - 2]);
        maskW = static_cast<int>(outShape[outShape.size() - 1]);
    }

    auto maskDecodeStart = std::chrono::steady_clock::now();
    const int64_t* rawPtr = reinterpret_cast<const int64_t*>(outTensor->data());
    npuStage.mask = cv::Mat(maskH, maskW, CV_8UC1);
    for (int i = 0; i < maskH * maskW; ++i) {
        npuStage.mask.data[i] = static_cast<uint8_t>(rawPtr[i]);
    }
    auto maskDecodeEnd = std::chrono::steady_clock::now();
    npuStage.maskDecodeMs =
        std::chrono::duration<double, std::milli>(maskDecodeEnd - maskDecodeStart).count();

    npuStage.supported = true;
    auto tEnd = std::chrono::steady_clock::now();
    npuStage.npuStageTimeMs =
        std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    return npuStage;
}

TableResult TableRecognizer::finalizeRecognizePostprocess(
    const cv::Mat& tableImage,
    const NpuStageResult& npuStage)
{
    TableResult result;
    result.type = npuStage.type;

    if (tableImage.empty()) {
        result.supported = false;
        return result;
    }

    if (!npuStage.supported) {
        result.supported = false;
        result.inferenceTimeMs = npuStage.npuStageTimeMs;
        if (result.type == TableType::WIRELESS) {
            LOG_INFO(
                "Table recognize profile (wireless fallback): total={:.2f}ms "
                "estimate_table_type={:.2f}ms",
                result.inferenceTimeMs,
                npuStage.estimateTableTypeMs);
        }
        return result;
    }

    auto postprocessStart = std::chrono::steady_clock::now();
    result.cells = postprocessDxEngine(
        tableImage,
        npuStage.mask,
        npuStage.scale,
        npuStage.padTop,
        npuStage.padLeft,
        npuStage.origH,
        npuStage.origW);
    auto postprocessEnd = std::chrono::steady_clock::now();
    const double postprocessMs =
        std::chrono::duration<double, std::milli>(postprocessEnd - postprocessStart).count();

    result.supported = true;
    result.inferenceTimeMs = npuStage.npuStageTimeMs + postprocessMs;

    LOG_INFO(
        "Table recognition: {} cells in {:.1f}ms",
        result.cells.size(),
        result.inferenceTimeMs);
    LOG_INFO(
        "Table recognize profile: total={:.2f}ms estimate_table_type={:.2f}ms "
        "preprocess={:.2f}ms dx_run={:.2f}ms mask_decode={:.2f}ms postprocess={:.2f}ms",
        result.inferenceTimeMs,
        npuStage.estimateTableTypeMs,
        npuStage.preprocessMs,
        npuStage.dxRunMs,
        npuStage.maskDecodeMs,
        postprocessMs);
    return result;
}

// ============================================================================
// estimateTableType (unchanged)
// ============================================================================
TableType TableRecognizer::estimateTableType(const cv::Mat& tableImage) {
    if (tableImage.empty()) return TableType::UNKNOWN;
    cv::Mat gray, edges;
    cv::cvtColor(tableImage, gray, cv::COLOR_BGR2GRAY);
    cv::Canny(gray, edges, 50, 150);
    cv::Mat horizontal;
    cv::Mat horizontalKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(tableImage.cols / 4, 1));
    cv::morphologyEx(edges, horizontal, cv::MORPH_OPEN, horizontalKernel);
    cv::Mat vertical;
    cv::Mat verticalKernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(1, tableImage.rows / 4));
    cv::morphologyEx(edges, vertical, cv::MORPH_OPEN, verticalKernel);
    int hLinePixels = cv::countNonZero(horizontal);
    int vLinePixels = cv::countNonZero(vertical);
    int totalPixels = tableImage.cols * tableImage.rows;
    float lineRatio = static_cast<float>(hLinePixels + vLinePixels) / totalPixels;
    const TableType resolved = (lineRatio > 0.01f) ? TableType::WIRED : TableType::WIRELESS;

    // Env-gated routing trace (no behavior change; zero I/O when unset).
    const char* traceRaw = std::getenv("RAPIDDOC_TABLE_TRACE");
    if (traceRaw != nullptr && traceRaw[0] != '\0' && traceRaw[0] != '0') {
        LOG_INFO(
            "TABLE_ROUTE_TRACE estimateTableType crop_size={}x{} hLinePixels={} "
            "vLinePixels={} totalPixels={} lineRatio={:.6f} type={}",
            tableImage.cols, tableImage.rows, hLinePixels, vLinePixels, totalPixels,
            lineRatio,
            (resolved == TableType::WIRED) ? "WIRED" : "WIRELESS");
    }
    return resolved;
}

} // namespace rapid_doc
