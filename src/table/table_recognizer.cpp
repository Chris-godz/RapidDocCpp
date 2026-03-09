/**
 * @file table_recognizer.cpp
 * @brief Full wired table recognition: UNET segmentation → line extraction →
 *        cell detection → logical structure recovery → HTML generation.
 *
 * Reference: RapidDoc/rapid_doc/model/table/img2table_self/RapidOcrTable.py
 *            RapidDoc/rapid_doc/model/table/rapid_table_self/wired_table_rec/
 */

#include "table/table_recognizer.h"
#include "common/logger.h"

#include <opencv2/opencv.hpp>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <numeric>
#include <sstream>
#include <cstring>
#include <fstream>

#ifdef HAS_DXRT
#include <dxrt/inference_engine.h>
#endif

#ifdef HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace rapid_doc {

// ============================================================================
// Impl (pimpl) — holds DX Engine instance and ONNX Runtime instances
// ============================================================================
struct TableRecognizer::Impl {
#ifdef HAS_DXRT
    std::unique_ptr<dxrt::InferenceEngine> engine;
#endif
    bool engineLoaded = false;

#ifdef HAS_ONNXRUNTIME
    std::unique_ptr<Ort::Env> ortEnv;
    // Table Classification
    std::unique_ptr<Ort::Session> clsSession;
    std::vector<std::string> clsInputNames;
    std::vector<std::string> clsOutputNames;

    // SLANet Wireless Table
    std::unique_ptr<Ort::Session> slanetSession;
    std::vector<std::string> slanetInputNames;
    std::vector<std::string> slanetOutputNames;
#endif
    std::vector<std::string> slanetDict;
};

// ============================================================================
// Constructor / Destructor
// ============================================================================
TableRecognizer::TableRecognizer(const TableRecognizerConfig& config)
    : config_(config)
    , impl_(std::make_unique<Impl>())
{
}

TableRecognizer::~TableRecognizer() = default;

// ============================================================================
// Initialize — load DX Engine and ONNX sessions
// ============================================================================
bool TableRecognizer::initialize() {
    LOG_INFO("Initializing Table recognizer...");
    LOG_INFO("  UNET DXNN model: {}", config_.unetDxnnModelPath);
    LOG_INFO("  Input size: {}x{}", config_.inputSize, config_.inputSize);

#ifdef HAS_DXRT
    if (!config_.unetDxnnModelPath.empty()) {
        try {
            impl_->engine = std::make_unique<dxrt::InferenceEngine>(config_.unetDxnnModelPath);
            impl_->engineLoaded = true;
            LOG_INFO("UNET DX Engine loaded successfully");
        } catch (const std::exception& e) {
            LOG_WARN("DX Engine init failed (will use fallback): {}", e.what());
            impl_->engineLoaded = false;
        }
    } else {
        LOG_WARN("No UNET model path specified — using morphology-only fallback");
    }
#else
    LOG_WARN("DX Runtime not available at build time — using morphology-only fallback");
#endif

#ifdef HAS_ONNXRUNTIME
    try {
        impl_->ortEnv = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "rapid_doc_table");
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        opts.SetIntraOpNumThreads(1);

        // Load Classification model
        if (config_.enableTableClassify && !config_.tableClsOnnxModelPath.empty()) {
            impl_->clsSession = std::make_unique<Ort::Session>(
                *impl_->ortEnv, config_.tableClsOnnxModelPath.c_str(), opts);
            
            Ort::AllocatorWithDefaultOptions allocator;
            size_t numInputNodes = impl_->clsSession->GetInputCount();
            for (size_t i = 0; i < numInputNodes; i++) {
                auto namePtr = impl_->clsSession->GetInputNameAllocated(i, allocator);
                impl_->clsInputNames.push_back(namePtr.get());
            }
            size_t numOutputNodes = impl_->clsSession->GetOutputCount();
            for (size_t i = 0; i < numOutputNodes; i++) {
                auto namePtr = impl_->clsSession->GetOutputNameAllocated(i, allocator);
                impl_->clsOutputNames.push_back(namePtr.get());
            }
            LOG_INFO("Table classification ONNX model loaded");
        }

        // Load SLANet model
        if (config_.enableWirelessTable && !config_.tableSlanetOnnxModelPath.empty()) {
            impl_->slanetSession = std::make_unique<Ort::Session>(
                *impl_->ortEnv, config_.tableSlanetOnnxModelPath.c_str(), opts);
            
            Ort::AllocatorWithDefaultOptions allocator;
            size_t numInputNodes = impl_->slanetSession->GetInputCount();
            for (size_t i = 0; i < numInputNodes; i++) {
                auto namePtr = impl_->slanetSession->GetInputNameAllocated(i, allocator);
                impl_->slanetInputNames.push_back(namePtr.get());
            }
            size_t numOutputNodes = impl_->slanetSession->GetOutputCount();
            for (size_t i = 0; i < numOutputNodes; i++) {
                auto namePtr = impl_->slanetSession->GetOutputNameAllocated(i, allocator);
                impl_->slanetOutputNames.push_back(namePtr.get());
            }

            // Load dict
            if (!config_.tableSlanetDictPath.empty()) {
                std::ifstream dictFile(config_.tableSlanetDictPath);
                std::string line;
                while (std::getline(dictFile, line)) {
                    // remove \r if present
                    if (!line.empty() && line.back() == '\r') line.pop_back();
                    impl_->slanetDict.push_back(line);
                }
                LOG_INFO("Loaded SLANet dictionary with {} entries", impl_->slanetDict.size());
            }

            LOG_INFO("SLANet ONNX model loaded");
        }
    } catch (const std::exception& e) {
        LOG_WARN("ONNX Runtime init failed for TableRecognizer: {}", e.what());
    }
#else
    if (config_.enableTableClassify || config_.enableWirelessTable) {
        LOG_WARN("ONNX Runtime not available, table classification and wireless table will be skipped.");
    }
#endif

    initialized_ = true;
    return true;
}

// ============================================================================
// Main recognition entry point
// ============================================================================
TableResult TableRecognizer::recognize(const cv::Mat& tableImage) {
    TableResult result;
    result.type = TableType::WIRED;

    if (!initialized_) {
        LOG_ERROR("Table recognizer not initialized");
        result.supported = false;
        return result;
    }

    if (tableImage.empty()) {
        LOG_WARN("Empty table image received");
        result.supported = false;
        return result;
    }

    LOG_INFO("Table recognition: image {}x{}", tableImage.cols, tableImage.rows);

    // --- Step 1: Preprocess ---
    float scaleX, scaleY;
    int padLeft, padTop;
    cv::Mat preprocessed = preprocess(tableImage, scaleX, scaleY, padLeft, padTop);

    // --- Step 2: Inference (segmentation mask) ---
    cv::Mat predMask;
    if (impl_->engineLoaded) {
        predMask = runInference(preprocessed);
    }

    // --- Step 3: Post-process masks ---
    cv::Mat hMask, vMask;
    if (!predMask.empty()) {
        // UNET path: separate H/V from segmentation output
        postprocessMasks(predMask, tableImage.size(), scaleX, scaleY,
                         padLeft, padTop, hMask, vMask);
    } else {
        // Fallback: use morphology directly on the original image
        LOG_INFO("Using morphology fallback (no UNET)");
        cv::Mat gray, edges;
        if (tableImage.channels() == 3) {
            cv::cvtColor(tableImage, gray, cv::COLOR_BGR2GRAY);
        } else {
            gray = tableImage.clone();
        }

        // Adaptive threshold for better line detection
        cv::adaptiveThreshold(gray, edges, 255,
                              cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                              cv::THRESH_BINARY_INV, 15, 10);

        int hKernelW = std::max(20, tableImage.cols / 4);
        int vKernelH = std::max(20, tableImage.rows / 4);

        cv::Mat hKernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(hKernelW, 1));
        cv::Mat vKernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(1, vKernelH));

        cv::morphologyEx(edges, hMask, cv::MORPH_OPEN, hKernel);
        cv::morphologyEx(edges, vMask, cv::MORPH_OPEN, vKernel);

        // Dilate slightly to connect broken lines
        cv::Mat dilateKernel = cv::getStructuringElement(
            cv::MORPH_RECT, cv::Size(3, 3));
        cv::dilate(hMask, hMask, dilateKernel, cv::Point(-1, -1), 1);
        cv::dilate(vMask, vMask, dilateKernel, cv::Point(-1, -1), 1);
    }

    // --- Step 4: Extract line segments ---
    std::vector<LineSegment> hLines = extractLineSegments(hMask, true);
    std::vector<LineSegment> vLines = extractLineSegments(vMask, false);

    LOG_INFO("Detected {} horizontal lines, {} vertical lines",
             hLines.size(), vLines.size());

    if (hLines.size() < 2 || vLines.size() < 2) {
        LOG_WARN("Insufficient lines for table structure (H={}, V={})",
                 hLines.size(), vLines.size());
        result.supported = false;
        result.html = "";
        return result;
    }

    // --- Step 5: Adjust lines ---
    adjustLines(hLines, vLines, tableImage.size());

    // --- Step 6: Extract cells ---
    std::vector<TableCell> cells = extractCells(hLines, vLines, tableImage.size());
    LOG_INFO("Extracted {} cells", cells.size());

    if (cells.empty()) {
        LOG_WARN("No cells extracted");
        result.supported = false;
        return result;
    }

    // --- Step 7: Recover logical structure ---
    std::vector<LogicCell> logicCells = recoverLogicStructure(cells, tableImage.size());

    // Propagate logical positions back to TableCell for downstream consumers
    if (!cells.empty() && !logicCells.empty()) {
        const size_t count = std::min(cells.size(), logicCells.size());
        for (size_t i = 0; i < count; ++i) {
            const auto& lc = logicCells[i];
            cells[i].row = lc.rowStart;
            cells[i].col = lc.colStart;
            cells[i].rowSpan = std::max(1, lc.rowEnd - lc.rowStart);
            cells[i].colSpan = std::max(1, lc.colEnd - lc.colStart);
        }
    }

    // --- Step 8: Generate HTML ---
    result.html = generateHtml(logicCells);
    result.cells = cells;
    result.supported = true;

    LOG_INFO("Table recognition complete: {} cells, HTML length={}",
             cells.size(), result.html.size());

    return result;
}

// ============================================================================
// estimateTableType — heuristic classification
// ============================================================================
TableType TableRecognizer::estimateTableType(const cv::Mat& tableImage) {
    if (tableImage.empty()) return TableType::UNKNOWN;

    cv::Mat gray, edges;
    if (tableImage.channels() == 3) {
        cv::cvtColor(tableImage, gray, cv::COLOR_BGR2GRAY);
    } else if (tableImage.channels() == 4) {
        cv::cvtColor(tableImage, gray, cv::COLOR_BGRA2GRAY);
    } else {
        gray = tableImage;
    }

    cv::Canny(gray, edges, 50, 150);

    int hKernelW = std::max(1, tableImage.cols / 4);
    int vKernelH = std::max(1, tableImage.rows / 4);

    cv::Mat horizontal;
    cv::Mat horizontalKernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(hKernelW, 1));
    cv::morphologyEx(edges, horizontal, cv::MORPH_OPEN, horizontalKernel);

    cv::Mat vertical;
    cv::Mat verticalKernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(1, vKernelH));
    cv::morphologyEx(edges, vertical, cv::MORPH_OPEN, verticalKernel);

    int hLinePixels = cv::countNonZero(horizontal);
    int vLinePixels = cv::countNonZero(vertical);
    int totalPixels = tableImage.cols * tableImage.rows;

    if (totalPixels == 0) return TableType::UNKNOWN;

    float lineRatio = static_cast<float>(hLinePixels + vLinePixels) / totalPixels;

    LOG_DEBUG("Table type estimation: lineRatio={:.4f} (H={}, V={}, total={})",
              lineRatio, hLinePixels, vLinePixels, totalPixels);

    if (lineRatio > 0.01f) {
        return TableType::WIRED;
    } else {
        return TableType::WIRELESS;
    }
}

// Helper to parse SLANet HTML tokens into TableCells
static std::vector<TableCell> decodeLogicPoints(const std::vector<std::string>& tokens,
                                                const std::vector<cv::Rect>& bboxes) {
    std::vector<TableCell> cells;
    int currentRow = 0;
    int currentCol = 0;
    std::set<std::pair<int, int>> occupied;

    auto isOccupied = [&](int r, int c) {
        return occupied.find({r, c}) != occupied.end();
    };

    auto markOccupied = [&](int r, int c, int rs, int cs) {
        for (int i = 0; i < rs; ++i) {
            for (int j = 0; j < cs; ++j) {
                occupied.insert({r + i, c + j});
            }
        }
    };

    size_t bboxIdx = 0;
    for (size_t i = 0; i < tokens.size(); ++i) {
        const std::string& token = tokens[i];

        if (token == "<tr>") {
            currentCol = 0;
        } else if (token == "</tr>") {
            currentRow++;
        } else if (token.find("<td") == 0) {
            int rowSpan = 1;
            int colSpan = 1;
            
            size_t j = i;
            if (token != "<td></td>") {
                j++;
                while (j < tokens.size() && tokens[j].find('>') == std::string::npos) {
                    const std::string& attr = tokens[j];
                    if (attr.find("colspan=") != std::string::npos) {
                        size_t pos = attr.find("=");
                        std::string val = attr.substr(pos + 1);
                        val.erase(std::remove(val.begin(), val.end(), '"'), val.end());
                        val.erase(std::remove(val.begin(), val.end(), '\''), val.end());
                        try { colSpan = std::stoi(val); } catch(...) {}
                    } else if (attr.find("rowspan=") != std::string::npos) {
                        size_t pos = attr.find("=");
                        std::string val = attr.substr(pos + 1);
                        val.erase(std::remove(val.begin(), val.end(), '"'), val.end());
                        val.erase(std::remove(val.begin(), val.end(), '\''), val.end());
                        try { rowSpan = std::stoi(val); } catch(...) {}
                    }
                    j++;
                }
            }
            i = j;

            while (isOccupied(currentRow, currentCol)) {
                currentCol++;
            }

            TableCell cell;
            cell.row = currentRow;
            cell.col = currentCol;
            cell.rowSpan = rowSpan;
            cell.colSpan = colSpan;

            if (bboxIdx < bboxes.size()) {
                cell.x0 = bboxes[bboxIdx].x;
                cell.y0 = bboxes[bboxIdx].y;
                cell.x1 = bboxes[bboxIdx].x + bboxes[bboxIdx].width;
                cell.y1 = bboxes[bboxIdx].y + bboxes[bboxIdx].height;
                bboxIdx++;
            }

            cells.push_back(cell);
            markOccupied(currentRow, currentCol, rowSpan, colSpan);
        }
    }
    return cells;
}

// recognizeWireless — SLANet execution
// ============================================================================
TableResult TableRecognizer::recognizeWireless(const cv::Mat& tableImage) {
    TableResult result;
    result.type = TableType::WIRELESS;

    if (!initialized_ || tableImage.empty()) {
        result.supported = false;
        return result;
    }

#ifdef HAS_ONNXRUNTIME
    if (impl_->slanetSession && !impl_->slanetDict.empty()) {
        int originalH = tableImage.rows;
        int originalW = tableImage.cols;
        int maxLen = 488;
        
        float ratio = static_cast<float>(maxLen) / std::max(originalH, originalW);
        int resizeH = static_cast<int>(originalH * ratio);
        int resizeW = static_cast<int>(originalW * ratio);

        cv::Mat resized;
        cv::resize(tableImage, resized, cv::Size(resizeW, resizeH));

        cv::Mat rgb;
        if (resized.channels() == 4) {
            cv::cvtColor(resized, rgb, cv::COLOR_BGRA2RGB);
        } else if (resized.channels() == 1) {
            cv::cvtColor(resized, rgb, cv::COLOR_GRAY2RGB);
        } else {
            cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        }
        
        rgb.convertTo(rgb, CV_32FC3, 1.0f / 255.0f);

        // Normalize
        std::vector<cv::Mat> channels(3);
        cv::split(rgb, channels);
        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float std_val[3] = {0.229f, 0.224f, 0.225f};
        for (int i = 0; i < 3; ++i) {
            channels[i] = (channels[i] - mean[i]) / std_val[i];
        }
        cv::merge(channels, rgb);

        // Pad
        cv::Mat padded = cv::Mat::zeros(maxLen, maxLen, CV_32FC3);
        rgb.copyTo(padded(cv::Rect(0, 0, resizeW, resizeH)));

        // HWC to CHW
        std::vector<float> inputTensorValues(1 * 3 * maxLen * maxLen);
        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < maxLen; ++h) {
                for (int w = 0; w < maxLen; ++w) {
                    inputTensorValues[c * maxLen * maxLen + h * maxLen + w] =
                        padded.at<cv::Vec3f>(h, w)[c];
                }
            }
        }

        std::vector<int64_t> inputDims = {1, 3, maxLen, maxLen};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorValues.size(),
            inputDims.data(), inputDims.size());

        std::vector<const char*> inputNamesPtrs;
        for (const auto& s : impl_->slanetInputNames) inputNamesPtrs.push_back(s.c_str());
        std::vector<const char*> outputNamesPtrs;
        for (const auto& s : impl_->slanetOutputNames) outputNamesPtrs.push_back(s.c_str());

        try {
            auto outputTensors = impl_->slanetSession->Run(
                Ort::RunOptions{nullptr},
                inputNamesPtrs.data(),
                &inputTensor,
                1,
                outputNamesPtrs.data(),
                outputNamesPtrs.size());

            // loc_preds and structure_probs
            // We need to find which is which. Assume outputNamesPtrs[0] is loc_preds (bbox) and [1] is structure_probs (tokens)
            // Or we check shapes
            int locIdx = -1;
            int probIdx = -1;
            for (size_t i = 0; i < outputTensors.size(); ++i) {
                auto shape = outputTensors[i].GetTensorTypeAndShapeInfo().GetShape();
                if (shape.size() == 3 && shape[2] == 4) locIdx = i; // [1, seq_len, 4]
                else probIdx = i; // [1, seq_len, vocab_size]
            }

            if (locIdx >= 0 && probIdx >= 0) {
                float* locData = outputTensors[locIdx].GetTensorMutableData<float>();
                float* probData = outputTensors[probIdx].GetTensorMutableData<float>();
                
                auto probShape = outputTensors[probIdx].GetTensorTypeAndShapeInfo().GetShape();
                int seqLen = probShape[1];
                int vocabSize = probShape[2];

                std::vector<std::string> tokens;
                std::vector<cv::Rect> bboxes;
                std::ostringstream html;
                
                html << "<table>\n";

                // SOS is typically 0, EOS is typically size-1. 
                int eosIdx = -1;
                for (size_t i = 0; i < impl_->slanetDict.size(); ++i) {
                    if (impl_->slanetDict[i] == "eos") eosIdx = i;
                }
                if (eosIdx == -1) eosIdx = impl_->slanetDict.size() - 1;

                for (int i = 0; i < seqLen; ++i) {
                    // Argmax
                    int bestClass = 0;
                    float bestProb = -1.0f;
                    for (int v = 0; v < vocabSize; ++v) {
                        float p = probData[i * vocabSize + v];
                        if (p > bestProb) {
                            bestProb = p;
                            bestClass = v;
                        }
                    }

                    if (bestClass == eosIdx) break;
                    
                    if (bestClass > 0 && bestClass < static_cast<int>(impl_->slanetDict.size())) {
                        std::string token = impl_->slanetDict[bestClass];
                        if (token != "sos" && token != "eos") {
                            tokens.push_back(token);
                            
                            // Reconstruct HTML for fallback/display
                            if (token.find("<td") == 0 || token == "<td></td>") {
                                float x0 = locData[i * 4 + 0];
                                float y0 = locData[i * 4 + 1];
                                float x1 = locData[i * 4 + 2];
                                float y1 = locData[i * 4 + 3];
                                
                                // BBox decode (multiply by original w, h)
                                int px0 = static_cast<int>(x0 * originalW);
                                int py0 = static_cast<int>(y0 * originalH);
                                int px1 = static_cast<int>(x1 * originalW);
                                int py1 = static_cast<int>(y1 * originalH);

                                // Clamp
                                px0 = std::max(0, std::min(px0, originalW - 1));
                                py0 = std::max(0, std::min(py0, originalH - 1));
                                px1 = std::max(0, std::min(px1, originalW));
                                py1 = std::max(0, std::min(py1, originalH));

                                bboxes.push_back(cv::Rect(px0, py0, px1 - px0, py1 - py0));
                            }
                        }
                    }
                }

                // Decode HTML roughly
                for (const auto& t : tokens) {
                    if (t == "<tr>") html << "  <tr>\n";
                    else if (t == "</tr>") html << "  </tr>\n";
                    else if (t == "<td>") html << "    <td></td>\n";
                    else if (t == "<td></td>") html << "    <td></td>\n";
                    else if (t.find("<td") == 0) html << "    " << t;
                    else if (t == ">") html << "></td>\n";
                    else html << t;
                }
                html << "</table>";

                result.html = html.str();
                result.cells = decodeLogicPoints(tokens, bboxes);
                result.supported = true;
                return result;
            }
        } catch (const std::exception& e) {
            LOG_WARN("Wireless table recognition failed: {}", e.what());
        }
    }
#else
    LOG_WARN("Wireless table recognition skipped: ONNX Runtime not available");
#endif

    result.supported = false;
    return result;
}

// ============================================================================
// classifyTableType
// ============================================================================
TableType TableRecognizer::classifyTableType(const cv::Mat& tableImage) {
    if (!initialized_) {
        return estimateTableType(tableImage);
    }

#ifdef HAS_ONNXRUNTIME
    if (impl_->clsSession) {
        cv::Mat resized;
        cv::resize(tableImage, resized, cv::Size(224, 224));
        
        cv::Mat rgb;
        if (resized.channels() == 4) {
            cv::cvtColor(resized, rgb, cv::COLOR_BGRA2RGB);
        } else if (resized.channels() == 1) {
            cv::cvtColor(resized, rgb, cv::COLOR_GRAY2RGB);
        } else {
            cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
        }
        
        rgb.convertTo(rgb, CV_32FC3, 1.0f / 255.0f);

        std::vector<float> inputTensorValues(1 * 3 * 224 * 224);
        const float mean[3] = {0.485f, 0.456f, 0.406f};
        const float std_val[3] = {0.229f, 0.224f, 0.225f};

        for (int c = 0; c < 3; ++c) {
            for (int h = 0; h < 224; ++h) {
                for (int w = 0; w < 224; ++w) {
                    inputTensorValues[c * 224 * 224 + h * 224 + w] =
                        (rgb.at<cv::Vec3f>(h, w)[c] - mean[c]) / std_val[c];
                }
            }
        }

        std::vector<int64_t> inputDims = {1, 3, 224, 224};
        Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memoryInfo, inputTensorValues.data(), inputTensorValues.size(),
            inputDims.data(), inputDims.size());

        std::vector<const char*> inputNamesPtrs;
        for (const auto& s : impl_->clsInputNames) inputNamesPtrs.push_back(s.c_str());
        std::vector<const char*> outputNamesPtrs;
        for (const auto& s : impl_->clsOutputNames) outputNamesPtrs.push_back(s.c_str());

        try {
            auto outputTensors = impl_->clsSession->Run(
                Ort::RunOptions{nullptr},
                inputNamesPtrs.data(),
                &inputTensor,
                1,
                outputNamesPtrs.data(),
                outputNamesPtrs.size());

            if (!outputTensors.empty()) {
                float* floatArr = outputTensors.front().GetTensorMutableData<float>();
                size_t count = outputTensors.front().GetTensorTypeAndShapeInfo().GetElementCount();
                if (count >= 2) {
                    // Usually paddle_cls for table outputs: [0] = wired, [1] = wireless
                    if (floatArr[0] > floatArr[1]) {
                        return TableType::WIRED;
                    } else {
                        return TableType::WIRELESS;
                    }
                }
            }
        } catch (const std::exception& e) {
            LOG_WARN("Table classification failed: {}", e.what());
        }
    }
#endif

    // Fallback
    return estimateTableType(tableImage);
}

// ============================================================================
// Preprocess — resize with padding to inputSize x inputSize
// ============================================================================
cv::Mat TableRecognizer::preprocess(const cv::Mat& image,
                                     float& scaleX, float& scaleY,
                                     int& padLeft, int& padTop) {
    int targetSize = config_.inputSize;  // 768
    int h = image.rows;
    int w = image.cols;

    // Compute scale to fit within targetSize while maintaining aspect ratio
    float scale = std::min(static_cast<float>(targetSize) / w,
                           static_cast<float>(targetSize) / h);

    int newW = static_cast<int>(w * scale);
    int newH = static_cast<int>(h * scale);

    // Resize
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(newW, newH), 0, 0, cv::INTER_LINEAR);

    // Create white canvas (padding color = 255)
    cv::Mat padded(targetSize, targetSize, image.type(), cv::Scalar(255, 255, 255));

    // Center the resized image
    padLeft = (targetSize - newW) / 2;
    padTop = (targetSize - newH) / 2;

    resized.copyTo(padded(cv::Rect(padLeft, padTop, newW, newH)));

    // Store scale info for coordinate mapping back
    scaleX = scale;
    scaleY = scale;

    // Convert BGR → RGB for model input
    cv::Mat rgb;
    if (padded.channels() == 3) {
        cv::cvtColor(padded, rgb, cv::COLOR_BGR2RGB);
    } else {
        rgb = padded;
    }

    LOG_DEBUG("Preprocess: {}x{} → {}x{} (scale={:.3f}, pad=({},{}))",
              w, h, targetSize, targetSize, scale, padLeft, padTop);

    return rgb;
}

// ============================================================================
// Run Inference — DX Engine
// ============================================================================
cv::Mat TableRecognizer::runInference(const cv::Mat& preprocessed) {
    if (!impl_->engineLoaded) {
        LOG_WARN("DX Engine not loaded, cannot run inference");
        return cv::Mat();
    }

#ifdef HAS_DXRT
    // The UNET model outputs a segmentation mask with 3 classes:
    //   0 = background
    //   1 = horizontal line
    //   2 = vertical line
    try {
        auto chooseHwFromShape = [&](const std::vector<int64_t>& shape, int& outH, int& outW) {
            const int expected = config_.inputSize;
            int countExpected = 0;
            for (auto d : shape) if (static_cast<int>(d) == expected) ++countExpected;
            if (countExpected >= 2) {
                outH = expected;
                outW = expected;
                return;
            }
            if (shape.size() >= 3) {
                // Common layouts:
                // - [1, H, W]
                // - [1, 1, H, W]
                // - [1, H, W, 1]
                if (shape.back() == 1 && shape.size() >= 4) {
                    outH = static_cast<int>(shape[shape.size() - 3]);
                    outW = static_cast<int>(shape[shape.size() - 2]);
                } else {
                    outH = static_cast<int>(shape[shape.size() - 2]);
                    outW = static_cast<int>(shape[shape.size() - 1]);
                }
            }
        };

        auto safeCount = [&](const std::vector<int64_t>& shape) -> size_t {
            size_t c = 1;
            for (auto d : shape) {
                if (d <= 0) return 0;
                c *= static_cast<size_t>(d);
            }
            return c;
        };

        if (!preprocessed.isContinuous()) {
            cv::Mat contiguous = preprocessed.clone();
            dxrt::TensorPtrs outputs = impl_->engine->Run(contiguous.data);
            if (outputs.empty() || !outputs[0]) {
                LOG_WARN("DX Engine returned no output tensors");
                return cv::Mat();
            }

            const auto& outTensor = outputs[0];
            const std::vector<int64_t>& shape = outTensor->shape();

            // Expected shape: [1, H, W] or [1, H, W, 1], values in {0,1,2}
            int outH = config_.inputSize;
            int outW = config_.inputSize;
            chooseHwFromShape(shape, outH, outW);

            cv::Mat predMask(outH, outW, CV_8UC1);
            const void* outData = outTensor->data();
            const size_t elemCount = safeCount(shape);
            if (elemCount < static_cast<size_t>(outH) * static_cast<size_t>(outW)) {
                LOG_WARN("UNET output too small for mask: elems={}, need={}",
                         elemCount, static_cast<size_t>(outH) * static_cast<size_t>(outW));
                return cv::Mat();
            }
            if (outTensor->elem_size() == 1) {
                // uint8 output
                std::memcpy(predMask.data, outData, static_cast<size_t>(outH) * static_cast<size_t>(outW));
            } else if (outTensor->elem_size() == 4) {
                // float output — convert to uint8 class indices
                const float* fdata = static_cast<const float*>(outData);
                for (int i = 0; i < outH * outW; ++i) {
                    predMask.data[i] = static_cast<uint8_t>(std::round(fdata[i]));
                }
            } else {
                // int64 output
                const int64_t* idata = static_cast<const int64_t*>(outData);
                for (int i = 0; i < outH * outW; ++i) {
                    predMask.data[i] = static_cast<uint8_t>(idata[i]);
                }
            }

            LOG_DEBUG("UNET inference done: output shape {}x{}", outH, outW);
            return predMask;
        }

        dxrt::TensorPtrs outputs = impl_->engine->Run(preprocessed.data);
        if (outputs.empty() || !outputs[0]) {
            LOG_WARN("DX Engine returned no output tensors");
            return cv::Mat();
        }

        const auto& outTensor = outputs[0];
        const std::vector<int64_t>& shape = outTensor->shape();

        int outH = config_.inputSize;
        int outW = config_.inputSize;
        chooseHwFromShape(shape, outH, outW);

        cv::Mat predMask(outH, outW, CV_8UC1);
        const void* outData = outTensor->data();
        const size_t elemCount = safeCount(shape);
        if (elemCount < static_cast<size_t>(outH) * static_cast<size_t>(outW)) {
            LOG_WARN("UNET output too small for mask: elems={}, need={}",
                     elemCount, static_cast<size_t>(outH) * static_cast<size_t>(outW));
            return cv::Mat();
        }
        if (outTensor->elem_size() == 1) {
            std::memcpy(predMask.data, outData, static_cast<size_t>(outH) * static_cast<size_t>(outW));
        } else if (outTensor->elem_size() == 4) {
            const float* fdata = static_cast<const float*>(outData);
            for (int i = 0; i < outH * outW; ++i) {
                predMask.data[i] = static_cast<uint8_t>(std::round(fdata[i]));
            }
        } else {
            const int64_t* idata = static_cast<const int64_t*>(outData);
            for (int i = 0; i < outH * outW; ++i) {
                predMask.data[i] = static_cast<uint8_t>(idata[i]);
            }
        }

        LOG_DEBUG("UNET inference done: output shape {}x{}", outH, outW);
        return predMask;

    } catch (const std::exception& e) {
        LOG_ERROR("DX Engine inference failed: {}", e.what());
        return cv::Mat();
    }
#else
    LOG_WARN("DX Runtime not available — returning empty mask");
    return cv::Mat();
#endif
}

// ============================================================================
// Post-process masks — separate H/V, resize back, morphological cleanup
// ============================================================================
void TableRecognizer::postprocessMasks(const cv::Mat& predMask,
                                        const cv::Size& originalSize,
                                        float scaleX, float scaleY,
                                        int padLeft, int padTop,
                                        cv::Mat& hMask, cv::Mat& vMask) {
    int targetSize = config_.inputSize;

    // Calculate the region of interest (without padding)
    int newW = static_cast<int>(originalSize.width * scaleX);
    int newH = static_cast<int>(originalSize.height * scaleY);

    // Crop out the padding region
    cv::Rect roi(padLeft, padTop, newW, newH);
    roi &= cv::Rect(0, 0, targetSize, targetSize);  // clamp to bounds

    cv::Mat cropped = predMask(roi);

    // Separate H and V masks: class 1 = horizontal, class 2 = vertical
    cv::Mat hPred = (cropped == 1);  // uint8, 0 or 255
    cv::Mat vPred = (cropped == 2);

    // Resize back to original image size
    cv::resize(hPred, hMask, originalSize, 0, 0, cv::INTER_NEAREST);
    cv::resize(vPred, vMask, originalSize, 0, 0, cv::INTER_NEAREST);

    // Morphological close to connect broken line segments
    // Adaptive kernel sizes following Python: sqrt(dim) * scaleFactor
    float sf = config_.lineScaleFactor;
    int hCloseW = std::max(3, static_cast<int>(std::sqrt(originalSize.width) * sf));
    int vCloseH = std::max(3, static_cast<int>(std::sqrt(originalSize.height) * sf));

    // Horizontal lines: close with wide horizontal kernel
    cv::Mat hCloseKernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(hCloseW, 1));
    cv::morphologyEx(hMask, hMask, cv::MORPH_CLOSE, hCloseKernel);

    // Vertical lines: close with tall vertical kernel
    cv::Mat vCloseKernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(1, vCloseH));
    cv::morphologyEx(vMask, vMask, cv::MORPH_CLOSE, vCloseKernel);

    LOG_DEBUG("Postprocess masks: H close kernel={}x1, V close kernel=1x{}",
              hCloseW, vCloseH);
}

// ============================================================================
// Extract line segments from binary mask using connected components + minAreaRect
// ============================================================================
std::vector<LineSegment> TableRecognizer::extractLineSegments(const cv::Mat& lineMask,
                                                               bool isHorizontal) {
    std::vector<LineSegment> lines;

    if (lineMask.empty()) return lines;

    // Connected components with stats
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(lineMask, labels, stats, centroids, 8);

    for (int i = 1; i < numLabels; ++i) {  // skip background (label 0)
        int area = stats.at<int>(i, cv::CC_STAT_AREA);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);

        // Filter small noise components
        if (area < 50) continue;

        // For horizontal lines: width should be much larger than height
        // For vertical lines: height should be much larger than width
        float aspectRatio = isHorizontal
            ? static_cast<float>(w) / std::max(1, h)
            : static_cast<float>(h) / std::max(1, w);

        if (aspectRatio < 2.0f) continue;  // not line-like

        // Extract the component pixels for minAreaRect
        cv::Mat componentMask = (labels == i);
        std::vector<cv::Point> points;
        cv::findNonZero(componentMask, points);

        if (points.size() < 2) continue;

        cv::RotatedRect rotRect = cv::minAreaRect(points);

        // Compute line segment from rotated rect
        cv::Point2f vertices[4];
        rotRect.points(vertices);

        // The longest edge of the rotated rect defines the line direction
        float d01 = pointDistance(vertices[0], vertices[1]);
        float d12 = pointDistance(vertices[1], vertices[2]);

        LineSegment seg;
        if (d01 > d12) {
            // Edge 0-1 is longer
            seg.start = (vertices[0] + vertices[3]) * 0.5f;
            seg.end = (vertices[1] + vertices[2]) * 0.5f;
        } else {
            // Edge 1-2 is longer
            seg.start = (vertices[0] + vertices[1]) * 0.5f;
            seg.end = (vertices[2] + vertices[3]) * 0.5f;
        }

        seg.length = pointDistance(seg.start, seg.end);

        // Ensure consistent direction
        if (isHorizontal) {
            if (seg.start.x > seg.end.x) std::swap(seg.start, seg.end);
            seg.angle = std::atan2(seg.end.y - seg.start.y,
                                   seg.end.x - seg.start.x) * 180.0f / CV_PI;
        } else {
            if (seg.start.y > seg.end.y) std::swap(seg.start, seg.end);
            seg.angle = std::atan2(seg.end.y - seg.start.y,
                                   seg.end.x - seg.start.x) * 180.0f / CV_PI;
        }

        // Filter by angle: horizontal lines ~0°, vertical lines ~90°
        float absAngle = std::abs(seg.angle);
        if (isHorizontal && absAngle > 15.0f) continue;
        if (!isHorizontal && std::abs(absAngle - 90.0f) > 15.0f) continue;

        lines.push_back(seg);
    }

    // Sort: horizontal by Y, vertical by X
    if (isHorizontal) {
        std::sort(lines.begin(), lines.end(), [](const LineSegment& a, const LineSegment& b) {
            float aY = (a.start.y + a.end.y) * 0.5f;
            float bY = (b.start.y + b.end.y) * 0.5f;
            return aY < bY;
        });
    } else {
        std::sort(lines.begin(), lines.end(), [](const LineSegment& a, const LineSegment& b) {
            float aX = (a.start.x + a.end.x) * 0.5f;
            float bX = (b.start.x + b.end.x) * 0.5f;
            return aX < bX;
        });
    }

    return lines;
}

// ============================================================================
// Adjust lines — merge close lines, extend endpoints to grid intersections
// ============================================================================
void TableRecognizer::adjustLines(std::vector<LineSegment>& hLines,
                                   std::vector<LineSegment>& vLines,
                                   const cv::Size& imageSize) {
    // --- Phase 1: Merge nearby parallel lines ---
    auto mergeCloseLines = [](std::vector<LineSegment>& lines,
                              bool isHorizontal, float mergeThresh) {
        if (lines.size() < 2) return;

        std::vector<LineSegment> merged;
        std::vector<bool> used(lines.size(), false);

        for (size_t i = 0; i < lines.size(); ++i) {
            if (used[i]) continue;

            LineSegment current = lines[i];
            float currentPos = isHorizontal
                ? (current.start.y + current.end.y) * 0.5f
                : (current.start.x + current.end.x) * 0.5f;

            // Find all lines close to this one
            std::vector<size_t> group = {i};
            for (size_t j = i + 1; j < lines.size(); ++j) {
                if (used[j]) continue;
                float otherPos = isHorizontal
                    ? (lines[j].start.y + lines[j].end.y) * 0.5f
                    : (lines[j].start.x + lines[j].end.x) * 0.5f;

                if (std::abs(currentPos - otherPos) < mergeThresh) {
                    group.push_back(j);
                    used[j] = true;
                }
            }

            // Merge group: take the longest line, extend endpoints
            if (group.size() == 1) {
                merged.push_back(current);
            } else {
                // Average position, union of extent
                float avgPos = 0;
                float minStart = std::numeric_limits<float>::max();
                float maxEnd = std::numeric_limits<float>::lowest();

                for (size_t idx : group) {
                    const auto& l = lines[idx];
                    if (isHorizontal) {
                        avgPos += (l.start.y + l.end.y) * 0.5f;
                        minStart = std::min(minStart, std::min(l.start.x, l.end.x));
                        maxEnd = std::max(maxEnd, std::max(l.start.x, l.end.x));
                    } else {
                        avgPos += (l.start.x + l.end.x) * 0.5f;
                        minStart = std::min(minStart, std::min(l.start.y, l.end.y));
                        maxEnd = std::max(maxEnd, std::max(l.start.y, l.end.y));
                    }
                }
                avgPos /= group.size();

                LineSegment mergedLine;
                if (isHorizontal) {
                    mergedLine.start = cv::Point2f(minStart, avgPos);
                    mergedLine.end = cv::Point2f(maxEnd, avgPos);
                } else {
                    mergedLine.start = cv::Point2f(avgPos, minStart);
                    mergedLine.end = cv::Point2f(avgPos, maxEnd);
                }
                mergedLine.length = pointDistance(mergedLine.start, mergedLine.end);
                mergedLine.angle = isHorizontal ? 0.0f : 90.0f;
                merged.push_back(mergedLine);
            }
            used[i] = true;
        }

        lines = merged;
    };

    float hMergeThresh = imageSize.height * 0.02f;  // 2% of height
    float vMergeThresh = imageSize.width * 0.02f;   // 2% of width

    mergeCloseLines(hLines, true, std::max(5.0f, hMergeThresh));
    mergeCloseLines(vLines, false, std::max(5.0f, vMergeThresh));

    // --- Phase 2: Extend line endpoints to nearest perpendicular line ---
    // Extend horizontal lines to reach the leftmost and rightmost vertical lines
    if (!vLines.empty() && !hLines.empty()) {
        // Find the extreme vertical line positions
        float leftMostX = std::numeric_limits<float>::max();
        float rightMostX = std::numeric_limits<float>::lowest();
        for (const auto& vl : vLines) {
            float x = (vl.start.x + vl.end.x) * 0.5f;
            leftMostX = std::min(leftMostX, x);
            rightMostX = std::max(rightMostX, x);
        }

        float topMostY = std::numeric_limits<float>::max();
        float bottomMostY = std::numeric_limits<float>::lowest();
        for (const auto& hl : hLines) {
            float y = (hl.start.y + hl.end.y) * 0.5f;
            topMostY = std::min(topMostY, y);
            bottomMostY = std::max(bottomMostY, y);
        }

        // Extend each horizontal line to span the full vertical range
        float extendTolerance = imageSize.width * 0.05f;
        for (auto& hl : hLines) {
            if (hl.start.x - leftMostX < extendTolerance) {
                hl.start.x = leftMostX;
            }
            if (rightMostX - hl.end.x < extendTolerance) {
                hl.end.x = rightMostX;
            }
            hl.length = pointDistance(hl.start, hl.end);
        }

        // Extend each vertical line similarly
        for (auto& vl : vLines) {
            if (vl.start.y - topMostY < extendTolerance) {
                vl.start.y = topMostY;
            }
            if (bottomMostY - vl.end.y < extendTolerance) {
                vl.end.y = bottomMostY;
            }
            vl.length = pointDistance(vl.start, vl.end);
        }
    }

    LOG_DEBUG("After adjustment: {} H-lines, {} V-lines", hLines.size(), vLines.size());
}

// ============================================================================
// Extract cells — draw grid lines, find connected regions as cells
// ============================================================================
std::vector<TableCell> TableRecognizer::extractCells(
        const std::vector<LineSegment>& hLines,
        const std::vector<LineSegment>& vLines,
        const cv::Size& imageSize) {

    std::vector<TableCell> cells;

    if (hLines.size() < 2 || vLines.size() < 2) return cells;

    // Method: Draw all lines on a blank image, then find enclosed regions
    // using connected components on the inverse.

    cv::Mat lineCanvas = cv::Mat::zeros(imageSize, CV_8UC1);

    // Draw horizontal lines (white, thickness=2)
    for (const auto& hl : hLines) {
        cv::line(lineCanvas,
                 cv::Point(static_cast<int>(hl.start.x), static_cast<int>(hl.start.y)),
                 cv::Point(static_cast<int>(hl.end.x), static_cast<int>(hl.end.y)),
                 cv::Scalar(255), 2);
    }

    // Draw vertical lines
    for (const auto& vl : vLines) {
        cv::line(lineCanvas,
                 cv::Point(static_cast<int>(vl.start.x), static_cast<int>(vl.start.y)),
                 cv::Point(static_cast<int>(vl.end.x), static_cast<int>(vl.end.y)),
                 cv::Scalar(255), 2);
    }

    // Invert: the enclosed regions become white
    cv::Mat invCanvas;
    cv::bitwise_not(lineCanvas, invCanvas);

    // Connected components on inverted image
    cv::Mat labels, stats, centroids;
    int numLabels = cv::connectedComponentsWithStats(invCanvas, labels, stats, centroids, 4);

    // The table bounding box: use the extent of lines
    float tableLeft = std::numeric_limits<float>::max();
    float tableRight = std::numeric_limits<float>::lowest();
    float tableTop = std::numeric_limits<float>::max();
    float tableBottom = std::numeric_limits<float>::lowest();

    for (const auto& hl : hLines) {
        tableLeft = std::min({tableLeft, hl.start.x, hl.end.x});
        tableRight = std::max({tableRight, hl.start.x, hl.end.x});
        tableTop = std::min({tableTop, hl.start.y, hl.end.y});
        tableBottom = std::max({tableBottom, hl.start.y, hl.end.y});
    }
    for (const auto& vl : vLines) {
        tableLeft = std::min({tableLeft, vl.start.x, vl.end.x});
        tableRight = std::max({tableRight, vl.start.x, vl.end.x});
        tableTop = std::min({tableTop, vl.start.y, vl.end.y});
        tableBottom = std::max({tableBottom, vl.start.y, vl.end.y});
    }

    cv::Rect tableBounds(
        static_cast<int>(tableLeft), static_cast<int>(tableTop),
        static_cast<int>(tableRight - tableLeft),
        static_cast<int>(tableBottom - tableTop));

    float tableArea = tableBounds.area();
    float minCellArea = tableArea * 0.001f;   // minimum 0.1% of table
    float maxCellArea = tableArea * 0.95f;    // skip background-like regions

    for (int i = 1; i < numLabels; ++i) {
        int x = stats.at<int>(i, cv::CC_STAT_LEFT);
        int y = stats.at<int>(i, cv::CC_STAT_TOP);
        int w = stats.at<int>(i, cv::CC_STAT_WIDTH);
        int h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
        int area = stats.at<int>(i, cv::CC_STAT_AREA);

        cv::Rect cellRect(x, y, w, h);

        // Filter: must be inside table bounds, reasonable size
        cv::Rect intersection = cellRect & tableBounds;
        if (intersection.area() < cellRect.area() * 0.5) continue;
        if (area < minCellArea || area > maxCellArea) continue;
        if (w < 5 || h < 5) continue;

        TableCell cell;
        cell.x0 = static_cast<float>(x);
        cell.y0 = static_cast<float>(y);
        cell.x1 = static_cast<float>(x + w);
        cell.y1 = static_cast<float>(y + h);
        cell.row = 0;      // will be refined in recoverLogicStructure
        cell.col = 0;
        cell.rowSpan = 1;
        cell.colSpan = 1;
        cells.push_back(cell);
    }

    // Sort cells: top-to-bottom, left-to-right
    std::sort(cells.begin(), cells.end(), [](const TableCell& a, const TableCell& b) {
        if (std::abs(a.y0 - b.y0) > 5) return a.y0 < b.y0;
        return a.x0 < b.x0;
    });

    return cells;
}

// ============================================================================
// Recover logical structure — assign row/col indices, detect spans
// ============================================================================
std::vector<LogicCell> TableRecognizer::recoverLogicStructure(
        const std::vector<TableCell>& cells,
        const cv::Size& imageSize) {

    (void)imageSize;
    if (cells.empty()) return {};

    // --- Step 1: Group cells into rows by Y-coordinate clustering ---
    // Collect all cell center Y coordinates
    std::vector<float> cellCenterYs;
    for (const auto& c : cells) {
        cellCenterYs.push_back((c.y0 + c.y1) * 0.5f);
    }

    // Cluster Y values into rows (simple gap-based clustering)
    std::vector<float> sortedYs = cellCenterYs;
    std::sort(sortedYs.begin(), sortedYs.end());

    // Find row boundaries: detect significant gaps
    std::vector<std::vector<int>> rowGroups;  // each group is a list of cell indices
    float rowMergeThresh = 0;

    // Use median cell height as merge threshold
    std::vector<int> cellHeights;
    for (const auto& c : cells) cellHeights.push_back(static_cast<int>(c.y1 - c.y0));
    std::sort(cellHeights.begin(), cellHeights.end());
    if (!cellHeights.empty()) {
        rowMergeThresh = cellHeights[cellHeights.size() / 2] * 0.3f;
    }
    rowMergeThresh = std::max(rowMergeThresh, 5.0f);

    // Assign each cell to a row
    std::vector<int> cellRowIdx(cells.size(), -1);
    std::vector<float> rowYValues;

    for (size_t i = 0; i < cells.size(); ++i) {
        float cy = (cells[i].y0 + cells[i].y1) * 0.5f;
        int foundRow = -1;
        for (size_t r = 0; r < rowYValues.size(); ++r) {
            if (std::abs(cy - rowYValues[r]) < rowMergeThresh) {
                foundRow = static_cast<int>(r);
                break;
            }
        }
        if (foundRow < 0) {
            foundRow = static_cast<int>(rowYValues.size());
            rowYValues.push_back(cy);
        }
        cellRowIdx[i] = foundRow;
    }

    // Sort row Y values and remap
    std::vector<size_t> rowOrder(rowYValues.size());
    std::iota(rowOrder.begin(), rowOrder.end(), 0);
    std::sort(rowOrder.begin(), rowOrder.end(), [&](size_t a, size_t b) {
        return rowYValues[a] < rowYValues[b];
    });
    std::vector<int> rowRemap(rowYValues.size());
    for (size_t i = 0; i < rowOrder.size(); ++i) {
        rowRemap[rowOrder[i]] = static_cast<int>(i);
    }
    for (auto& ri : cellRowIdx) {
        ri = rowRemap[ri];
    }
    int numRows = static_cast<int>(rowYValues.size());

    // --- Step 2: Determine column positions ---
    // Collect all cell left-edge X coordinates, cluster into columns
    std::vector<float> cellCenterXs;
    for (const auto& c : cells) {
        cellCenterXs.push_back((c.x0 + c.x1) * 0.5f);
    }

    std::vector<int> cellWidths;
    for (const auto& c : cells) cellWidths.push_back(static_cast<int>(c.x1 - c.x0));
    std::sort(cellWidths.begin(), cellWidths.end());
    float colMergeThresh = 5.0f;
    if (!cellWidths.empty()) {
        colMergeThresh = cellWidths[cellWidths.size() / 2] * 0.3f;
    }
    colMergeThresh = std::max(colMergeThresh, 5.0f);

    std::vector<int> cellColIdx(cells.size(), -1);
    std::vector<float> colXValues;

    for (size_t i = 0; i < cells.size(); ++i) {
        float cx = (cells[i].x0 + cells[i].x1) * 0.5f;
        int foundCol = -1;
        for (size_t c = 0; c < colXValues.size(); ++c) {
            if (std::abs(cx - colXValues[c]) < colMergeThresh) {
                foundCol = static_cast<int>(c);
                break;
            }
        }
        if (foundCol < 0) {
            foundCol = static_cast<int>(colXValues.size());
            colXValues.push_back(cx);
        }
        cellColIdx[i] = foundCol;
    }

    // Sort column X values and remap
    std::vector<size_t> colOrder(colXValues.size());
    std::iota(colOrder.begin(), colOrder.end(), 0);
    std::sort(colOrder.begin(), colOrder.end(), [&](size_t a, size_t b) {
        return colXValues[a] < colXValues[b];
    });
    std::vector<int> colRemap(colXValues.size());
    for (size_t i = 0; i < colOrder.size(); ++i) {
        colRemap[colOrder[i]] = static_cast<int>(i);
    }
    for (auto& ci : cellColIdx) {
        ci = colRemap[ci];
    }
    int numCols = static_cast<int>(colXValues.size());

    // --- Step 3: Detect spans (rowspan/colspan) ---
    // Build a grid to detect spanning cells
    // A cell that covers multiple row/col positions implies spanning.
    // Use sorted row Y boundaries and col X boundaries.

    std::vector<float> sortedRowYs(numRows);
    for (size_t i = 0; i < rowOrder.size(); ++i) {
        sortedRowYs[i] = rowYValues[rowOrder[i]];
    }
    std::vector<float> sortedColXs(numCols);
    for (size_t i = 0; i < colOrder.size(); ++i) {
        sortedColXs[i] = colXValues[colOrder[i]];
    }

    // --- Step 4: Build LogicCells ---
    std::vector<LogicCell> logicCells;
    logicCells.reserve(cells.size());

    for (size_t i = 0; i < cells.size(); ++i) {
        LogicCell lc;
        lc.rowStart = cellRowIdx[i];
        lc.colStart = cellColIdx[i];
        lc.x0 = cells[i].x0;
        lc.y0 = cells[i].y0;
        lc.x1 = cells[i].x1;
        lc.y1 = cells[i].y1;

        // Detect rowspan: how many row Y-values does this cell's vertical extent cover?
        float cellTop = cells[i].y0;
        float cellBottom = cells[i].y1;
        int rspan = 0;
        for (int r = lc.rowStart; r < numRows; ++r) {
            if (sortedRowYs[r] >= cellTop - rowMergeThresh &&
                sortedRowYs[r] <= cellBottom + rowMergeThresh) {
                rspan++;
            } else if (sortedRowYs[r] > cellBottom + rowMergeThresh) {
                break;
            }
        }
        lc.rowEnd = lc.rowStart + std::max(1, rspan);

        // Detect colspan
        float cellLeft = cells[i].x0;
        float cellRight = cells[i].x1;
        int cspan = 0;
        for (int c = lc.colStart; c < numCols; ++c) {
            if (sortedColXs[c] >= cellLeft - colMergeThresh &&
                sortedColXs[c] <= cellRight + colMergeThresh) {
                cspan++;
            } else if (sortedColXs[c] > cellRight + colMergeThresh) {
                break;
            }
        }
        lc.colEnd = lc.colStart + std::max(1, cspan);

        logicCells.push_back(lc);
    }

    LOG_DEBUG("Logic structure: {} rows x {} cols, {} cells",
              numRows, numCols, logicCells.size());

    return logicCells;
}

// ============================================================================
// Generate HTML from logical cells
// ============================================================================
std::string TableRecognizer::generateHtml(const std::vector<LogicCell>& logicCells) {
    if (logicCells.empty()) return "";

    // Find grid dimensions
    int maxRow = 0, maxCol = 0;
    for (const auto& lc : logicCells) {
        maxRow = std::max(maxRow, lc.rowEnd);
        maxCol = std::max(maxCol, lc.colEnd);
    }

    // Build an occupancy grid to handle spans properly
    // occupied[row][col] = true if already covered by a spanning cell
    std::vector<std::vector<bool>> occupied(maxRow, std::vector<bool>(maxCol, false));

    // Organize cells by (rowStart, colStart)
    std::map<std::pair<int, int>, const LogicCell*> cellMap;
    for (const auto& lc : logicCells) {
        cellMap[{lc.rowStart, lc.colStart}] = &lc;
    }

    std::ostringstream html;
    html << "<table border=\"1\">\n";

    for (int r = 0; r < maxRow; ++r) {
        html << "  <tr>\n";
        for (int c = 0; c < maxCol; ++c) {
            if (occupied[r][c]) continue;

            auto it = cellMap.find({r, c});
            if (it != cellMap.end()) {
                const LogicCell* lc = it->second;
                int rowSpan = lc->rowEnd - lc->rowStart;
                int colSpan = lc->colEnd - lc->colStart;

                // Mark occupied cells
                for (int dr = 0; dr < rowSpan; ++dr) {
                    for (int dc = 0; dc < colSpan; ++dc) {
                        int nr = r + dr, nc = c + dc;
                        if (nr < maxRow && nc < maxCol) {
                            occupied[nr][nc] = true;
                        }
                    }
                }

                html << "    <td";
                if (rowSpan > 1) html << " rowspan=\"" << rowSpan << "\"";
                if (colSpan > 1) html << " colspan=\"" << colSpan << "\"";
                html << ">";

                // Escape HTML special characters in text
                std::string escapedText;
                for (char ch : lc->text) {
                    switch (ch) {
                        case '<': escapedText += "&lt;"; break;
                        case '>': escapedText += "&gt;"; break;
                        case '&': escapedText += "&amp;"; break;
                        case '"': escapedText += "&quot;"; break;
                        default: escapedText += ch;
                    }
                }
                html << escapedText;
                html << "</td>\n";
            } else {
                // Empty cell (not covered by any detected cell)
                occupied[r][c] = true;
                html << "    <td></td>\n";
            }
        }
        html << "  </tr>\n";
    }

    html << "</table>";
    return html.str();
}

// ============================================================================
// Utility functions
// ============================================================================

float TableRecognizer::pointDistance(const cv::Point2f& a, const cv::Point2f& b) {
    float dx = a.x - b.x;
    float dy = a.y - b.y;
    return std::sqrt(dx * dx + dy * dy);
}

bool TableRecognizer::isBoxContained(const cv::Rect& inner, const cv::Rect& outer,
                                      float threshold) {
    cv::Rect intersection = inner & outer;
    if (intersection.empty()) return false;
    float ratio = static_cast<float>(intersection.area()) / std::max(1, inner.area());
    return ratio >= threshold;
}

cv::Point2f TableRecognizer::lineIntersection(const LineSegment& l1,
                                               const LineSegment& l2,
                                               bool& found) {
    // Line 1: P1 + t*(P2-P1)
    // Line 2: P3 + u*(P4-P3)
    float x1 = l1.start.x, y1 = l1.start.y;
    float x2 = l1.end.x, y2 = l1.end.y;
    float x3 = l2.start.x, y3 = l2.start.y;
    float x4 = l2.end.x, y4 = l2.end.y;

    float denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
    if (std::abs(denom) < 1e-6f) {
        found = false;
        return cv::Point2f(0, 0);
    }

    float t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom;

    found = true;
    return cv::Point2f(x1 + t * (x2 - x1), y1 + t * (y2 - y1));
}

} // namespace rapid_doc