/**
 * @file table_wireless_recognizer.cpp
 * @brief Wireless (SLANet+) table backend — C++ port of RapidDoc's Python
 *        reference. Parity targets, in order:
 *
 *   Python file                                                      → C++ fn
 *   -----------------------------------------------------------------  -------
 *   table_structure/pp_structure/pre_process.py::TablePreprocess       preprocess()
 *   table_structure/pp_structure/main.py::PPTableStructurer.__call__   runInference()
 *   table_structure/pp_structure/post_process.py::TableLabelDecode     decode()
 *   table_structure/pp_structure/main.py::rescale_cell_bboxes          rescaleCellBboxes()
 *   table_structure/pp_structure/main.py::filter_blank_bbox            filterBlankBboxes()
 *   table_structure/utils.py::get_struct_str                           wrapStructure()
 *   table_matcher/main.py::TableMatch                                  matchAndEmitHtml()
 *   table_matcher/utils.py::distance / compute_iou                     l1Distance / iou
 */

#include "table/table_wireless_recognizer.h"

#include "common/logger.h"

#ifdef HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace fs = std::filesystem;

namespace rapid_doc {

namespace {

constexpr float kMean[3] = {0.485f, 0.456f, 0.406f};
constexpr float kStd[3]  = {0.229f, 0.224f, 0.225f};

struct CellAabb {
    float x1 = 0.0f, y1 = 0.0f, x2 = 0.0f, y2 = 0.0f;
};

// L1 corner distance + min(head, tail) corner distance — byte-for-byte
// translation of table_matcher/utils.py::distance.
float l1Distance(const CellAabb& a, const CellAabb& b) {
    const float head = std::fabs(b.x1 - a.x1) + std::fabs(b.y1 - a.y1);
    const float tail = std::fabs(b.x2 - a.x2) + std::fabs(b.y2 - a.y2);
    return (head + tail) + std::min(head, tail);
}

// Axis-aligned IoU — translation of table_matcher/utils.py::compute_iou. The
// Python code's docstring mislabels axes but the math works symmetrically
// because it uses indices 0/2 and 1/3 uniformly.
float axisIou(const CellAabb& a, const CellAabb& b) {
    const float aArea = std::max(0.0f, a.x2 - a.x1) * std::max(0.0f, a.y2 - a.y1);
    const float bArea = std::max(0.0f, b.x2 - b.x1) * std::max(0.0f, b.y2 - b.y1);
    const float sumArea = aArea + bArea;
    const float left   = std::max(a.x1, b.x1);
    const float right  = std::min(a.x2, b.x2);
    const float top    = std::max(a.y1, b.y1);
    const float bottom = std::min(a.y2, b.y2);
    if (left >= right || top >= bottom) {
        return 0.0f;
    }
    const float inter = (right - left) * (bottom - top);
    const float denom = sumArea - inter;
    if (denom <= 0.0f) {
        return 0.0f;
    }
    return inter / denom;
}

// Trim/strip CR/LF whitespace tokens from the character list the same way
// Python's str.splitlines() does (the ONNX metadata stores tokens one per
// line, but the last line may be empty).
std::vector<std::string> splitLinesKeepEmpty(const std::string& raw) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : raw) {
        if (c == '\n') {
            out.push_back(cur);
            cur.clear();
        } else if (c != '\r') {
            cur.push_back(c);
        }
    }
    if (!cur.empty()) {
        out.push_back(cur);
    }
    // Drop trailing pure-empty sentinel if any (Python's splitlines does not
    // append one, but some serializers include a trailing \n).
    while (!out.empty() && out.back().empty()) {
        out.pop_back();
    }
    return out;
}

} // namespace

// -----------------------------------------------------------------------------

struct TableWirelessRecognizer::Impl {
#ifdef HAS_ONNXRUNTIME
    Ort::Env ortEnv{ORT_LOGGING_LEVEL_WARNING, "slanet_plus_onnx"};
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
#endif
    std::string inputName;
    // We identify outputs by last-dim: structure_probs has last dim
    // = character_.size(); bbox_preds has last dim = 8 (slanet-plus) / 4.
    std::string bboxOutputName;
    std::string structOutputName;
    int bboxCoords = 8;              // 8 for slanet-plus, 4 for plain SLANet
    int numClasses = 0;              // == character_.size()

    std::vector<std::string> character;         // final vocab incl. sos/eos
    std::unordered_set<int> ignoredIdx;         // sos + eos indices
    int eosIdx = -1;
    std::unordered_set<std::string> tdToken;    // {"<td>", "<td", "<td></td>"}
};

// -----------------------------------------------------------------------------

TableWirelessRecognizer::TableWirelessRecognizer(const TableWirelessRecognizerConfig& config)
    : config_(config)
    , impl_(std::make_unique<Impl>())
{}

TableWirelessRecognizer::~TableWirelessRecognizer() = default;

bool TableWirelessRecognizer::initialize() {
#ifndef HAS_ONNXRUNTIME
    LOG_ERROR("TableWirelessRecognizer requires ONNX Runtime.");
    return false;
#else
    if (config_.onnxModelPath.empty()) {
        LOG_WARN("Wireless table model path empty; wireless backend disabled.");
        return false;
    }
    if (!fs::exists(config_.onnxModelPath)) {
        LOG_WARN("Wireless table ONNX not found: {} (wireless backend disabled)",
                 config_.onnxModelPath);
        return false;
    }

    try {
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (config_.disableCpuMemArena) {
            opts.DisableCpuMemArena();
        }
        if (config_.intraOpThreads > 0) {
            opts.SetIntraOpNumThreads(config_.intraOpThreads);
        }
        impl_->session = std::make_unique<Ort::Session>(
            impl_->ortEnv, config_.onnxModelPath.c_str(), opts);

        Ort::AllocatorWithDefaultOptions alloc;

        {
            auto n = impl_->session->GetInputNameAllocated(0, alloc);
            impl_->inputName = n.get();
        }

        // Parity: Python code destructures as `(bbox_preds, struct_probs) = session(...)`.
        // We identify which output is which by shape (bbox ends in 8 (slanet-plus)
        // or 4 (plain SLANet); struct ends in len(character) == 50 here).
        const size_t numOutputs = impl_->session->GetOutputCount();
        if (numOutputs < 2) {
            throw std::runtime_error("Wireless model must have >= 2 outputs");
        }

        std::vector<std::string> outNames(numOutputs);
        std::vector<int64_t> lastDims(numOutputs, 0);
        for (size_t i = 0; i < numOutputs; ++i) {
            auto n = impl_->session->GetOutputNameAllocated(i, alloc);
            outNames[i] = n.get();
            auto tinfo = impl_->session->GetOutputTypeInfo(i);
            auto tshape = tinfo.GetTensorTypeAndShapeInfo().GetShape();
            if (!tshape.empty()) {
                lastDims[i] = tshape.back();
            }
        }

        // Read character metadata and build vocab before finalizing output
        // assignment — we need len(character) to disambiguate struct vs bbox.
        Ort::AllocatorWithDefaultOptions metaAlloc;
        auto meta = impl_->session->GetModelMetadata();
        auto rawChar = meta.LookupCustomMetadataMapAllocated("character", metaAlloc);
        if (!rawChar) {
            throw std::runtime_error("ONNX metadata missing 'character' key");
        }
        std::vector<std::string> dictCharacter = splitLinesKeepEmpty(rawChar.get());
        if (dictCharacter.empty()) {
            throw std::runtime_error("ONNX 'character' metadata parsed empty");
        }

        // merge_no_span_structure=True: drop "<td>" (if present) and ensure
        // "<td></td>" is present — matches Python TableLabelDecode.__init__.
        {
            auto it = std::find(dictCharacter.begin(), dictCharacter.end(), std::string("<td>"));
            if (it != dictCharacter.end()) dictCharacter.erase(it);
            if (std::find(dictCharacter.begin(), dictCharacter.end(), std::string("<td></td>"))
                == dictCharacter.end()) {
                dictCharacter.emplace_back("<td></td>");
            }
        }

        // add_special_char: prepend "sos", append "eos"
        impl_->character.clear();
        impl_->character.reserve(dictCharacter.size() + 2);
        impl_->character.emplace_back("sos");
        for (auto& c : dictCharacter) impl_->character.emplace_back(std::move(c));
        impl_->character.emplace_back("eos");
        impl_->numClasses = static_cast<int>(impl_->character.size());

        // Find sos/eos indices
        for (int i = 0; i < impl_->numClasses; ++i) {
            if (impl_->character[i] == "sos") {
                impl_->ignoredIdx.insert(i);
            } else if (impl_->character[i] == "eos") {
                impl_->ignoredIdx.insert(i);
                impl_->eosIdx = i;
            }
        }
        impl_->tdToken = {std::string("<td>"), std::string("<td"), std::string("<td></td>")};

        // Resolve outputs by last-dim: struct last-dim must equal numClasses;
        // bbox last-dim must be 4 or 8.
        int structIdx = -1, bboxIdx = -1;
        for (size_t i = 0; i < numOutputs; ++i) {
            if (lastDims[i] == impl_->numClasses) {
                structIdx = static_cast<int>(i);
            } else if (lastDims[i] == 8 || lastDims[i] == 4) {
                bboxIdx = static_cast<int>(i);
                impl_->bboxCoords = static_cast<int>(lastDims[i]);
            }
        }
        if (structIdx < 0 || bboxIdx < 0) {
            // Fallback: Python assumes output0=bbox, output1=struct.
            bboxIdx = 0;
            structIdx = 1;
            impl_->bboxCoords = static_cast<int>(lastDims[0]);
        }
        impl_->bboxOutputName = outNames[bboxIdx];
        impl_->structOutputName = outNames[structIdx];

    } catch (const std::exception& e) {
        LOG_ERROR("Failed to initialize TableWirelessRecognizer: {}", e.what());
        return false;
    }

    initialized_ = true;
    LOG_INFO("TableWirelessRecognizer initialized: {} (vocab={}, bbox_dim={})",
             config_.onnxModelPath, impl_->numClasses, impl_->bboxCoords);
    return true;
#endif
}

// -----------------------------------------------------------------------------

#ifdef HAS_ONNXRUNTIME

namespace {

// Python TablePreprocess: resize so max(h,w)->max_len, zero-pad top-left to
// max_len×max_len, normalize with BGR-order mean/std (Python applies mean/std
// to the raw np.ndarray which is BGR from OpenCV — no swap). Produces
// NCHW [1,3,max_len,max_len] and shape_list=[h,w,ratio,ratio,max_len,max_len].
struct PreResult {
    std::vector<float> chw;       // size 3 * maxLen * maxLen
    float ratio = 0.0f;
    int origH = 0;
    int origW = 0;
    int maxLen = 0;
};

PreResult preprocess(const cv::Mat& bgr, int maxLen) {
    PreResult pr;
    pr.maxLen = maxLen;
    pr.origH = bgr.rows;
    pr.origW = bgr.cols;
    if (bgr.empty() || bgr.cols <= 0 || bgr.rows <= 0) {
        return pr;
    }

    const int longSide = std::max(bgr.cols, bgr.rows);
    const float ratio = static_cast<float>(maxLen) / static_cast<float>(longSide);
    pr.ratio = ratio;
    const int resizeW = static_cast<int>(bgr.cols * ratio);  // Python: int(w*ratio) — truncates.
    const int resizeH = static_cast<int>(bgr.rows * ratio);

    cv::Mat working = bgr;
    if (working.channels() == 1) {
        cv::cvtColor(working, working, cv::COLOR_GRAY2BGR);
    } else if (working.channels() == 4) {
        cv::cvtColor(working, working, cv::COLOR_BGRA2BGR);
    }

    cv::Mat resized;
    cv::resize(working, resized, cv::Size(std::max(1, resizeW), std::max(1, resizeH)),
               0.0, 0.0, cv::INTER_LINEAR);

    const int planeSize = maxLen * maxLen;
    pr.chw.assign(static_cast<size_t>(3 * planeSize), 0.0f);

    // Normalize + top-left zero pad + HWC→CHW in one pass.
    const int copyH = std::min(resized.rows, maxLen);
    const int copyW = std::min(resized.cols, maxLen);
    for (int y = 0; y < copyH; ++y) {
        const uint8_t* row = resized.ptr<uint8_t>(y);
        for (int x = 0; x < copyW; ++x) {
            const uint8_t b = row[3 * x + 0];
            const uint8_t g = row[3 * x + 1];
            const uint8_t r = row[3 * x + 2];
            const float vb = (static_cast<float>(b) / 255.0f - kMean[0]) / kStd[0];
            const float vg = (static_cast<float>(g) / 255.0f - kMean[1]) / kStd[1];
            const float vr = (static_cast<float>(r) / 255.0f - kMean[2]) / kStd[2];
            const size_t idx = static_cast<size_t>(y) * maxLen + static_cast<size_t>(x);
            pr.chw[0 * planeSize + idx] = vb;
            pr.chw[1 * planeSize + idx] = vg;
            pr.chw[2 * planeSize + idx] = vr;
        }
    }
    return pr;
}

} // namespace

#endif // HAS_ONNXRUNTIME

// -----------------------------------------------------------------------------

namespace {

// Bulk emit HTML — mirrors table_matcher/main.py::TableMatch.get_pred_html.
// Returns the final joined string with <thead>/</thead>/<tbody>/</tbody>
// filtered out (per Python's behavior on this vocabulary).
std::string matchAndEmitHtml(
    const std::vector<std::string>& structureTokens,
    const std::vector<CellAabb>& cellBboxes,
    const std::vector<CellAabb>& ocrBoxes,
    const std::vector<std::string>& ocrTexts)
{
    // filter_ocr_result: drop OCR boxes whose max-y is above all cells' min-y.
    std::vector<CellAabb> dtBoxes;
    std::vector<std::string> recRes;
    dtBoxes.reserve(ocrBoxes.size());
    recRes.reserve(ocrTexts.size());
    if (!cellBboxes.empty() && !ocrBoxes.empty()) {
        float minCellY = cellBboxes.front().y1;
        for (const auto& c : cellBboxes) {
            if (c.y1 < minCellY) minCellY = c.y1;
            if (c.y2 < minCellY) minCellY = c.y2;
        }
        for (size_t i = 0; i < ocrBoxes.size(); ++i) {
            const float maxBoxY = std::max(ocrBoxes[i].y1, ocrBoxes[i].y2);
            if (maxBoxY < minCellY) continue;
            dtBoxes.push_back(ocrBoxes[i]);
            recRes.push_back(i < ocrTexts.size() ? ocrTexts[i] : std::string());
        }
    }

    // match_result: tdIdx -> vector<ocrIdx>.
    // Python uses min_iou = 0.1**8 = 1e-8 threshold: skip if (1 - iou) >= 1-min_iou.
    // i.e. iou <= 1e-8 (i.e. no practical overlap) -> skip.
    constexpr float kMinIou = 1e-8f;

    std::vector<std::vector<int>> matched(cellBboxes.size());
    for (size_t i = 0; i < dtBoxes.size(); ++i) {
        const CellAabb& gt = dtBoxes[i];
        int bestIdx = -1;
        float bestIouComp = 2.0f;  // (1 - IoU), primary sort key asc
        float bestDist = 0.0f;     // secondary sort key asc
        for (size_t j = 0; j < cellBboxes.size(); ++j) {
            const CellAabb& pred = cellBboxes[j];
            const float iou = axisIou(gt, pred);
            const float dist = l1Distance(gt, pred);
            const float iouComp = 1.0f - iou;
            if (iouComp < bestIouComp
                || (iouComp == bestIouComp && dist < bestDist)) {
                bestIouComp = iouComp;
                bestDist = dist;
                bestIdx = static_cast<int>(j);
            }
        }
        if (bestIdx < 0) continue;
        // Skip when (1-IoU) >= 1 - min_iou   ⇔   IoU <= min_iou
        if (bestIouComp >= (1.0f - kMinIou)) continue;
        matched[static_cast<size_t>(bestIdx)].push_back(static_cast<int>(i));
    }

    // get_pred_html
    std::string endHtml;
    endHtml.reserve(256);
    int tdIndex = 0;
    auto appendBytes = [&](const std::string& s) { endHtml.append(s); };
    static const std::array<std::string, 4> kFilter =
        {"<thead>", "</thead>", "<tbody>", "</tbody>"};
    auto isFiltered = [&](const std::string& tok) {
        for (const auto& f : kFilter) if (tok == f) return true;
        return false;
    };

    for (const auto& tag : structureTokens) {
        if (tag.find("</td>") == std::string::npos) {
            if (!isFiltered(tag)) appendBytes(tag);
            continue;
        }

        if (tag == "<td></td>") {
            // Python: end_html.extend("<td>") — extending with a string puts
            // each char separately, which joined becomes the literal "<td>".
            appendBytes("<td>");
        }

        if (tdIndex >= 0 && static_cast<size_t>(tdIndex) < matched.size()
            && !matched[static_cast<size_t>(tdIndex)].empty()) {
            auto& ocrIdxList = matched[static_cast<size_t>(tdIndex)];

            // Mirror Python's b_with logic: if the first matched ocr text
            // already contains "<b>" and there's >1 text, emit a wrapping <b>.
            bool bWith = false;
            if (ocrIdxList.size() > 1
                && recRes[static_cast<size_t>(ocrIdxList.front())].find("<b>") != std::string::npos) {
                bWith = true;
                appendBytes("<b>");
            }

            for (size_t i = 0; i < ocrIdxList.size(); ++i) {
                std::string content = recRes[static_cast<size_t>(ocrIdxList[i])];
                if (ocrIdxList.size() > 1) {
                    if (content.empty()) continue;
                    if (content.front() == ' ') content.erase(0, 1);
                    if (content.rfind("<b>", 0) == 0) content.erase(0, 3);
                    if (content.size() >= 4
                        && content.compare(content.size() - 4, 4, "</b>") == 0) {
                        content.erase(content.size() - 4, 4);
                    }
                    if (content.empty()) continue;
                    if (i != ocrIdxList.size() - 1 && content.back() != ' ') {
                        content.push_back(' ');
                    }
                }
                appendBytes(content);
            }

            if (bWith) appendBytes("</b>");
        }

        if (tag == "<td></td>") {
            appendBytes("</td>");
        } else {
            appendBytes(tag);
        }
        tdIndex += 1;
    }

    return endHtml;
}

} // namespace

// -----------------------------------------------------------------------------

TableResult TableWirelessRecognizer::recognize(
    const cv::Mat& bgrImage,
    const std::vector<WirelessOcrBox>& ocrBoxes)
{
    TableResult result;
    result.type = TableType::WIRELESS;
    result.supported = false;

#ifndef HAS_ONNXRUNTIME
    LOG_WARN("TableWirelessRecognizer::recognize called but HAS_ONNXRUNTIME disabled.");
    return result;
#else
    if (!initialized_ || !impl_->session || bgrImage.empty()) {
        return result;
    }

    const auto tStart = std::chrono::steady_clock::now();
    try {
        // 1) Preprocess
        PreResult pre = preprocess(bgrImage, config_.inputSize);
        if (pre.chw.empty()) return result;
        const int maxLen = pre.maxLen;

        const std::array<int64_t, 4> inputShape{
            1, 3, static_cast<int64_t>(maxLen), static_cast<int64_t>(maxLen)};
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            impl_->memInfo, pre.chw.data(), pre.chw.size(),
            inputShape.data(), inputShape.size());

        const char* inputNames[] = {impl_->inputName.c_str()};
        const char* outputNames[] = {
            impl_->bboxOutputName.c_str(),
            impl_->structOutputName.c_str()
        };
        auto outputs = impl_->session->Run(
            Ort::RunOptions{nullptr},
            inputNames, &inputTensor, 1,
            outputNames, 2);
        if (outputs.size() < 2) {
            throw std::runtime_error("SLANet+ produced fewer than 2 outputs");
        }
        auto bboxInfo = outputs[0].GetTensorTypeAndShapeInfo();
        auto structInfo = outputs[1].GetTensorTypeAndShapeInfo();
        auto bboxShape = bboxInfo.GetShape();
        auto structShape = structInfo.GetShape();
        if (bboxShape.size() < 3 || structShape.size() < 3) {
            throw std::runtime_error("SLANet+ output rank < 3");
        }
        // Expect batch=1.
        const int T = static_cast<int>(structShape[1]);
        const int C = static_cast<int>(structShape[2]);
        const int B = static_cast<int>(bboxShape[2]);
        if (C != impl_->numClasses) {
            throw std::runtime_error("SLANet+ structure class dim != vocab size");
        }
        if (B != impl_->bboxCoords) {
            throw std::runtime_error("SLANet+ bbox coord dim unexpected");
        }
        const float* structData = outputs[1].GetTensorData<float>();
        const float* bboxData = outputs[0].GetTensorData<float>();

        // 2) Decode: argmax per step; stop at first eos (for t>0); skip sos.
        //    For tokens in td_token, collect bbox scaled by shape_list[0,1].
        std::vector<std::string> structureList;
        std::vector<CellAabb> cellBboxes;  // after _bbox_decode (on orig h/w)

        const int eosIdx = impl_->eosIdx;
        const auto& ignored = impl_->ignoredIdx;
        const auto& vocab = impl_->character;

        for (int t = 0; t < T; ++t) {
            const float* logits = structData + static_cast<size_t>(t) * C;
            int argmax = 0;
            float bestVal = logits[0];
            for (int c = 1; c < C; ++c) {
                if (logits[c] > bestVal) { bestVal = logits[c]; argmax = c; }
            }
            if (t > 0 && argmax == eosIdx) break;
            if (ignored.count(argmax)) continue;

            const std::string& tok = vocab[static_cast<size_t>(argmax)];
            if (impl_->tdToken.count(tok)) {
                const float* rawBbox = bboxData + static_cast<size_t>(t) * B;
                // _bbox_decode: bbox[0::2] *= w; bbox[1::2] *= h;
                std::array<float, 8> scaled{};
                for (int k = 0; k < B; ++k) {
                    if ((k & 1) == 0) scaled[k] = rawBbox[k] * static_cast<float>(pre.origW);
                    else              scaled[k] = rawBbox[k] * static_cast<float>(pre.origH);
                }
                // Convert to AABB (min/max over pairs); works for B=4 or 8.
                CellAabb aabb;
                aabb.x1 = aabb.x2 = scaled[0];
                aabb.y1 = aabb.y2 = scaled[1];
                for (int k = 0; k < B; k += 2) {
                    aabb.x1 = std::min(aabb.x1, scaled[k]);
                    aabb.x2 = std::max(aabb.x2, scaled[k]);
                    aabb.y1 = std::min(aabb.y1, scaled[k + 1]);
                    aabb.y2 = std::max(aabb.y2, scaled[k + 1]);
                }
                cellBboxes.push_back(aabb);
            }
            structureList.push_back(tok);
        }

        // 3) rescale_cell_bboxes (SLANet+ only)
        //    ratio = min(maxLen/h, maxLen/w); w_ratio = maxLen/(w*ratio); h_ratio = maxLen/(h*ratio)
        {
            const float h = static_cast<float>(pre.origH);
            const float w = static_cast<float>(pre.origW);
            const float ratio = std::min(static_cast<float>(maxLen) / h,
                                         static_cast<float>(maxLen) / w);
            const float wRatio = static_cast<float>(maxLen) / (w * ratio);
            const float hRatio = static_cast<float>(maxLen) / (h * ratio);
            for (auto& c : cellBboxes) {
                c.x1 *= wRatio; c.x2 *= wRatio;
                c.y1 *= hRatio; c.y2 *= hRatio;
            }
        }

        // 4) filter_blank_bbox: drop cells that are all-zero (placeholder).
        cellBboxes.erase(
            std::remove_if(cellBboxes.begin(), cellBboxes.end(),
                [](const CellAabb& c) {
                    return c.x1 == 0.0f && c.y1 == 0.0f
                        && c.x2 == 0.0f && c.y2 == 0.0f;
                }),
            cellBboxes.end());

        // 5) Wrap structure — Python get_struct_str
        std::vector<std::string> structureTokens;
        structureTokens.reserve(structureList.size() + 6);
        structureTokens.emplace_back("<html>");
        structureTokens.emplace_back("<body>");
        structureTokens.emplace_back("<table>");
        for (auto& s : structureList) structureTokens.emplace_back(std::move(s));
        structureTokens.emplace_back("</table>");
        structureTokens.emplace_back("</body>");
        structureTokens.emplace_back("</html>");

        // 6) Build OCR box arrays in AABB form (crop-local coordinates).
        std::vector<CellAabb> ocrAabb;
        std::vector<std::string> ocrTexts;
        ocrAabb.reserve(ocrBoxes.size());
        ocrTexts.reserve(ocrBoxes.size());
        for (const auto& r : ocrBoxes) {
            CellAabb a;
            a.x1 = static_cast<float>(r.aabb.x);
            a.y1 = static_cast<float>(r.aabb.y);
            a.x2 = static_cast<float>(r.aabb.x + r.aabb.width);
            a.y2 = static_cast<float>(r.aabb.y + r.aabb.height);
            ocrAabb.push_back(a);
            ocrTexts.push_back(r.text);
        }

        // 7) Match + emit HTML.
        std::string html = matchAndEmitHtml(structureTokens, cellBboxes, ocrAabb, ocrTexts);

        result.html = std::move(html);
        result.supported = !result.html.empty();
        result.cells.clear();
        const auto tEnd = std::chrono::steady_clock::now();
        result.inferenceTimeMs =
            std::chrono::duration<double, std::milli>(tEnd - tStart).count();

        const char* routeTrace = std::getenv("RAPIDDOC_TABLE_TRACE");
        if (routeTrace && routeTrace[0] != '\0' && routeTrace[0] != '0') {
            LOG_INFO(
                "TABLE_ROUTE_TRACE wireless crop_size={}x{} cells={} "
                "html_len={} infer_ms={:.2f}",
                bgrImage.cols, bgrImage.rows,
                static_cast<int>(cellBboxes.size()),
                static_cast<int>(result.html.size()),
                result.inferenceTimeMs);
        }
        return result;
    } catch (const std::exception& e) {
        LOG_WARN("TableWirelessRecognizer::recognize failed: {}", e.what());
        result.supported = false;
        result.html.clear();
        return result;
    }
#endif
}

} // namespace rapid_doc
