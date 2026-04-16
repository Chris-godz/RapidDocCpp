/**
 * @file formula_recognizer.cpp
 * @brief Formula recognition implementation (PP-FormulaNet+ ONNX).
 */

#include "formula/formula_recognizer.h"

#include "common/logger.h"

#include <nlohmann/json.hpp>

#ifdef HAS_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <numeric>
#include <regex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace rapid_doc {

namespace {

constexpr float kNormMean = 0.7931f;
constexpr float kNormStd = 0.1738f;

std::string replaceAllCopy(std::string value, const std::string& from, const std::string& to) {
    if (from.empty()) {
        return value;
    }
    size_t pos = 0;
    while ((pos = value.find(from, pos)) != std::string::npos) {
        value.replace(pos, from.size(), to);
        pos += to.size();
    }
    return value;
}

std::string trimCopy(const std::string& input) {
    const auto begin = input.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const auto end = input.find_last_not_of(" \t\r\n");
    return input.substr(begin, end - begin + 1);
}

cv::Mat cropMargin(const cv::Mat& image) {
    if (image.empty()) {
        return {};
    }

    cv::Mat gray;
    if (image.channels() == 1) {
        gray = image;
    } else {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }

    double minVal = 0.0;
    double maxVal = 0.0;
    cv::minMaxLoc(gray, &minVal, &maxVal);
    if (maxVal <= minVal) {
        return image.clone();
    }

    cv::Mat grayFloat;
    gray.convertTo(grayFloat, CV_32F);
    grayFloat = (grayFloat - static_cast<float>(minVal)) *
                (255.0f / static_cast<float>(maxVal - minVal));

    cv::Mat mask;
    cv::threshold(grayFloat, mask, 200.0, 255.0, cv::THRESH_BINARY_INV);
    mask.convertTo(mask, CV_8U);

    std::vector<cv::Point> nonZero;
    cv::findNonZero(mask, nonZero);
    if (nonZero.empty()) {
        return image.clone();
    }
    const cv::Rect rect = cv::boundingRect(nonZero);
    return image(rect).clone();
}

cv::Mat resizeAndPadToSquare(const cv::Mat& image, int size) {
    if (image.empty() || size <= 0) {
        return {};
    }

    const int srcH = image.rows;
    const int srcW = image.cols;
    if (srcH <= 0 || srcW <= 0) {
        return {};
    }

    const int shortSide = std::min(srcH, srcW);
    const float firstScale = static_cast<float>(size) / static_cast<float>(shortSide);
    int resizedH = std::max(1, static_cast<int>(std::lround(srcH * firstScale)));
    int resizedW = std::max(1, static_cast<int>(std::lround(srcW * firstScale)));

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(resizedW, resizedH), 0, 0, cv::INTER_LINEAR);

    float thumbScale = 1.0f;
    if (resizedW > size || resizedH > size) {
        thumbScale = std::min(
            static_cast<float>(size) / static_cast<float>(resizedW),
            static_cast<float>(size) / static_cast<float>(resizedH));
    }
    if (thumbScale < 1.0f) {
        resizedW = std::max(1, static_cast<int>(std::floor(resizedW * thumbScale)));
        resizedH = std::max(1, static_cast<int>(std::floor(resizedH * thumbScale)));
        cv::resize(resized, resized, cv::Size(resizedW, resizedH), 0, 0, cv::INTER_LINEAR);
    } else {
        resizedW = resized.cols;
        resizedH = resized.rows;
    }

    cv::Mat padded(size, size, CV_8UC3, cv::Scalar(0, 0, 0));
    const int padX = (size - resizedW) / 2;
    const int padY = (size - resizedH) / 2;
    resized.copyTo(padded(cv::Rect(padX, padY, resizedW, resizedH)));
    return padded;
}

void decodeTokenizerMetadata(
    const json& characterJson,
    std::vector<std::string>& idToToken,
    int64_t& eosTokenId)
{
    if (!characterJson.contains("fast_tokenizer_file")) {
        throw std::runtime_error("Formula metadata missing fast_tokenizer_file");
    }
    const auto& tokenizer = characterJson.at("fast_tokenizer_file");
    if (!tokenizer.contains("model") || !tokenizer.at("model").contains("vocab")) {
        throw std::runtime_error("Formula metadata missing tokenizer vocab");
    }

    const auto& vocab = tokenizer.at("model").at("vocab");
    int64_t maxId = 0;
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        maxId = std::max(maxId, static_cast<int64_t>(it.value().get<int64_t>()));
    }

    if (tokenizer.contains("added_tokens") && tokenizer.at("added_tokens").is_array()) {
        for (const auto& token : tokenizer.at("added_tokens")) {
            if (!token.is_object() || !token.contains("id")) {
                continue;
            }
            maxId = std::max(maxId, token.at("id").get<int64_t>());
        }
    }

    idToToken.assign(static_cast<size_t>(maxId + 1), "");
    for (auto it = vocab.begin(); it != vocab.end(); ++it) {
        const int64_t id = it.value().get<int64_t>();
        if (id >= 0 && static_cast<size_t>(id) < idToToken.size()) {
            idToToken[static_cast<size_t>(id)] = it.key();
        }
    }

    if (tokenizer.contains("added_tokens") && tokenizer.at("added_tokens").is_array()) {
        for (const auto& token : tokenizer.at("added_tokens")) {
            if (!token.is_object() || !token.contains("id") || !token.contains("content")) {
                continue;
            }
            const int64_t id = token.at("id").get<int64_t>();
            if (id >= 0 && static_cast<size_t>(id) < idToToken.size() &&
                idToToken[static_cast<size_t>(id)].empty()) {
                idToToken[static_cast<size_t>(id)] = token.at("content").get<std::string>();
            }
        }
    }

    eosTokenId = 2;
    for (size_t i = 0; i < idToToken.size(); ++i) {
        if (idToToken[i] == "</s>") {
            eosTokenId = static_cast<int64_t>(i);
            break;
        }
    }
}

} // namespace

struct FormulaRecognizer::Impl {
#ifdef HAS_ONNXRUNTIME
    Ort::Env ortEnv{ORT_LOGGING_LEVEL_WARNING, "formula_onnx"};
    std::unique_ptr<Ort::Session> session;
    Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
#endif
    std::string inputName;
    std::string outputName;
    std::vector<std::string> idToToken;
    int64_t eosTokenId = 2;
};

FormulaRecognizer::FormulaRecognizer(const FormulaRecognizerConfig& config)
    : config_(config)
    , impl_(std::make_unique<Impl>())
{
}

FormulaRecognizer::~FormulaRecognizer() = default;

bool FormulaRecognizer::initialize() {
#ifndef HAS_ONNXRUNTIME
    LOG_ERROR("Formula recognizer requires ONNX Runtime, but HAS_ONNXRUNTIME is not enabled.");
    return false;
#else
    if (!fs::exists(config_.onnxModelPath)) {
        LOG_ERROR("Formula ONNX model not found: {}", config_.onnxModelPath);
        return false;
    }

    try {
        Ort::SessionOptions opts;
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        if (config_.sequentialExecution) {
            opts.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
        }
        if (!config_.enableCpuMemArena) {
            opts.DisableCpuMemArena();
        }
        if (config_.intraOpThreads > 0) {
            opts.SetIntraOpNumThreads(config_.intraOpThreads);
        }
        if (config_.interOpThreads > 0) {
            opts.SetInterOpNumThreads(config_.interOpThreads);
        }
        impl_->session =
            std::make_unique<Ort::Session>(impl_->ortEnv, config_.onnxModelPath.c_str(), opts);

        Ort::AllocatorWithDefaultOptions allocator;
        {
            auto inputName = impl_->session->GetInputNameAllocated(0, allocator);
            impl_->inputName = inputName.get();
        }
        {
            auto outputName = impl_->session->GetOutputNameAllocated(0, allocator);
            impl_->outputName = outputName.get();
        }

        const auto metadata = impl_->session->GetModelMetadata();
        auto characterMeta = metadata.LookupCustomMetadataMapAllocated(
            "character", allocator);
        if (!characterMeta || characterMeta.get() == nullptr ||
            std::string(characterMeta.get()).empty()) {
            throw std::runtime_error("Formula model metadata key 'character' not found");
        }
        const json characterJson = json::parse(characterMeta.get());
        decodeTokenizerMetadata(characterJson, impl_->idToToken, impl_->eosTokenId);
    } catch (const std::exception& e) {
        LOG_ERROR("Failed to initialize formula recognizer: {}", e.what());
        return false;
    }

    initialized_ = true;
    LOG_INFO("Formula recognizer initialized: {}", config_.onnxModelPath);
    return true;
#endif
}

std::vector<float> FormulaRecognizer::preprocessSingle(
    const cv::Mat& image,
    int& outH,
    int& outW) const
{
    outH = 0;
    outW = 0;
    if (image.empty()) {
        return {};
    }

    cv::Mat working = image;
    if (working.channels() == 1) {
        cv::cvtColor(working, working, cv::COLOR_GRAY2BGR);
    } else if (working.channels() == 4) {
        cv::cvtColor(working, working, cv::COLOR_BGRA2BGR);
    }

    cv::Mat cropped = cropMargin(working);
    if (cropped.empty()) {
        return {};
    }

    cv::Mat square = resizeAndPadToSquare(cropped, config_.inputSize);
    if (square.empty()) {
        return {};
    }

    cv::Mat norm;
    square.convertTo(norm, CV_32FC3, 1.0 / 255.0);
    norm = (norm - cv::Scalar(kNormMean, kNormMean, kNormMean)) / kNormStd;

    cv::Mat gray;
    cv::cvtColor(norm, gray, cv::COLOR_BGR2GRAY);

    outH = ((gray.rows + 15) / 16) * 16;
    outW = ((gray.cols + 15) / 16) * 16;
    cv::Mat padded(outH, outW, CV_32F, cv::Scalar(1.0f));
    gray.copyTo(padded(cv::Rect(0, 0, gray.cols, gray.rows)));

    std::vector<float> data(static_cast<size_t>(outH * outW));
    std::memcpy(data.data(), padded.ptr<float>(0), data.size() * sizeof(float));
    return data;
}

std::string FormulaRecognizer::normalizeLatex(const std::string& text) {
    std::string out = text;
    out = replaceAllCopy(out, "Ġ", " ");
    out = replaceAllCopy(out, "Ċ", "\n");
    out.erase(std::remove(out.begin(), out.end(), '"'), out.end());

    static const std::regex nonLetterNonLetter(R"(([^A-Za-z0-9_\\\s])\s+([^A-Za-z0-9_\\\s]))");
    static const std::regex nonLetterLetter(R"(([^A-Za-z0-9_\\\s])\s+([A-Za-z]))");
    static const std::regex letterNonLetter(R"(([A-Za-z])\s+([^A-Za-z0-9_\\\s]))");

    while (true) {
        const std::string prev = out;
        out = std::regex_replace(out, nonLetterNonLetter, "$1$2");
        out = std::regex_replace(out, nonLetterLetter, "$1$2");
        out = std::regex_replace(out, letterNonLetter, "$1$2");
        if (out == prev) {
            break;
        }
    }

    return trimCopy(out);
}

std::string FormulaRecognizer::decodeTokensForTest(
    const std::vector<int64_t>& tokenIds,
    const std::vector<std::string>& idToToken)
{
    static const std::unordered_set<std::string> kSpecial{
        "<s>", "</s>", "<pad>", "<unk>",
    };

    std::string raw;
    for (const int64_t id : tokenIds) {
        if (id < 0 || static_cast<size_t>(id) >= idToToken.size()) {
            continue;
        }
        const std::string& token = idToToken[static_cast<size_t>(id)];
        if (token.empty() || kSpecial.count(token) > 0) {
            continue;
        }
        raw += token;
    }
    return normalizeLatex(raw);
}

std::string FormulaRecognizer::decodeTokensRaw(const std::vector<int64_t>& tokenIds) const {
    return decodeTokensRaw(tokenIds.data(), static_cast<int64_t>(tokenIds.size()));
}

std::string FormulaRecognizer::decodeTokensRaw(const int64_t* tokenIds, int64_t tokenCount) const {
    static const std::unordered_set<std::string> kSpecial{
        "<s>", "</s>", "<pad>", "<unk>",
    };

    std::string raw;
    for (int64_t i = 0; i < tokenCount; ++i) {
        const int64_t id = tokenIds[i];
        if (id == impl_->eosTokenId) {
            break;
        }
        if (id < 0 || static_cast<size_t>(id) >= impl_->idToToken.size()) {
            continue;
        }
        const std::string& token = impl_->idToToken[static_cast<size_t>(id)];
        if (token.empty() || kSpecial.count(token) > 0) {
            continue;
        }
        raw += token;
    }
    return raw;
}

std::string FormulaRecognizer::decodeTokens(const std::vector<int64_t>& tokenIds) const {
    return normalizeLatex(decodeTokensRaw(tokenIds));
}

std::vector<std::string> FormulaRecognizer::recognizeBatch(
    const std::vector<cv::Mat>& formulaCrops,
    BatchTiming* timing) const
{
    std::vector<std::string> outputs(formulaCrops.size());
    if (timing != nullptr) {
        *timing = BatchTiming{};
    }
    if (!initialized_ || formulaCrops.empty()) {
        return outputs;
    }

#ifndef HAS_ONNXRUNTIME
    (void)formulaCrops;
    (void)timing;
    return outputs;
#else
    struct PackedCrop {
        size_t index = 0;
        int h = 0;
        int w = 0;
        std::vector<float> tensor;
    };

    const auto totalStart = std::chrono::steady_clock::now();
    double preprocessMs = 0.0;
    double inferMs = 0.0;
    double decodeMs = 0.0;
    double normalizeMs = 0.0;
    int batchCount = 0;

    std::vector<PackedCrop> packed;
    packed.reserve(formulaCrops.size());
    for (size_t i = 0; i < formulaCrops.size(); ++i) {
        PackedCrop item;
        item.index = i;
        const auto preprocessStart = std::chrono::steady_clock::now();
        item.tensor = preprocessSingle(formulaCrops[i], item.h, item.w);
        const auto preprocessEnd = std::chrono::steady_clock::now();
        preprocessMs +=
            std::chrono::duration<double, std::milli>(preprocessEnd - preprocessStart).count();
        if (!item.tensor.empty()) {
            packed.push_back(std::move(item));
        }
    }
    if (packed.empty()) {
        if (timing != nullptr) {
            timing->totalMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - totalStart).count();
            timing->preprocessMs = preprocessMs;
            timing->cropCount = 0;
            timing->batchCount = 0;
        }
        return outputs;
    }

    auto runPackedBatch = [&](
        const std::vector<PackedCrop>& packedRef,
        const std::vector<size_t>& groupIndices,
        size_t begin,
        size_t end,
        std::vector<float>& inputScratch)
    {
        const int h = packedRef[groupIndices[begin]].h;
        const int w = packedRef[groupIndices[begin]].w;
        const size_t batchSizeU = end - begin;
        const int64_t batchSize = static_cast<int64_t>(batchSizeU);
        const int64_t sampleSize = static_cast<int64_t>(h) * static_cast<int64_t>(w);
        const size_t sampleSizeU = static_cast<size_t>(sampleSize);
        const size_t requiredSize = static_cast<size_t>(batchSize) * sampleSizeU;
        if (inputScratch.size() < requiredSize) {
            inputScratch.resize(requiredSize);
        }
        for (size_t i = 0; i < batchSizeU; ++i) {
            const size_t packedIdx = groupIndices[begin + i];
            std::memcpy(
                inputScratch.data() + (i * sampleSizeU),
                packedRef[packedIdx].tensor.data(),
                sampleSizeU * sizeof(float));
        }

        const std::array<int64_t, 4> inputShape{batchSize, 1, h, w};
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            impl_->memInfo,
            inputScratch.data(),
            requiredSize,
            inputShape.data(),
            inputShape.size());

        const char* inputNames[] = {impl_->inputName.c_str()};
        const char* outputNames[] = {impl_->outputName.c_str()};
        const auto inferStart = std::chrono::steady_clock::now();
        auto ortOutputs = impl_->session->Run(
            Ort::RunOptions{nullptr},
            inputNames,
            &inputTensor,
            1,
            outputNames,
            1);
        const auto inferEnd = std::chrono::steady_clock::now();
        inferMs += std::chrono::duration<double, std::milli>(inferEnd - inferStart).count();
        ++batchCount;

        if (ortOutputs.empty() || !ortOutputs.front().IsTensor()) {
            throw std::runtime_error("Formula ONNX output tensor missing");
        }

        const auto outInfo = ortOutputs.front().GetTensorTypeAndShapeInfo();
        const auto outShape = outInfo.GetShape();
        if (outShape.size() < 2) {
            throw std::runtime_error("Formula ONNX output rank is invalid");
        }
        const int64_t seqLen = outShape[1];
        if (seqLen <= 0) {
            return;
        }

        const int64_t* tokenPtr = ortOutputs.front().GetTensorData<int64_t>();
        for (size_t i = 0; i < batchSizeU; ++i) {
            const int64_t* row = tokenPtr + (i * seqLen);
            const auto decodeStart = std::chrono::steady_clock::now();
            const std::string raw = decodeTokensRaw(row, seqLen);
            const auto decodeEnd = std::chrono::steady_clock::now();
            decodeMs += std::chrono::duration<double, std::milli>(decodeEnd - decodeStart).count();

            const auto normalizeStart = std::chrono::steady_clock::now();
            const size_t packedIdx = groupIndices[begin + i];
            outputs[packedRef[packedIdx].index] = normalizeLatex(raw);
            const auto normalizeEnd = std::chrono::steady_clock::now();
            normalizeMs += std::chrono::duration<double, std::milli>(
                normalizeEnd - normalizeStart).count();
        }
    };

    // Group by tensor shape to keep dynamic-shape batch inference safe.
    std::unordered_map<std::string, std::vector<size_t>> groups;
    groups.reserve(packed.size());
    for (size_t idx = 0; idx < packed.size(); ++idx) {
        const auto& item = packed[idx];
        const std::string key = std::to_string(item.h) + "x" + std::to_string(item.w);
        groups[key].push_back(idx);
    }

    for (const auto& kv : groups) {
        const auto& groupIndices = kv.second;
        if (groupIndices.empty()) {
            continue;
        }
        const size_t maxBatchSize =
            (config_.maxBatchSize > 0)
                ? static_cast<size_t>(config_.maxBatchSize)
                : groupIndices.size();
        const size_t sampleSizeU = static_cast<size_t>(
            packed[groupIndices.front()].h * packed[groupIndices.front()].w);
        std::vector<float> inputScratch(maxBatchSize * sampleSizeU);
        for (size_t begin = 0; begin < groupIndices.size(); begin += maxBatchSize) {
            const size_t end = std::min(groupIndices.size(), begin + maxBatchSize);
            runPackedBatch(packed, groupIndices, begin, end, inputScratch);
        }
    }

    if (timing != nullptr) {
        const auto totalEnd = std::chrono::steady_clock::now();
        timing->preprocessMs = preprocessMs;
        timing->inferMs = inferMs;
        timing->decodeMs = decodeMs;
        timing->normalizeMs = normalizeMs;
        timing->cropCount = static_cast<int>(packed.size());
        timing->batchCount = batchCount;
        timing->totalMs =
            std::chrono::duration<double, std::milli>(totalEnd - totalStart).count();
    }
    return outputs;
#endif
}

} // namespace rapid_doc
