#pragma once

/**
 * @file formula_recognizer.h
 * @brief Formula recognition using ONNX Runtime (PP-FormulaNet+).
 */

#include <opencv2/opencv.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace rapid_doc {

struct FormulaRecognizerConfig {
    std::string onnxModelPath;
    int inputSize = 384;
    int intraOpThreads = -1;
    int interOpThreads = -1;
    bool sequentialExecution = false;
    bool enableCpuMemArena = false;
    int maxBatchSize = 1;
};

class FormulaRecognizer {
public:
    struct BatchTiming {
        double preprocessMs = 0.0;
        double inferMs = 0.0;
        double decodeMs = 0.0;
        double normalizeMs = 0.0;
        double totalMs = 0.0;
        int cropCount = 0;
        int batchCount = 0;
    };

    explicit FormulaRecognizer(const FormulaRecognizerConfig& config);
    ~FormulaRecognizer();

    bool initialize();
    bool isInitialized() const { return initialized_; }

    std::vector<std::string> recognizeBatch(
        const std::vector<cv::Mat>& formulaCrops,
        BatchTiming* timing = nullptr) const;

    // Exposed for unit tests.
    static std::string decodeTokensForTest(
        const std::vector<int64_t>& tokenIds,
        const std::vector<std::string>& idToToken);
    static std::string normalizeLatex(const std::string& text);

private:
    std::vector<float> preprocessSingle(const cv::Mat& image, int& outH, int& outW) const;
    std::string decodeTokensRaw(const std::vector<int64_t>& tokenIds) const;
    std::string decodeTokensRaw(const int64_t* tokenIds, int64_t tokenCount) const;
    std::string decodeTokens(const std::vector<int64_t>& tokenIds) const;

    struct Impl;
    FormulaRecognizerConfig config_;
    std::unique_ptr<Impl> impl_;
    bool initialized_ = false;
};

} // namespace rapid_doc
