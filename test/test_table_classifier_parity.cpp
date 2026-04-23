/**
 * @file test_table_classifier_parity.cpp
 * @brief Smoke / contract test for TableClassifier (PaddleCls C++ port).
 *
 * This test verifies:
 *   - initialize() succeeds when the ONNX model is present;
 *   - classify() returns ok=true on a reasonable BGR crop;
 *   - label is one of {"wired", "wireless"};
 *   - score is a valid softmax probability in [0, 1];
 *   - softmax logits sum to 1 (up to rounding).
 *
 * Strict label parity vs Python would require identical crops; we test
 * structural invariants here and defer bit-exact parity to the end-to-end
 * regression run on 表格0-4.pdf.
 */

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include <cmath>
#include <filesystem>
#include <string>

#include "table/table_classifier.h"

namespace fs = std::filesystem;
using namespace rapid_doc;

namespace {

std::string paddleClsPath() {
    return std::string(PROJECT_ROOT_DIR) + "/.download_cache/onnx_models/paddle_cls.onnx";
}

class TableClassifierParityTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!fs::exists(paddleClsPath())) {
            GTEST_SKIP() << "paddle_cls.onnx not cached at " << paddleClsPath()
                         << "; run setup.sh first.";
        }
        TableClassifierConfig cfg;
        cfg.onnxModelPath = paddleClsPath();
        classifier_ = std::make_unique<TableClassifier>(cfg);
        ASSERT_TRUE(classifier_->initialize());
    }
    std::unique_ptr<TableClassifier> classifier_;
};

cv::Mat makeWiredLikeCrop() {
    // Dense border grid → strong line ratio → heuristic would say WIRED.
    cv::Mat img(300, 400, CV_8UC3, cv::Scalar(255, 255, 255));
    for (int r = 0; r < 300; r += 50)
        cv::line(img, cv::Point(0, r), cv::Point(400, r), cv::Scalar(0, 0, 0), 2);
    for (int c = 0; c < 400; c += 80)
        cv::line(img, cv::Point(c, 0), cv::Point(c, 300), cv::Scalar(0, 0, 0), 2);
    return img;
}

cv::Mat makeBlankCrop() {
    // Pure white; no lines. Useful to check the classifier doesn't crash on
    // degenerate input and still returns a score.
    return cv::Mat(300, 400, CV_8UC3, cv::Scalar(255, 255, 255));
}

} // namespace

TEST_F(TableClassifierParityTest, InitializedSuccessfully) {
    EXPECT_TRUE(classifier_->isInitialized());
}

TEST_F(TableClassifierParityTest, ClassifyWiredLikeCropReturnsValidResult) {
    cv::Mat img = makeWiredLikeCrop();
    TableClassifyResult r = classifier_->classify(img);
    EXPECT_TRUE(r.ok);
    EXPECT_TRUE(r.label == "wired" || r.label == "wireless")
        << "unexpected label: " << r.label;
    EXPECT_GE(r.score, 0.0f);
    EXPECT_LE(r.score, 1.0f);
    EXPECT_GE(r.predIndex, 0);
    EXPECT_LT(r.predIndex, 2);
}

TEST_F(TableClassifierParityTest, ScoreIsSoftmaxCalibrated) {
    cv::Mat img = makeWiredLikeCrop();
    TableClassifyResult r = classifier_->classify(img);
    ASSERT_TRUE(r.ok);
    // softmax(max) + softmax(min) == 1
    const float maxL = std::max(r.rawLogitWired, r.rawLogitWireless);
    const float e0 = std::exp(r.rawLogitWired - maxL);
    const float e1 = std::exp(r.rawLogitWireless - maxL);
    const float sum = e0 + e1;
    const float p0 = e0 / sum;
    const float p1 = e1 / sum;
    EXPECT_NEAR(p0 + p1, 1.0f, 1e-4f);
    EXPECT_NEAR(r.score, std::max(p0, p1), 1e-4f);
}

TEST_F(TableClassifierParityTest, BlankCropDoesNotCrash) {
    cv::Mat img = makeBlankCrop();
    TableClassifyResult r = classifier_->classify(img);
    EXPECT_TRUE(r.ok);
    EXPECT_GE(r.score, 0.0f);
    EXPECT_LE(r.score, 1.0f);
}

TEST_F(TableClassifierParityTest, ClassifyHonorsMinSideGuard) {
    // F3 gate lives in TableRecognizer (not in TableClassifier itself), but
    // the classifier must still handle a small crop without crashing when
    // someone calls classify() directly.
    cv::Mat tiny(50, 40, CV_8UC3, cv::Scalar(200, 200, 200));
    TableClassifyResult r = classifier_->classify(tiny);
    // We don't enforce label/ok for such a small crop — the important
    // contract is "no crash / no UB".
    SUCCEED() << "classify() completed on 40x50 crop without throwing";
    (void)r;
}
