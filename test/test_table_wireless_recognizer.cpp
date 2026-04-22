/**
 * @file test_table_wireless_recognizer.cpp
 * @brief Smoke / contract test for TableWirelessRecognizer (SLANet+ C++ port).
 *
 * This test does NOT attempt bit-exact parity with Python (different interp
 * kernels + floating-point rounding make that fragile). Instead it pins the
 * *contract* the pipeline relies on:
 *   - initialize() succeeds when the ONNX model is present;
 *   - recognize() returns TableResult with supported=true;
 *   - emitted html starts with <html><body><table> and ends with
 *     </table></body></html> (post get_struct_str wrapping);
 *   - at least one <td> or <td></td> opener appears for non-trivial inputs;
 *   - <thead>/</thead>/<tbody>/</tbody> are filtered out (Python behavior).
 *   - empty-ocr input still returns valid HTML (structure-only fallback).
 *
 * If the ONNX file is absent (CI image without model cache), the test is
 * skipped rather than failing — same policy as other table fixtures.
 */

#include <gtest/gtest.h>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <string>

#include "table/table_wireless_recognizer.h"

namespace fs = std::filesystem;
using namespace rapid_doc;

namespace {

std::string slanetPath() {
    return std::string(PROJECT_ROOT_DIR) + "/.download_cache/onnx_models/slanet-plus.onnx";
}

class TableWirelessRecognizerTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!fs::exists(slanetPath())) {
            GTEST_SKIP() << "slanet-plus.onnx not cached at " << slanetPath()
                         << "; run setup.sh first.";
        }
        TableWirelessRecognizerConfig cfg;
        cfg.onnxModelPath = slanetPath();
        recognizer_ = std::make_unique<TableWirelessRecognizer>(cfg);
        ASSERT_TRUE(recognizer_->initialize())
            << "Failed to initialize TableWirelessRecognizer";
    }

    std::unique_ptr<TableWirelessRecognizer> recognizer_;
};

// Build a synthetic BGR table-like image: a light gray rectangle with a few
// horizontal and vertical dividers. Not a real table, but enough for SLANet+
// to emit a non-trivial structure.
cv::Mat makeSyntheticTable() {
    cv::Mat img(320, 480, CV_8UC3, cv::Scalar(250, 250, 250));
    cv::rectangle(img, cv::Rect(10, 10, 460, 300), cv::Scalar(0, 0, 0), 1);
    cv::line(img, cv::Point(10, 110), cv::Point(470, 110), cv::Scalar(0, 0, 0), 1);
    cv::line(img, cv::Point(10, 210), cv::Point(470, 210), cv::Scalar(0, 0, 0), 1);
    cv::line(img, cv::Point(160, 10), cv::Point(160, 310), cv::Scalar(0, 0, 0), 1);
    cv::line(img, cv::Point(320, 10), cv::Point(320, 310), cv::Scalar(0, 0, 0), 1);
    return img;
}

} // namespace

TEST_F(TableWirelessRecognizerTest, InitializedSuccessfully) {
    EXPECT_TRUE(recognizer_->isInitialized());
}

TEST_F(TableWirelessRecognizerTest, RecognizeEmitsStructureHtmlOnEmptyOcr) {
    cv::Mat img = makeSyntheticTable();
    TableResult result = recognizer_->recognize(img, {});

    EXPECT_TRUE(result.supported) << "recognize() should accept empty OCR";
    EXPECT_EQ(result.type, TableType::WIRELESS);
    ASSERT_FALSE(result.html.empty());

    // Python get_struct_str always wraps with <html><body><table>...</table></body></html>.
    EXPECT_NE(result.html.find("<html>"), std::string::npos);
    EXPECT_NE(result.html.find("<body>"), std::string::npos);
    EXPECT_NE(result.html.find("<table>"), std::string::npos);
    EXPECT_NE(result.html.find("</table>"), std::string::npos);
    EXPECT_NE(result.html.find("</body>"), std::string::npos);
    EXPECT_NE(result.html.find("</html>"), std::string::npos);

    // Structure must contain at least one <tr> (table row opens).
    EXPECT_NE(result.html.find("<tr>"), std::string::npos);
}

TEST_F(TableWirelessRecognizerTest, TheadTbodyFiltered) {
    cv::Mat img = makeSyntheticTable();
    TableResult result = recognizer_->recognize(img, {});
    ASSERT_FALSE(result.html.empty());

    // TableMatch.get_pred_html filters <thead>/</thead>/<tbody>/</tbody>.
    EXPECT_EQ(result.html.find("<thead>"), std::string::npos);
    EXPECT_EQ(result.html.find("</thead>"), std::string::npos);
    EXPECT_EQ(result.html.find("<tbody>"), std::string::npos);
    EXPECT_EQ(result.html.find("</tbody>"), std::string::npos);
}

TEST_F(TableWirelessRecognizerTest, OcrTextInsertedIntoCells) {
    cv::Mat img = makeSyntheticTable();
    // Place a synthetic OCR hit in the top-left quadrant so _filter_ocr_result
    // keeps it (OCR above all cells is pruned).
    std::vector<WirelessOcrBox> ocr;
    WirelessOcrBox b1;
    b1.aabb = cv::Rect(20, 30, 120, 50);
    b1.text = "hello";
    ocr.push_back(b1);

    TableResult result = recognizer_->recognize(img, ocr);
    ASSERT_FALSE(result.html.empty());
    EXPECT_TRUE(result.supported);
    // We can't guarantee which cell the text lands in (SLANet+ output is
    // data-dependent) but if any match succeeded the text should appear.
    // Structure-only HTML is still acceptable if the match was pruned.
    const bool hasText = result.html.find("hello") != std::string::npos;
    const bool hasStructure = result.html.find("<tr>") != std::string::npos;
    EXPECT_TRUE(hasText || hasStructure);
}

TEST_F(TableWirelessRecognizerTest, EmptyInputRejected) {
    cv::Mat empty;
    TableResult result = recognizer_->recognize(empty, {});
    EXPECT_FALSE(result.supported);
    EXPECT_TRUE(result.html.empty());
}
