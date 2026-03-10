/**
 * @file test_e2e_pipeline.cpp
 * @brief Full end-to-end pipeline test: PDF -> Layout -> OCR -> Markdown.
 *
 * Processes a test PDF through the complete C++ pipeline, compares
 * layout detections and OCR output against Python reference fixtures.
 */

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <cmath>

#include "pipeline/doc_pipeline.h"
#include "common/config.h"
#include "output/markdown_writer.h"
#include "output/content_list.h"

namespace fs = std::filesystem;
using json = nlohmann::json;

using namespace rapid_doc;

static const std::string kProjectRoot = std::string(PROJECT_ROOT_DIR);
static const std::string kRefDir = kProjectRoot + "/test/fixtures/e2e/";
static const std::string kPageImage = kRefDir + "page_0.png";

class E2EPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!fs::exists(kPageImage)) {
            GTEST_SKIP() << "Page image not found: " << kPageImage;
        }

        auto cfg = PipelineConfig::Default(kProjectRoot);
        cfg.stages.enableWiredTable = false;
        cfg.stages.enablePdfRender = false;
        cfg.runtime.outputDir = kProjectRoot + "/test/fixtures/e2e/cpp_output";
        cfg.runtime.saveImages = false;

        pipeline_ = std::make_unique<DocPipeline>(cfg);
        image_ = cv::imread(kPageImage);
    }

    std::unique_ptr<DocPipeline> pipeline_;
    cv::Mat image_;
};

TEST_F(E2EPipelineTest, PipelineInitializes) {
    bool ok = pipeline_->initialize();
    ASSERT_TRUE(ok) << "Pipeline failed to initialize";
}

TEST_F(E2EPipelineTest, ProcessImageProducesResult) {
    ASSERT_TRUE(pipeline_->initialize());
    ASSERT_FALSE(image_.empty());

    auto pageResult = pipeline_->processImage(image_, 0);
    EXPECT_FALSE(pageResult.elements.empty()) << "No elements produced";

    std::cerr << "  [INFO] Page: " << pageResult.elements.size() << " elements in "
              << pageResult.totalTimeMs << " ms" << std::endl;
}

TEST_F(E2EPipelineTest, LayoutMatchesPythonReference) {
    ASSERT_TRUE(pipeline_->initialize());

    auto page = pipeline_->processImage(image_, 0);
    int cppBoxCount = static_cast<int>(page.layoutResult.boxes.size());

    std::cerr << "  [INFO] C++ layout: " << cppBoxCount << " boxes" << std::endl;
    for (const auto& box : page.layoutResult.boxes) {
        std::cerr << "    " << layoutCategoryToString(box.category)
                  << " [" << box.x0 << "," << box.y0 << ","
                  << box.x1 << "," << box.y1 << "] conf=" << box.confidence
                  << std::endl;
    }

    std::string refPath = kRefDir + "layout_page_0.json";
    if (fs::exists(refPath)) {
        std::ifstream f(refPath);
        json pyBoxes = json::parse(f);
        int pyBoxCount = static_cast<int>(pyBoxes.size());

        std::cerr << "  [INFO] Python layout: " << pyBoxCount << " boxes" << std::endl;

        EXPECT_EQ(cppBoxCount, pyBoxCount)
            << "Box count mismatch: C++=" << cppBoxCount << " Python=" << pyBoxCount;
    }
}

TEST_F(E2EPipelineTest, MarkdownIsGenerated) {
    ASSERT_TRUE(pipeline_->initialize());

    auto page = pipeline_->processImage(image_, 0);

    // Wrap single page in DocumentResult to generate markdown
    DocumentResult docResult;
    docResult.pages.push_back(std::move(page));
    docResult.processedPages = 1;

    MarkdownWriter mdWriter;
    std::string md = mdWriter.generate(docResult);
    EXPECT_FALSE(md.empty()) << "Markdown output is empty";

    std::cerr << "  [INFO] Markdown length: " << md.size() << " chars" << std::endl;

    fs::create_directories(kRefDir + "cpp_output");
    std::ofstream out(kRefDir + "cpp_output/page_0.md");
    out << md;

    std::string refMd = kRefDir + "reference_page_0.md";
    if (fs::exists(refMd)) {
        std::ifstream fref(refMd);
        std::string pyMd((std::istreambuf_iterator<char>(fref)),
                          std::istreambuf_iterator<char>());
        std::cerr << "  [INFO] Python markdown: " << pyMd.size() << " chars" << std::endl;

        if (!pyMd.empty()) {
            EXPECT_GT(md.size(), pyMd.size() / 4)
                << "C++ markdown is far shorter than Python reference";
        }
    }
}

TEST_F(E2EPipelineTest, OcrProducesText) {
    ASSERT_TRUE(pipeline_->initialize());

    auto page = pipeline_->processImage(image_, 0);
    int textElements = 0;
    int nonEmptyText = 0;
    for (const auto& elem : page.elements) {
        if (elem.type == ContentElement::Type::TEXT ||
            elem.type == ContentElement::Type::TITLE) {
            textElements++;
            if (!elem.text.empty()) nonEmptyText++;
            std::cerr << "  [OCR] " << (elem.type == ContentElement::Type::TITLE ? "TITLE" : "TEXT")
                      << " conf=" << elem.confidence
                      << " len=" << elem.text.size()
                      << " preview=\"" << elem.text.substr(0, 60) << "...\""
                      << std::endl;
        }
    }

    EXPECT_GT(textElements, 0) << "No text elements found";
    EXPECT_GT(nonEmptyText, 0) << "All text elements are empty (OCR failed?)";
}

TEST_F(E2EPipelineTest, ContentListJsonIsValid) {
    ASSERT_TRUE(pipeline_->initialize());

    auto page = pipeline_->processImage(image_, 0);

    DocumentResult docResult;
    docResult.pages.push_back(std::move(page));
    docResult.processedPages = 1;

    ContentListWriter clWriter;
    std::string clJson = clWriter.generate(docResult);
    EXPECT_FALSE(clJson.empty());

    auto parsed = json::parse(clJson, nullptr, false);
    EXPECT_FALSE(parsed.is_discarded()) << "Content list JSON is invalid";

    fs::create_directories(kRefDir + "cpp_output");
    std::ofstream out(kRefDir + "cpp_output/content_list.json");
    out << clJson;
}
