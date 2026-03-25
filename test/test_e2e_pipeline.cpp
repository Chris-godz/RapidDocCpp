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
#include <dxrt/exception/exception.h>

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
static const std::string kFallbackPageImage = kProjectRoot + "/test/fixtures/layout/input_image.png";
static const std::string kPdfFixture = kProjectRoot + "/test_files/small_ocr_origin.pdf";

namespace {

bool isDxrtRuntimeIssue(const dxrt::Exception& e) {
    return e.code() == dxrt::ERROR_CODE::SERVICE_IO ||
           e.code() == dxrt::ERROR_CODE::DEVICE_IO;
}

std::string resolvePageFixture() {
    if (fs::exists(kPageImage)) {
        return kPageImage;
    }
    if (fs::exists(kFallbackPageImage)) {
        return kFallbackPageImage;
    }
    return {};
}

} // namespace

class E2EPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        pageImagePath_ = resolvePageFixture();
        if (pageImagePath_.empty()) {
            GTEST_SKIP() << "No E2E page fixture found. Tried: "
                         << kPageImage << " and " << kFallbackPageImage;
        }

        auto cfg = PipelineConfig::Default(kProjectRoot);
        cfg.stages.enableWiredTable = false;
        cfg.stages.enablePdfRender = false;
        cfg.runtime.outputDir = kProjectRoot + "/test/fixtures/e2e/cpp_output";
        cfg.runtime.saveImages = false;

        pipeline_ = std::make_unique<DocPipeline>(cfg);
        image_ = cv::imread(pageImagePath_);
        if (image_.empty()) {
            GTEST_SKIP() << "Failed to read page image fixture: " << pageImagePath_;
        }
    }

    std::unique_ptr<DocPipeline> pipeline_;
    cv::Mat image_;
    std::string pageImagePath_;
};

TEST_F(E2EPipelineTest, PipelineInitializes) {
    try {
        bool ok = pipeline_->initialize();
        ASSERT_TRUE(ok) << "Pipeline failed to initialize";
    } catch (const dxrt::Exception& e) {
        if (isDxrtRuntimeIssue(e)) {
            GTEST_SKIP() << "DXRT runtime unavailable in E2E env: " << e.what();
        }
        FAIL() << "DXRT exception during initialize: " << e.what();
    } catch (const std::exception& e) {
        FAIL() << "Unexpected std::exception during initialize: " << e.what();
    } catch (...) {
        FAIL() << "Unknown non-std exception during initialize";
    }
}

TEST_F(E2EPipelineTest, ProcessImageProducesResult) {
    try {
        ASSERT_TRUE(pipeline_->initialize());
        ASSERT_FALSE(image_.empty());

        auto pageResult = pipeline_->processImage(image_, 0);
        EXPECT_FALSE(pageResult.elements.empty()) << "No elements produced";

        std::cerr << "  [INFO] Page: " << pageResult.elements.size() << " elements in "
                  << pageResult.totalTimeMs << " ms" << std::endl;
    } catch (const dxrt::Exception& e) {
        if (isDxrtRuntimeIssue(e)) {
            GTEST_SKIP() << "DXRT runtime unavailable in E2E env: " << e.what();
        }
        FAIL() << "DXRT exception during image process: " << e.what();
    } catch (const std::exception& e) {
        FAIL() << "Unexpected std::exception during image process: " << e.what();
    } catch (...) {
        FAIL() << "Unknown non-std exception during image process";
    }
}

TEST_F(E2EPipelineTest, LayoutMatchesPythonReference) {
    try {
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
    } catch (const dxrt::Exception& e) {
        if (isDxrtRuntimeIssue(e)) {
            GTEST_SKIP() << "DXRT runtime unavailable in E2E env: " << e.what();
        }
        FAIL() << "DXRT exception during layout compare: " << e.what();
    } catch (const std::exception& e) {
        FAIL() << "Unexpected std::exception during layout compare: " << e.what();
    } catch (...) {
        FAIL() << "Unknown non-std exception during layout compare";
    }
}

TEST_F(E2EPipelineTest, MarkdownIsGenerated) {
    try {
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
    } catch (const dxrt::Exception& e) {
        if (isDxrtRuntimeIssue(e)) {
            GTEST_SKIP() << "DXRT runtime unavailable in E2E env: " << e.what();
        }
        FAIL() << "DXRT exception during markdown generation: " << e.what();
    } catch (const std::exception& e) {
        FAIL() << "Unexpected std::exception during markdown generation: " << e.what();
    } catch (...) {
        FAIL() << "Unknown non-std exception during markdown generation";
    }
}

TEST_F(E2EPipelineTest, OcrProducesText) {
    try {
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
    } catch (const dxrt::Exception& e) {
        if (isDxrtRuntimeIssue(e)) {
            GTEST_SKIP() << "DXRT runtime unavailable in E2E env: " << e.what();
        }
        FAIL() << "DXRT exception during OCR validation: " << e.what();
    } catch (const std::exception& e) {
        FAIL() << "Unexpected std::exception during OCR validation: " << e.what();
    } catch (...) {
        FAIL() << "Unknown non-std exception during OCR validation";
    }
}

TEST_F(E2EPipelineTest, ContentListJsonIsValid) {
    try {
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
    } catch (const dxrt::Exception& e) {
        if (isDxrtRuntimeIssue(e)) {
            GTEST_SKIP() << "DXRT runtime unavailable in E2E env: " << e.what();
        }
        FAIL() << "DXRT exception during content_list validation: " << e.what();
    } catch (const std::exception& e) {
        FAIL() << "Unexpected std::exception during content_list validation: " << e.what();
    } catch (...) {
        FAIL() << "Unknown non-std exception during content_list validation";
    }
}

class E2ENoDxrtPipelineTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!fs::exists(kPdfFixture)) {
            GTEST_SKIP() << "PDF fixture not found: " << kPdfFixture;
        }

        auto cfg = PipelineConfig::Default(kProjectRoot);
        cfg.stages.enableLayout = false;
        cfg.stages.enableOcr = false;
        cfg.stages.enableWiredTable = false;
        cfg.stages.enableFormula = false;
        cfg.stages.enableReadingOrder = false;
        cfg.stages.enableMarkdownOutput = true;
        cfg.runtime.outputDir = kProjectRoot + "/test/fixtures/e2e/cpp_output_no_dxrt";
        cfg.runtime.saveImages = false;
        cfg.runtime.saveVisualization = false;

        pipeline_ = std::make_unique<DocPipeline>(cfg);
    }

    std::unique_ptr<DocPipeline> pipeline_;
};

TEST_F(E2ENoDxrtPipelineTest, RenderOnlyPdfProducesPages) {
    ASSERT_TRUE(pipeline_->initialize());
    const auto result = pipeline_->processPdf(kPdfFixture);

    EXPECT_GE(result.totalPages, 1);
    EXPECT_EQ(result.processedPages, static_cast<int>(result.pages.size()));
    EXPECT_GE(result.processedPages, 1);
}

TEST_F(E2ENoDxrtPipelineTest, RenderOnlyPdfContentListShapeMatchesPages) {
    ASSERT_TRUE(pipeline_->initialize());
    const auto result = pipeline_->processPdf(kPdfFixture);

    const auto contentList = json::parse(result.contentListJson);
    ASSERT_TRUE(contentList.is_array());
    EXPECT_EQ(static_cast<int>(contentList.size()), result.processedPages);
    for (const auto& pageItems : contentList) {
        EXPECT_TRUE(pageItems.is_array());
    }
}
