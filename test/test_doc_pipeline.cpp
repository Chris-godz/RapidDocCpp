/**
 * @file test_doc_pipeline.cpp
 * @brief Tests for DocPipeline
 *
 * Unit tests: config defaults, stage toggles, empty/small input handling.
 * Optional integration: process a real PDF (skip if not found).
 */

#include <gtest/gtest.h>
#include "pipeline/doc_pipeline.h"
#include "common/config.h"
#include <opencv2/opencv.hpp>
#include <filesystem>

namespace fs = std::filesystem;

namespace {

const char* SAMPLE_PDF = PROJECT_ROOT_DIR
    "/3rd-party/DXNN-OCR-cpp/server/webui/examples_pdf/3M-7770.pdf";

bool samplePdfExists() {
    return fs::exists(SAMPLE_PDF);
}

} // anonymous namespace

// ========================================
// Config Tests
// ========================================

TEST(DocPipeline, DefaultPipelineConfig) {
    rapid_doc::PipelineConfig config;

    EXPECT_EQ(config.runtime.pdfDpi, 200);
    EXPECT_EQ(config.runtime.maxPages, 0);
    EXPECT_EQ(config.runtime.maxConcurrentPages, 4);
    EXPECT_EQ(config.runtime.layoutInputSize, 640);
    EXPECT_FLOAT_EQ(config.runtime.layoutConfThreshold, 0.4f);
    EXPECT_EQ(config.runtime.outputDir, "./output");
    EXPECT_TRUE(config.runtime.saveImages);
}

TEST(DocPipeline, StageConfigDefaults) {
    rapid_doc::PipelineConfig config;

    EXPECT_TRUE(config.stages.enablePdfRender);
    EXPECT_TRUE(config.stages.enableLayout);
    EXPECT_TRUE(config.stages.enableOcr);
    EXPECT_TRUE(config.stages.enableWiredTable);
    EXPECT_TRUE(config.stages.enableReadingOrder);
    EXPECT_TRUE(config.stages.enableMarkdownOutput);
}

TEST(DocPipeline, StageDisable) {
    rapid_doc::PipelineConfig config;
    config.stages.enablePdfRender = false;
    config.stages.enableLayout = false;
    config.stages.enableOcr = false;
    config.stages.enableWiredTable = false;

    EXPECT_FALSE(config.stages.enablePdfRender);
    EXPECT_FALSE(config.stages.enableLayout);
    EXPECT_FALSE(config.stages.enableOcr);
    EXPECT_FALSE(config.stages.enableWiredTable);
}

// ========================================
// Initialization Tests
// ========================================

TEST(DocPipeline, InitializeWithAllDisabled) {
    rapid_doc::PipelineConfig config;
    config.stages.enablePdfRender = false;
    config.stages.enableLayout = false;
    config.stages.enableOcr = false;
    config.stages.enableWiredTable = false;
    config.stages.enableReadingOrder = false;
    config.stages.enableMarkdownOutput = true;

    rapid_doc::DocPipeline pipeline(config);
    bool ok = pipeline.initialize();

    EXPECT_TRUE(ok);
    EXPECT_TRUE(pipeline.isInitialized());
}

TEST(DocPipeline, ConfigValidate_Empty) {
    rapid_doc::PipelineConfig config;
    config.models.ocrModelDir = "";

    std::string err = config.validate();
    EXPECT_FALSE(err.empty());
}

TEST(DocPipeline, ConfigValidate_Valid) {
    rapid_doc::PipelineConfig config;
    config.stages.enableLayout = false;
    config.stages.enableOcr = false;
    config.stages.enableWiredTable = false;
    config.models.ocrModelDir = "/tmp/models";

    std::string err = config.validate();
    EXPECT_TRUE(err.empty());
}

// ========================================
// Process Tests (No Model / Synthetic Data)
// ========================================

TEST(DocPipeline, ProcessEmptyImage) {
    rapid_doc::PipelineConfig config;
    config.stages.enablePdfRender = false;
    config.stages.enableLayout = false;
    config.stages.enableOcr = false;
    config.stages.enableWiredTable = false;
    config.stages.enableReadingOrder = false;
    config.stages.enableMarkdownOutput = true;

    rapid_doc::DocPipeline pipeline(config);
    ASSERT_TRUE(pipeline.initialize());

    cv::Mat emptyImg;
    auto result = pipeline.processImage(emptyImg, 0);

    EXPECT_EQ(result.pageIndex, 0);
    EXPECT_TRUE(result.elements.empty());
    EXPECT_GE(result.totalTimeMs, 0.0);
}

TEST(DocPipeline, ProcessSmallImage) {
    rapid_doc::PipelineConfig config;
    config.stages.enablePdfRender = false;
    config.stages.enableLayout = false;
    config.stages.enableOcr = false;
    config.stages.enableWiredTable = false;
    config.stages.enableReadingOrder = false;
    config.stages.enableMarkdownOutput = true;

    rapid_doc::DocPipeline pipeline(config);
    ASSERT_TRUE(pipeline.initialize());

    cv::Mat smallImg(100, 100, CV_8UC3, cv::Scalar(255, 255, 255));
    auto result = pipeline.processImage(smallImg, 0);

    EXPECT_EQ(result.pageIndex, 0);
    EXPECT_GE(result.totalTimeMs, 0.0);
}

TEST(DocPipeline, ProcessImageWithReadingOrder) {
    rapid_doc::PipelineConfig config;
    config.stages.enablePdfRender = false;
    config.stages.enableLayout = false;
    config.stages.enableOcr = false;
    config.stages.enableWiredTable = false;
    config.stages.enableReadingOrder = true;
    config.stages.enableMarkdownOutput = true;

    rapid_doc::DocPipeline pipeline(config);
    ASSERT_TRUE(pipeline.initialize());

    cv::Mat img(200, 200, CV_8UC3, cv::Scalar(255, 255, 255));
    auto result = pipeline.processImage(img, 5);

    EXPECT_EQ(result.pageIndex, 5);
    EXPECT_GE(result.totalTimeMs, 0.0);
}

// ========================================
// Result Structure Tests
// ========================================

TEST(DocPipeline, DocumentResultEmpty) {
    rapid_doc::DocumentResult result;

    EXPECT_EQ(result.totalPages, 0);
    EXPECT_EQ(result.processedPages, 0);
    EXPECT_EQ(result.skippedElements, 0);
    EXPECT_TRUE(result.pages.empty());
    EXPECT_TRUE(result.markdown.empty());
    EXPECT_TRUE(result.contentListJson.empty());
    EXPECT_GE(result.totalTimeMs, 0.0);
}

TEST(DocPipeline, PageResultEmpty) {
    rapid_doc::PageResult result;

    EXPECT_EQ(result.pageIndex, 0);
    EXPECT_TRUE(result.elements.empty());
    EXPECT_GE(result.totalTimeMs, 0.0);
}

// ========================================
// Progress Callback Test
// ========================================

TEST(DocPipeline, ProgressCallback) {
    rapid_doc::PipelineConfig config;
    config.stages.enablePdfRender = false;
    config.stages.enableLayout = false;
    config.stages.enableOcr = false;
    config.stages.enableWiredTable = false;
    config.stages.enableReadingOrder = false;
    config.stages.enableMarkdownOutput = false;

    rapid_doc::DocPipeline pipeline(config);
    ASSERT_TRUE(pipeline.initialize());

    int callCount = 0;
    pipeline.setProgressCallback([&callCount](const std::string& stage, int current, int total) {
        callCount++;
    });

    cv::Mat img(50, 50, CV_8UC3, cv::Scalar(255, 255, 255));
    pipeline.processImage(img, 0);
}

// ========================================
// Integration Test (Optional - requires sample PDF)
// ========================================

TEST(DocPipeline, DISABLED_ProcessPdf_WithSample) {
    if (!samplePdfExists()) GTEST_SKIP() << "Sample PDF not found";

    rapid_doc::PipelineConfig config;
    config.stages.enablePdfRender = true;
    config.stages.enableLayout = false;
    config.stages.enableOcr = false;
    config.stages.enableWiredTable = false;
    config.stages.enableReadingOrder = false;
    config.stages.enableMarkdownOutput = true;

    rapid_doc::DocPipeline pipeline(config);
    ASSERT_TRUE(pipeline.initialize());

    auto result = pipeline.processPdf(SAMPLE_PDF);

    EXPECT_GE(result.totalPages, 1);
    EXPECT_GE(result.processedPages, 1);
    EXPECT_FALSE(result.markdown.empty());
    EXPECT_FALSE(result.contentListJson.empty());
}
