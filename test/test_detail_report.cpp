#include <gtest/gtest.h>

#include "output/detail_report.h"

using namespace rapid_doc;

namespace {

ContentElement makeElement(
    ContentElement::Type type,
    const std::string& text,
    bool skipped,
    const std::string& imagePath = {})
{
    ContentElement elem;
    elem.type = type;
    elem.text = text;
    elem.skipped = skipped;
    elem.imagePath = imagePath;
    elem.layoutBox = LayoutBox{10.0f, 20.0f, 30.0f, 40.0f, LayoutCategory::TEXT, 0.9f, 0, 0, "text"};
    return elem;
}

} // namespace

TEST(DetailReportTest, reportIncludesStageTimingsArtifactsAndFlaggedElements) {
    DocumentResult result;
    result.totalPages = 1;
    result.processedPages = 1;
    result.skippedElements = 2;
    result.totalTimeMs = 123.0;
    result.stats.pdfRenderTimeMs = 10.0;
    result.stats.layoutTimeMs = 20.0;
    result.stats.ocrTimeMs = 30.0;
    result.stats.tableTimeMs = 40.0;
    result.stats.figureTimeMs = 5.0;
    result.stats.formulaTimeMs = 6.0;
    result.stats.unsupportedTimeMs = 7.0;
    result.stats.readingOrderTimeMs = 8.0;
    result.stats.outputGenTimeMs = 9.0;
    result.stats.npuSerialTimeMs = 90.0;
    result.stats.cpuOnlyTimeMs = 33.0;
    result.stats.npuLockWaitTimeMs = 5.0;
    result.stats.npuLockHoldTimeMs = 95.0;

    PageResult page;
    page.pageIndex = 0;
    page.pageWidth = 800;
    page.pageHeight = 1200;
    page.totalTimeMs = 111.0;
    page.stats.layoutTimeMs = 20.0;
    page.stats.ocrTimeMs = 30.0;
    page.stats.tableTimeMs = 40.0;
    page.stats.figureTimeMs = 5.0;
    page.stats.formulaTimeMs = 6.0;
    page.stats.unsupportedTimeMs = 7.0;
    page.stats.readingOrderTimeMs = 8.0;
    page.stats.npuSerialTimeMs = 90.0;
    page.stats.cpuOnlyTimeMs = 33.0;
    page.stats.npuLockWaitTimeMs = 5.0;
    page.stats.npuLockHoldTimeMs = 95.0;
    page.elements.push_back(makeElement(
        ContentElement::Type::TABLE,
        "[Unsupported table: wireless_table]",
        true));
    page.elements.push_back(makeElement(
        ContentElement::Type::UNKNOWN,
        "[Unsupported layout category: toc]",
        true));
    page.elements.push_back(makeElement(
        ContentElement::Type::IMAGE,
        "",
        false,
        "images/page0_fig0.png"));
    result.pages.push_back(page);

    DetailReportOptions options;
    options.inputPath = "/tmp/sample.pdf";
    options.stageConfig.enablePdfRender = true;
    options.stageConfig.enableLayout = true;
    options.stageConfig.enableOcr = true;
    options.stageConfig.enableWiredTable = true;
    options.stageConfig.enableFormula = true;
    options.stageConfig.enableReadingOrder = true;
    options.stageConfig.enableMarkdownOutput = true;
    options.saveImages = true;
    options.saveVisualization = true;
    options.artifacts.outputDir = "/tmp/out";
    options.artifacts.markdownPath = "/tmp/out/sample.md";
    options.artifacts.contentListPath = "/tmp/out/sample_content.json";
    options.artifacts.layoutDir = "/tmp/out/layout";
    options.artifacts.imagesDir = "/tmp/out/images";

    const std::string report = buildDetailReport(result, options);

    EXPECT_NE(report.find("RapidDoc Detail Report"), std::string::npos);
    EXPECT_NE(report.find("pdf_render=ON"), std::string::npos);
    EXPECT_NE(report.find("/tmp/out/sample.md"), std::string::npos);
    EXPECT_NE(report.find("tracked_stage_time"), std::string::npos);
    EXPECT_NE(report.find("npu_serial"), std::string::npos);
    EXPECT_NE(report.find("cpu_only"), std::string::npos);
    EXPECT_NE(report.find("npu_lock_wait"), std::string::npos);
    EXPECT_NE(report.find("npu_lock_hold"), std::string::npos);
    EXPECT_NE(report.find("Page 0"), std::string::npos);
    EXPECT_NE(report.find("[fallback]"), std::string::npos);
    EXPECT_NE(report.find("[unsupported]"), std::string::npos);
    EXPECT_NE(report.find("images/page0_fig0.png"), std::string::npos);
    EXPECT_NE(report.find("wireless_table"), std::string::npos);
}

TEST(DetailReportTest, reportMarksWhenNoFlaggedElementsExist) {
    DocumentResult result;
    result.totalPages = 1;
    result.processedPages = 1;

    PageResult page;
    page.pageIndex = 2;
    page.pageWidth = 100;
    page.pageHeight = 100;
    page.elements.push_back(makeElement(ContentElement::Type::TEXT, "hello", false));
    result.pages.push_back(page);

    DetailReportOptions options;
    options.stageConfig.enablePdfRender = false;

    const std::string report = buildDetailReport(result, options);
    EXPECT_NE(report.find("no skipped/fallback/unsupported elements"), std::string::npos);
}
