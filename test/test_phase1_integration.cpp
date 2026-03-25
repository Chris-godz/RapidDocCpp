#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <filesystem>
#include <fstream>
#include <deque>
#include <string>
#include <tuple>
#include <sstream>

#include "common/config.h"
#include "common/types.h"
#include "output/content_list.h"
#include "output/markdown_writer.h"
#include "pipeline/doc_pipeline.h"
#include "server/server.h"
#include "test_access.h"

using json = nlohmann::json;
using namespace rapid_doc;

namespace {

PipelineConfig makePdfOnlyPipelineConfig() {
    PipelineConfig cfg = PipelineConfig::Default(PROJECT_ROOT_DIR);
    cfg.stages.enableLayout = false;
    cfg.stages.enableOcr = false;
    cfg.stages.enableWiredTable = false;
    cfg.stages.enableFormula = false;
    cfg.stages.enableReadingOrder = false;
    cfg.stages.enableMarkdownOutput = true;
    cfg.runtime.outputDir = std::string(PROJECT_ROOT_DIR) + "/test/fixtures/integration_output";
    return cfg;
}

LayoutBox makeBox(LayoutCategory category, float x0, float y0, float x1, float y1) {
    LayoutBox box{};
    box.x0 = x0;
    box.y0 = y0;
    box.x1 = x1;
    box.y1 = y1;
    box.category = category;
    box.confidence = 0.9f;
    box.index = 0;
    box.clsId = static_cast<int>(category);
    box.label = layoutCategoryToString(category);
    return box;
}

std::string readFileBytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return {};
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

ocr::PipelineOCRResult makeOcrResult(const std::string& text) {
    ocr::PipelineOCRResult result;
    result.text = text;
    result.confidence = 0.99f;
    result.index = 0;
    result.box = {
        cv::Point2f(0.0f, 0.0f),
        cv::Point2f(10.0f, 0.0f),
        cv::Point2f(10.0f, 10.0f),
        cv::Point2f(0.0f, 10.0f),
    };
    return result;
}

} // namespace

TEST(Phase1Integration, unknown_layout_category_survives_pipeline) {
    auto cfg = makePdfOnlyPipelineConfig();
    cfg.stages.enablePdfRender = false;
    DocPipeline pipeline(cfg);

    const int pageIndex = 5;
    LayoutResult layoutResult;
    layoutResult.boxes = {
        makeBox(LayoutCategory::TEXT, 1, 1, 4, 4),
        makeBox(LayoutCategory::UNKNOWN, 5, 5, 30, 20),
        makeBox(LayoutCategory::TOC, 5, 22, 30, 40),
        makeBox(LayoutCategory::SEPARATOR, 5, 42, 30, 55),
    };
    const auto unsupportedBoxes = layoutResult.getUnsupportedBoxes();
    ASSERT_EQ(unsupportedBoxes.size(), 3u);

    auto elems = DocPipelineTestAccess::handleUnsupportedElements(
        pipeline, unsupportedBoxes, pageIndex);
    ASSERT_EQ(elems.size(), unsupportedBoxes.size());
    for (const auto& elem : elems) {
        EXPECT_TRUE(elem.skipped);
        EXPECT_EQ(elem.pageIndex, pageIndex);
        EXPECT_EQ(elem.type, ContentElement::Type::UNKNOWN);
        EXPECT_NE(elem.text.find("Unsupported layout category"), std::string::npos);
    }

    DocumentResult doc;
    PageResult page;
    page.pageIndex = pageIndex;
    page.pageWidth = 100;
    page.pageHeight = 100;
    page.elements = elems;
    doc.pages.push_back(page);

    MarkdownWriter mdWriter;
    ContentListWriter clWriter;
    const std::string md = mdWriter.generate(doc);
    const json contentList = json::parse(clWriter.generate(doc));

    EXPECT_NE(md.find("unknown"), std::string::npos);
    EXPECT_NE(md.find("toc"), std::string::npos);
    EXPECT_NE(md.find("separator"), std::string::npos);
    ASSERT_EQ(contentList.size(), 1u);
    ASSERT_EQ(contentList[0].size(), 3u);
    for (const auto& item : contentList[0]) {
        EXPECT_EQ(item["type"], "unknown");
        EXPECT_EQ(item["page"], pageIndex);
        EXPECT_EQ(item["skipped"], true);
    }
}

TEST(Phase1Integration, process_pdf_two_pages_preserves_page_indices) {
    const std::filesystem::path pdfPath =
        std::filesystem::path(PROJECT_ROOT_DIR) / "test_files" / "small_ocr_origin.pdf";
    if (!std::filesystem::exists(pdfPath)) {
        GTEST_SKIP() << "Missing fixture PDF: " << pdfPath;
    }

    auto cfg = makePdfOnlyPipelineConfig();
    cfg.runtime.startPageId = 0;
    cfg.runtime.endPageId = 1;

    DocPipeline pipeline(cfg);
    ASSERT_TRUE(pipeline.initialize());

    const DocumentResult result = pipeline.processPdf(pdfPath.string());
    ASSERT_EQ(result.pages.size(), 2u);
    EXPECT_EQ(result.totalPages, 2);
    EXPECT_EQ(result.processedPages, 2);
    EXPECT_EQ(result.pages[0].pageIndex, 0);
    EXPECT_EQ(result.pages[1].pageIndex, 1);

    const json contentList = json::parse(result.contentListJson);
    ASSERT_EQ(contentList.size(), 2u);
    EXPECT_TRUE(contentList[0].is_array());
    EXPECT_TRUE(contentList[1].is_array());
}

TEST(Phase1Integration, multi_page_content_list_server_contract) {
    const std::filesystem::path pdfPath =
        std::filesystem::path(PROJECT_ROOT_DIR) / "test_files" / "BVRC_Meeting_Minutes_2024-04_origin.pdf";
    if (!std::filesystem::exists(pdfPath)) {
        GTEST_SKIP() << "Missing fixture PDF: " << pdfPath;
    }

    ServerConfig serverCfg;
    serverCfg.uploadDir = (std::filesystem::path(PROJECT_ROOT_DIR) / "test" / "fixtures" / "server_output").string();
    serverCfg.pipelineConfig = makePdfOnlyPipelineConfig();

    DocServer server(serverCfg);
    const std::string bytes = readFileBytes(pdfPath);
    ASSERT_FALSE(bytes.empty());

    const std::string responseText = DocServerTestAccess::handleProcess(
        server, bytes, pdfPath.filename().string());
    const json response = json::parse(responseText);

    ASSERT_TRUE(response.contains("pages"));
    ASSERT_TRUE(response.contains("total_pages"));
    ASSERT_TRUE(response.contains("content_list"));
    ASSERT_TRUE(response["content_list"].is_array());

    const int pages = response["pages"].get<int>();
    const int totalPages = response["total_pages"].get<int>();
    EXPECT_EQ(pages, totalPages);
    EXPECT_GE(pages, 2);
    EXPECT_EQ(static_cast<int>(response["content_list"].size()), pages);

    for (const auto& pageItems : response["content_list"]) {
        EXPECT_TRUE(pageItems.is_array());
    }
}

TEST(Phase1Integration, ocr_timeout_does_not_poison_next_request) {
    struct FakeOcrBackend {
        std::deque<std::tuple<int64_t, bool, std::vector<ocr::PipelineOCRResult>>> queue;

        bool push(const cv::Mat&, int64_t) { return true; }

        bool fetch(std::vector<ocr::PipelineOCRResult>& out, int64_t& id, bool& success) {
            if (queue.empty()) {
                return false;
            }
            auto item = std::move(queue.front());
            queue.pop_front();
            id = std::get<0>(item);
            success = std::get<1>(item);
            out = std::move(std::get<2>(item));
            return true;
        }
    } fake;

    auto cfg = makePdfOnlyPipelineConfig();
    DocPipeline pipeline(cfg);
    DocPipelineTestAccess::setOcrHooks(
        pipeline,
        [&fake](const cv::Mat& img, int64_t id) { return fake.push(img, id); },
        [&fake](std::vector<ocr::PipelineOCRResult>& out, int64_t& id, bool& success) {
            return fake.fetch(out, id, success);
        });
    DocPipelineTestAccess::setOcrTimeout(pipeline, std::chrono::milliseconds(10));

    cv::Mat crop(8, 8, CV_8UC3, cv::Scalar::all(255));

    // Request A timed out (no result ever returned during its wait window).
    EXPECT_TRUE(DocPipelineTestAccess::ocrOnCrop(pipeline, crop, 100).empty());

    // A stale result from request A arrives later, followed by request B's result.
    fake.queue.push_back({100, true, {makeOcrResult("stale-from-request-A")}});
    fake.queue.push_back({101, true, {makeOcrResult("fresh-from-request-B")}});

    EXPECT_EQ(
        DocPipelineTestAccess::ocrOnCrop(pipeline, crop, 101),
        "fresh-from-request-B");
}

TEST(Phase1Integration, table_model_missing_is_fail_fast) {
    auto cfg = makePdfOnlyPipelineConfig();
    cfg.stages.enableWiredTable = true;
    cfg.models.tableUnetDxnnModel =
        std::string(PROJECT_ROOT_DIR) + "/engine/model_files/table/_missing_unet.dxnn";

    DocPipeline pipeline(cfg);
    EXPECT_FALSE(pipeline.initialize());
}
