#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <deque>
#include <string>
#include <tuple>

#include "common/config.h"
#include "output/content_list.h"
#include "output/markdown_writer.h"
#include "pipeline/doc_pipeline.h"
#include "table/table_recognizer.h"
#include "test_access.h"

using json = nlohmann::json;
using namespace rapid_doc;

namespace {

PipelineConfig makeContractConfig() {
    PipelineConfig cfg = PipelineConfig::Default(PROJECT_ROOT_DIR);
    cfg.stages.enablePdfRender = false;
    cfg.stages.enableLayout = false;
    cfg.stages.enableReadingOrder = false;
    cfg.stages.enableMarkdownOutput = false;
    cfg.stages.enableWirelessTable = false;
    cfg.runtime.outputDir = std::string(PROJECT_ROOT_DIR) + "/test/fixtures/contract_output";
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

TableCell makeCell(int row, int col, int rowSpan, int colSpan, const std::string& content) {
    TableCell cell{};
    cell.row = row;
    cell.col = col;
    cell.rowSpan = rowSpan;
    cell.colSpan = colSpan;
    cell.x0 = static_cast<float>(col * 10);
    cell.y0 = static_cast<float>(row * 10);
    cell.x1 = cell.x0 + static_cast<float>(colSpan * 10);
    cell.y1 = cell.y0 + static_cast<float>(rowSpan * 10);
    cell.content = content;
    return cell;
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

size_t countSubstr(const std::string& s, const std::string& needle) {
    if (needle.empty()) return 0;
    size_t count = 0;
    size_t pos = 0;
    while ((pos = s.find(needle, pos)) != std::string::npos) {
        ++count;
        pos += needle.size();
    }
    return count;
}

} // namespace

TEST(Phase1CorrectnessContracts, ocr_result_must_match_task_id) {
    struct FakeOcrBackend {
        std::deque<std::tuple<int64_t, bool, std::vector<ocr::PipelineOCRResult>>> queue;
        std::vector<int64_t> pushed;

        bool push(const cv::Mat&, int64_t id) {
            pushed.push_back(id);
            return true;
        }

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

    fake.queue.push_back({12, true, {makeOcrResult("future-task")}}); // out-of-order first
    fake.queue.push_back({11, true, {makeOcrResult("target-task")}});

    auto cfg = makeContractConfig();
    cfg.stages.enableOcr = true;
    DocPipeline pipeline(cfg);
    DocPipelineTestAccess::setOcrHooks(
        pipeline,
        [&fake](const cv::Mat& img, int64_t id) { return fake.push(img, id); },
        [&fake](std::vector<ocr::PipelineOCRResult>& out, int64_t& id, bool& success) {
            return fake.fetch(out, id, success);
        });

    cv::Mat crop(8, 8, CV_8UC3, cv::Scalar::all(255));
    EXPECT_EQ(DocPipelineTestAccess::ocrOnCrop(pipeline, crop, 11), "target-task");
    EXPECT_EQ(DocPipelineTestAccess::ocrOnCrop(pipeline, crop, 12), "future-task");
}

TEST(Phase1CorrectnessContracts, content_element_page_index_propagation) {
    struct EchoOcrBackend {
        std::deque<int64_t> pending;

        bool push(const cv::Mat&, int64_t id) {
            pending.push_back(id);
            return true;
        }

        bool fetch(std::vector<ocr::PipelineOCRResult>& out, int64_t& id, bool& success) {
            if (pending.empty()) return false;
            id = pending.front();
            pending.pop_front();
            success = true;
            out = {makeOcrResult("ok")};
            return true;
        }
    } fakeOcr;

    auto cfg = makeContractConfig();
    cfg.stages.enableOcr = true;
    cfg.stages.enableWiredTable = true;
    DocPipeline pipeline(cfg);

    DocPipelineTestAccess::setOcrHooks(
        pipeline,
        [&fakeOcr](const cv::Mat& img, int64_t id) { return fakeOcr.push(img, id); },
        [&fakeOcr](std::vector<ocr::PipelineOCRResult>& out, int64_t& id, bool& success) {
            return fakeOcr.fetch(out, id, success);
        });

    DocPipelineTestAccess::setTableHooks(
        pipeline,
        [](const cv::Mat&) {
            TableResult result;
            result.type = TableType::WIRED;
            result.supported = true;
            result.cells = {makeCell(0, 0, 1, 1, "C")};
            return result;
        },
        [](const std::vector<TableCell>&) {
            return std::string("<table border=\"1\">\n  <tr><td>C</td></tr>\n</table>");
        });

    const int pageIndex = 7;
    cv::Mat page(80, 80, CV_8UC3, cv::Scalar::all(255));

    auto textElems = DocPipelineTestAccess::runOcrOnRegions(
        pipeline,
        page,
        {makeBox(LayoutCategory::TEXT, 2, 2, 30, 20)},
        pageIndex);
    ASSERT_EQ(textElems.size(), 1u);
    EXPECT_EQ(textElems[0].pageIndex, pageIndex);

    auto tableElems = DocPipelineTestAccess::runTableRecognition(
        pipeline,
        page,
        {makeBox(LayoutCategory::TABLE, 10, 20, 70, 70)},
        pageIndex);
    ASSERT_EQ(tableElems.size(), 1u);
    EXPECT_EQ(tableElems[0].pageIndex, pageIndex);

    auto unsupportedElems = DocPipelineTestAccess::handleUnsupportedElements(
        pipeline,
        {makeBox(LayoutCategory::UNKNOWN, 1, 1, 10, 10)},
        pageIndex);
    ASSERT_EQ(unsupportedElems.size(), 1u);
    EXPECT_EQ(unsupportedElems[0].pageIndex, pageIndex);
}

TEST(Phase1CorrectnessContracts, table_html_preserves_empty_leading_header_and_stub) {
    TableRecognizer recognizer(TableRecognizerConfig{});
    const std::vector<TableCell> cells = {
        makeCell(0, 0, 1, 1, ""),
        makeCell(0, 1, 1, 1, "H2"),
        makeCell(1, 0, 1, 1, ""),
        makeCell(1, 1, 1, 1, "A1"),
    };

    const std::string html = recognizer.generateHtml(cells);
    EXPECT_NE(html.find("<tr><td></td><td>H2</td></tr>"), std::string::npos);
    EXPECT_NE(html.find("<tr><td></td><td>A1</td></tr>"), std::string::npos);
}

TEST(Phase1CorrectnessContracts, table_html_rejects_or_detects_span_overlap) {
    TableRecognizer recognizer(TableRecognizerConfig{});
    const std::vector<TableCell> cells = {
        makeCell(0, 0, 1, 2, "A"),
        makeCell(0, 1, 1, 1, "B"), // overlaps with first cell span
    };
    EXPECT_THROW(recognizer.generateHtml(cells), std::runtime_error);
}

TEST(Phase1CorrectnessContracts, wireless_table_returns_defined_fallback) {
    auto cfg = makeContractConfig();
    cfg.stages.enableWiredTable = true;
    cfg.stages.enableOcr = false;
    DocPipeline pipeline(cfg);

    DocPipelineTestAccess::setTableHooks(pipeline, [](const cv::Mat&) {
        TableResult result;
        result.type = TableType::WIRELESS;
        result.supported = false;
        return result;
    });

    cv::Mat page(80, 120, CV_8UC3, cv::Scalar::all(255));
    auto elems = DocPipelineTestAccess::runTableRecognition(
        pipeline,
        page,
        {makeBox(LayoutCategory::TABLE, 5, 5, 70, 60)},
        3);

    ASSERT_EQ(elems.size(), 1u);
    const auto& elem = elems.front();
    EXPECT_EQ(elem.type, ContentElement::Type::TABLE);
    EXPECT_TRUE(elem.skipped);
    EXPECT_EQ(elem.pageIndex, 3);
    EXPECT_NE(elem.text.find("wireless_table"), std::string::npos);
    EXPECT_TRUE(elem.html.empty());

    DocumentResult doc;
    PageResult pageResult;
    pageResult.pageIndex = 3;
    pageResult.pageWidth = page.cols;
    pageResult.pageHeight = page.rows;
    pageResult.elements.push_back(elem);
    doc.pages.push_back(pageResult);

    MarkdownWriter mdWriter;
    ContentListWriter clWriter;
    std::string md = mdWriter.generate(doc);
    std::string cl = clWriter.generate(doc);
    auto parsed = json::parse(cl);

    ASSERT_EQ(parsed.size(), 1u);
    ASSERT_EQ(parsed[0].size(), 1u);
    EXPECT_NE(md.find("[Unsupported table: wireless_table]"), std::string::npos);
    EXPECT_EQ(parsed[0][0]["type"], "table");
    EXPECT_EQ(parsed[0][0]["skipped"], true);
    EXPECT_EQ(parsed[0][0]["text"], "[Unsupported table: wireless_table]");
}

TEST(Phase1CorrectnessContracts, no_cell_table_returns_defined_fallback) {
    auto cfg = makeContractConfig();
    cfg.stages.enableWiredTable = true;
    cfg.stages.enableOcr = false;
    DocPipeline pipeline(cfg);

    DocPipelineTestAccess::setTableHooks(pipeline, [](const cv::Mat&) {
        TableResult result;
        result.type = TableType::WIRED;
        result.supported = true;
        result.cells.clear();
        return result;
    });

    cv::Mat page(80, 120, CV_8UC3, cv::Scalar::all(255));
    auto elems = DocPipelineTestAccess::runTableRecognition(
        pipeline,
        page,
        {makeBox(LayoutCategory::TABLE, 5, 5, 70, 60)},
        1);

    ASSERT_EQ(elems.size(), 1u);
    EXPECT_TRUE(elems[0].skipped);
    EXPECT_EQ(elems[0].text, "[Unsupported table: no_cell_table]");
}

TEST(Phase1CorrectnessContracts, table_model_missing_fails_initialization) {
    auto cfg = makeContractConfig();
    cfg.stages.enableWiredTable = true;
    cfg.stages.enableOcr = false;
    cfg.models.tableUnetDxnnModel = std::string(PROJECT_ROOT_DIR) + "/engine/model_files/table/_missing_unet.dxnn";
    DocPipeline pipeline(cfg);
    EXPECT_FALSE(pipeline.initialize());
}

TEST(Phase1CorrectnessContracts, ocr_empty_table_preserves_structure) {
    auto cfg = makeContractConfig();
    cfg.stages.enableWiredTable = true;
    cfg.stages.enableOcr = false;
    DocPipeline pipeline(cfg);

    TableRecognizer htmlRenderer(TableRecognizerConfig{});
    DocPipelineTestAccess::setTableHooks(
        pipeline,
        [](const cv::Mat&) {
            TableResult result;
            result.type = TableType::WIRED;
            result.supported = true;
            result.cells = {
                makeCell(0, 0, 1, 1, ""),
                makeCell(0, 1, 1, 1, ""),
                makeCell(1, 0, 1, 1, ""),
                makeCell(1, 1, 1, 1, ""),
            };
            return result;
        },
        [&htmlRenderer](const std::vector<TableCell>& cells) {
            return htmlRenderer.generateHtml(cells);
        });

    cv::Mat page(80, 120, CV_8UC3, cv::Scalar::all(255));
    auto elems = DocPipelineTestAccess::runTableRecognition(
        pipeline,
        page,
        {makeBox(LayoutCategory::TABLE, 5, 5, 70, 60)},
        4);

    ASSERT_EQ(elems.size(), 1u);
    EXPECT_FALSE(elems[0].skipped);
    EXPECT_FALSE(elems[0].html.empty());
    EXPECT_EQ(countSubstr(elems[0].html, "<td></td>"), 4u);
}

TEST(Phase1CorrectnessContracts, table_html_rejects_hole) {
    TableRecognizer recognizer(TableRecognizerConfig{});
    const std::vector<TableCell> cells = {
        makeCell(0, 0, 1, 1, "A"),
        makeCell(0, 1, 1, 1, "B"),
        makeCell(1, 1, 1, 1, "C"), // row=1,col=0 hole
    };
    EXPECT_THROW(recognizer.generateHtml(cells), std::runtime_error);
}

TEST(Phase1CorrectnessContracts, table_html_rejects_ragged_row_width) {
    TableRecognizer recognizer(TableRecognizerConfig{});
    const std::vector<TableCell> cells = {
        makeCell(0, 0, 1, 2, "A"),
        makeCell(1, 0, 1, 1, "B"), // second row narrower than first row
    };
    EXPECT_THROW(recognizer.generateHtml(cells), std::runtime_error);
}

TEST(Phase1CorrectnessContracts, save_images_false_does_not_emit_broken_paths) {
    auto cfg = makeContractConfig();
    cfg.runtime.saveImages = false;
    DocPipeline pipeline(cfg);

    cv::Mat page(100, 100, CV_8UC3, cv::Scalar::all(255));
    std::vector<ContentElement> elems;
    DocPipelineTestAccess::saveExtractedImages(
        pipeline,
        page,
        {makeBox(LayoutCategory::FIGURE, 10, 10, 50, 50)},
        0,
        elems);
    DocPipelineTestAccess::saveFormulaImages(
        pipeline,
        page,
        {makeBox(LayoutCategory::EQUATION, 20, 20, 60, 60)},
        0,
        elems);

    ASSERT_EQ(elems.size(), 2u);
    EXPECT_TRUE(elems[0].imagePath.empty());
    EXPECT_TRUE(elems[1].imagePath.empty());

    DocumentResult doc;
    PageResult pageResult;
    pageResult.pageIndex = 0;
    pageResult.pageWidth = page.cols;
    pageResult.pageHeight = page.rows;
    pageResult.elements = elems;
    doc.pages.push_back(pageResult);

    MarkdownWriter mdWriter;
    ContentListWriter clWriter;
    std::string md = mdWriter.generate(doc);
    std::string cl = clWriter.generate(doc);

    EXPECT_EQ(md.find("![]("), std::string::npos);
    auto parsed = json::parse(cl);
    ASSERT_EQ(parsed.size(), 1u);
    ASSERT_EQ(parsed[0].size(), 2u);
    EXPECT_FALSE(parsed[0][0].contains("image_path"));
    EXPECT_FALSE(parsed[0][1].contains("image_path"));
}
