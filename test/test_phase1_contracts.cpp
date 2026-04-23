#include <gtest/gtest.h>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <deque>
#include <filesystem>
#include <fstream>
#include <cstdlib>
#include <string>
#include <tuple>

#include "common/config.h"
#include "formula/formula_recognizer.h"
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

std::string readFileBytes(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return {};
    }
    return std::string(
        (std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());
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

// When the wireless backend is enabled AND the hook returns a pre-baked
// wireless HTML, the pipeline must promote result.html into ContentElement.html
// instead of running generateTableHtml(cells) (the latter would throw on empty
// cells). This pins the Phase D invariant: elem.html prefers result.html.
TEST(Phase1CorrectnessContracts, wireless_result_html_is_promoted_to_content) {
    auto cfg = makeContractConfig();
    cfg.stages.enableWiredTable = true;
    cfg.stages.enableWirelessTable = false;  // no real SLANet+ needed for the hook path
    cfg.stages.enableOcr = false;
    DocPipeline pipeline(cfg);

    const std::string kFakeHtml = "<html><body><table><tr><td>stub</td></tr></table></body></html>";

    DocPipelineTestAccess::setTableHooks(pipeline, [&](const cv::Mat&) {
        TableResult result;
        result.type = TableType::WIRELESS;
        result.supported = true;
        result.html = kFakeHtml;
        result.cells.clear();  // no cells — pure HTML from wireless backend
        return result;
    });

    cv::Mat page(80, 120, CV_8UC3, cv::Scalar::all(255));
    auto elems = DocPipelineTestAccess::runTableRecognition(
        pipeline,
        page,
        {makeBox(LayoutCategory::TABLE, 5, 5, 70, 60)},
        2);

    ASSERT_EQ(elems.size(), 1u);
    const auto& elem = elems.front();
    EXPECT_FALSE(elem.skipped);
    EXPECT_EQ(elem.type, ContentElement::Type::TABLE);
    EXPECT_EQ(elem.html, kFakeHtml);
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

TEST(Phase1CorrectnessContracts, formula_decode_normalize_fixture) {
    const std::vector<std::string> idToToken = {
        "<s>", "<pad>", "</s>", "x", "Ġ+", "Ġ1",
    };
    const std::vector<int64_t> tokenIds = {0, 3, 4, 5, 2, 1};
    const std::string latex = FormulaRecognizer::decodeTokensForTest(tokenIds, idToToken);
    EXPECT_EQ(latex, "x+ 1");
}

TEST(Phase1CorrectnessContracts, formula_elements_prefer_latex_over_image_fallback) {
    auto cfg = makeContractConfig();
    cfg.runtime.saveImages = true;
    DocPipeline pipeline(cfg);

    DocPipelineTestAccess::setFormulaHook(
        pipeline,
        [](const std::vector<cv::Mat>& crops) {
            std::vector<std::string> outputs(crops.size());
            for (size_t i = 0; i < crops.size(); ++i) {
                outputs[i] = "\\frac{a}{b}";
            }
            return outputs;
        });

    cv::Mat page(120, 140, CV_8UC3, cv::Scalar::all(255));
    auto elems = DocPipelineTestAccess::runFormulaRecognition(
        pipeline,
        page,
        {makeBox(LayoutCategory::EQUATION, 20, 20, 100, 80)},
        2);

    ASSERT_EQ(elems.size(), 1u);
    ASSERT_EQ(elems[0].type, ContentElement::Type::EQUATION);
    EXPECT_EQ(elems[0].text, "\\frac{a}{b}");
    EXPECT_TRUE(elems[0].imagePath.empty());

    DocumentResult doc;
    PageResult pageResult;
    pageResult.pageIndex = 2;
    pageResult.pageWidth = page.cols;
    pageResult.pageHeight = page.rows;
    pageResult.elements = elems;
    doc.pages.push_back(pageResult);

    MarkdownWriter mdWriter;
    const std::string md = mdWriter.generate(doc);
    EXPECT_NE(md.find("\\frac{a}{b}"), std::string::npos);
    EXPECT_EQ(md.find("![]("), std::string::npos);
}

TEST(Phase1CorrectnessContracts, formula_trace_sidecar_contains_expected_fields) {
    constexpr const char* kTraceEnv = "RAPIDDOC_FORMULA_TRACE";
    const char* previousRaw = std::getenv(kTraceEnv);
    const std::string previousValue = previousRaw == nullptr ? std::string() : std::string(previousRaw);
    ASSERT_EQ(::setenv(kTraceEnv, "1", 1), 0);

    auto cfg = makeContractConfig();
    cfg.stages.enableFormula = true;
    cfg.runtime.saveImages = false;
    DocPipeline pipeline(cfg);

    DocPipelineTestAccess::setFormulaHook(
        pipeline,
        [](const std::vector<cv::Mat>& crops) {
            std::vector<std::string> outputs(crops.size());
            for (size_t i = 0; i < crops.size(); ++i) {
                outputs[i] = (i % 2 == 0) ? "\\alpha+\\beta" : "x^2";
            }
            return outputs;
        });

    PageImage pageImage;
    pageImage.image = cv::Mat(120, 180, CV_8UC3, cv::Scalar::all(255));
    pageImage.pageIndex = 0;

    PageResult pageResult;
    pageResult.pageIndex = 0;
    pageResult.pageWidth = pageImage.image.cols;
    pageResult.pageHeight = pageImage.image.rows;
    pageResult.layoutResult.boxes = {
        makeBox(LayoutCategory::EQUATION, 10, 12, 90, 40),
        makeBox(LayoutCategory::INTERLINE_EQUATION, 20, 56, 120, 96),
    };

    DocumentResult doc;
    doc.pages.push_back(pageResult);

    DocPipelineTestAccess::runDocumentFormulaStage(pipeline, {pageImage}, doc);

    ASSERT_FALSE(doc.formulaTraceJson.empty());
    const json trace = json::parse(doc.formulaTraceJson);
    ASSERT_TRUE(trace.contains("layout_raw_boxes"));
    ASSERT_TRUE(trace.contains("layout_prefilter_boxes"));
    ASSERT_TRUE(trace.contains("layout_boxes"));
    ASSERT_TRUE(trace.contains("raw_candidates"));
    ASSERT_TRUE(trace.contains("candidate_refinements"));
    ASSERT_TRUE(trace.contains("filtered_candidates"));
    ASSERT_TRUE(trace.contains("crop_mappings"));
    ASSERT_TRUE(trace.contains("crop_outputs"));
    ASSERT_TRUE(trace.contains("batch_profile"));
    ASSERT_TRUE(trace.contains("timing_bill"));
    EXPECT_EQ(trace.at("layout_raw_boxes").size(), 0u);
    EXPECT_EQ(trace.at("layout_prefilter_boxes").size(), 0u);
    EXPECT_EQ(trace.at("raw_candidates").size(), 2u);
    EXPECT_EQ(trace.at("layout_boxes").size(), 2u);
    EXPECT_EQ(trace.at("crop_mappings").size(), 2u);
    EXPECT_EQ(trace.at("crop_outputs").size(), 2u);
    EXPECT_EQ(trace.at("crop_outputs").at(0).value("latex", ""), "\\alpha+\\beta");
    EXPECT_EQ(trace.at("batch_profile").value("configured_batch_size", 0), 8);
    EXPECT_EQ(trace.at("batch_profile").value("max_effective_batch_size", 0), 2);
    ASSERT_TRUE(trace.at("batch_profile").contains("batches"));
    ASSERT_EQ(trace.at("batch_profile").at("batches").size(), 1u);
    EXPECT_TRUE(trace.at("timing_bill").contains("infer_ms"));

    if (previousRaw == nullptr) {
        ::unsetenv(kTraceEnv);
    } else {
        ASSERT_EQ(::setenv(kTraceEnv, previousValue.c_str(), 1), 0);
    }
}

TEST(Phase1CorrectnessContracts, formula_trace_records_conservative_fragment_merge) {
    constexpr const char* kTraceEnv = "RAPIDDOC_FORMULA_TRACE";
    const char* previousRaw = std::getenv(kTraceEnv);
    const std::string previousValue = previousRaw == nullptr ? std::string() : std::string(previousRaw);
    ASSERT_EQ(::setenv(kTraceEnv, "1", 1), 0);

    auto cfg = makeContractConfig();
    cfg.stages.enableFormula = true;
    DocPipeline pipeline(cfg);

    DocPipelineTestAccess::setFormulaHook(
        pipeline,
        [](const std::vector<cv::Mat>& crops) {
            std::vector<std::string> outputs(crops.size());
            for (size_t i = 0; i < crops.size(); ++i) {
                outputs[i] = "latex_" + std::to_string(i);
            }
            return outputs;
        });

    PageImage pageImage;
    pageImage.image = cv::Mat(180, 900, CV_8UC3, cv::Scalar::all(255));
    pageImage.pageIndex = 0;

    PageResult pageResult;
    pageResult.pageIndex = 0;
    pageResult.pageWidth = pageImage.image.cols;
    pageResult.pageHeight = pageImage.image.rows;
    pageResult.layoutResult.boxes = {
        makeBox(LayoutCategory::EQUATION, 10, 40, 140, 70),
        makeBox(LayoutCategory::EQUATION, 160, 42, 290, 72),
        makeBox(LayoutCategory::EQUATION, 10, 120, 80, 150),
    };

    DocumentResult doc;
    doc.pages.push_back(pageResult);

    DocPipelineTestAccess::runDocumentFormulaStage(pipeline, {pageImage}, doc);

    const json trace = json::parse(doc.formulaTraceJson);
    EXPECT_EQ(trace.at("summary").value("raw_candidate_count", 0), 3);
    EXPECT_EQ(trace.at("summary").value("refined_candidate_count", 0), 2);
    EXPECT_EQ(trace.at("summary").value("kept_candidate_count", 0), 2);
    ASSERT_EQ(trace.at("candidate_refinements").size(), 1u);
    EXPECT_EQ(
        trace.at("candidate_refinements").at(0).value("action", ""),
        "merge_same_line_fragments");
    EXPECT_EQ(trace.at("crop_outputs").size(), 2u);

    if (previousRaw == nullptr) {
        ::unsetenv(kTraceEnv);
    } else {
        ASSERT_EQ(::setenv(kTraceEnv, previousValue.c_str(), 1), 0);
    }
}

TEST(Phase1CorrectnessContracts, request_local_overrides_do_not_mutate_shared_pipeline_config) {
    auto cfg = makeContractConfig();
    cfg.stages.enablePdfRender = true;
    cfg.stages.enableLayout = false;
    cfg.stages.enableOcr = false;
    cfg.stages.enableWiredTable = false;
    cfg.stages.enableFormula = false;
    cfg.stages.enableReadingOrder = false;
    cfg.stages.enableMarkdownOutput = true;
    cfg.runtime.maxPages = 0;
    cfg.runtime.startPageId = 0;
    cfg.runtime.endPageId = -1;
    cfg.runtime.outputDir = std::string(PROJECT_ROOT_DIR) + "/test/fixtures/contract_output/default";
    cfg.runtime.saveImages = false;
    cfg.runtime.saveVisualization = false;

    DocPipeline pipeline(cfg);
    ASSERT_TRUE(pipeline.initialize());

    const std::filesystem::path pdfPath =
        std::filesystem::path(PROJECT_ROOT_DIR) / "test_files" / "small_ocr_origin.pdf";
    if (!std::filesystem::exists(pdfPath)) {
        GTEST_SKIP() << "Missing fixture PDF: " << pdfPath;
    }

    const std::string bytes = readFileBytes(pdfPath);
    ASSERT_FALSE(bytes.empty());

    const PipelineConfig before = pipeline.config();

    PipelineRunOverrides overrides;
    overrides.outputDir = std::string(PROJECT_ROOT_DIR) + "/test/fixtures/contract_output/override";
    overrides.saveImages = true;
    overrides.saveVisualization = true;
    overrides.startPageId = 0;
    overrides.endPageId = 0;
    overrides.maxPages = 1;
    overrides.enableWiredTable = false;
    overrides.enableFormula = false;
    overrides.enableMarkdownOutput = false;

    const DocumentResult result = pipeline.processPdfFromMemoryWithOverrides(
        reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), overrides);
    ASSERT_EQ(result.processedPages, 1);
    ASSERT_EQ(result.totalPages, 1);

    const PipelineConfig after = pipeline.config();
    EXPECT_EQ(after.runtime.outputDir, before.runtime.outputDir);
    EXPECT_EQ(after.runtime.saveImages, before.runtime.saveImages);
    EXPECT_EQ(after.runtime.saveVisualization, before.runtime.saveVisualization);
    EXPECT_EQ(after.runtime.startPageId, before.runtime.startPageId);
    EXPECT_EQ(after.runtime.endPageId, before.runtime.endPageId);
    EXPECT_EQ(after.runtime.maxPages, before.runtime.maxPages);
    EXPECT_EQ(after.stages.enableFormula, before.stages.enableFormula);
    EXPECT_EQ(after.stages.enableWiredTable, before.stages.enableWiredTable);
    EXPECT_EQ(after.stages.enableMarkdownOutput, before.stages.enableMarkdownOutput);
}
