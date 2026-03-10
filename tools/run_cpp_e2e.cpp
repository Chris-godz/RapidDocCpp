/**
 * @file run_cpp_e2e.cpp
 * @brief Run the full C++ pipeline on all test PDFs and save per-PDF results.
 *
 * Output structure per PDF:
 *   {output_dir}/{pdf_stem}/{pdf_stem}.md, content_list.json, layout_page_*.json, summary.json
 *
 * Usage:
 *   ./run_cpp_e2e [--pdf-dir test_files] [--output test/fixtures/e2e/cpp]
 * Project root is set at build time (CMake PROJECT_ROOT_DIR); use --project-root to override.
 */

#include "pipeline/doc_pipeline.h"
#include "common/config.h"
#include "output/markdown_writer.h"
#include "output/content_list.h"

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace rapid_doc;

static std::string layoutCatStr(LayoutCategory c) {
    return layoutCategoryToString(c);
}

static void savePageLayouts(const DocumentResult& result, const std::string& outDir) {
    for (const auto& page : result.pages) {
        json arr = json::array();
        for (const auto& box : page.layoutResult.boxes) {
            arr.push_back({
                {"cls_id", static_cast<int>(box.category)},
                {"label", layoutCatStr(box.category)},
                {"score", box.confidence},
                {"coordinate", {box.x0, box.y0, box.x1, box.y1}},
            });
        }
        std::string path = outDir + "/layout_page_" + std::to_string(page.pageIndex) + ".json";
        std::ofstream f(path);
        f << arr.dump(2);
    }
}

static void saveContentList(const DocumentResult& result, const std::string& outDir) {
    ContentListWriter writer;
    std::string clJson = writer.generate(result);
    std::ofstream f(outDir + "/content_list.json");
    f << clJson;
}

static json makeSummary(const DocumentResult& result, const std::string& pdfName, double totalMs) {
    json summary;
    summary["pdf"] = pdfName;
    summary["stem"] = fs::path(pdfName).stem().string();
    summary["total_pages"] = result.processedPages;
    summary["total_time_ms"] = totalMs;
    summary["markdown_chars"] = static_cast<int>(result.markdown.size());
    summary["skipped_elements"] = result.skippedElements;

    json pages = json::array();
    for (const auto& page : result.pages) {
        int textElems = 0;
        for (const auto& e : page.elements)
            if (!e.text.empty()) textElems++;
        pages.push_back({
            {"index", page.pageIndex},
            {"layout_boxes", static_cast<int>(page.layoutResult.boxes.size())},
            {"text_elements", textElems},
            {"time_ms", page.totalTimeMs},
        });
    }
    summary["pages"] = pages;
    return summary;
}

// Project root from CMake at build time (no need for --project-root in normal use)
#define STRINGIFY_HELPER(x) #x
#define STRINGIFY(x) STRINGIFY_HELPER(x)
#ifdef PROJECT_ROOT_DIR
static std::string getDefaultProjectRoot() {
    std::string s = STRINGIFY(PROJECT_ROOT_DIR);
    // Strip surrounding double quotes if CMake passed -DPROJECT_ROOT_DIR="/path"
    if (s.size() >= 2 && s.front() == '"' && s.back() == '"')
        return s.substr(1, s.size() - 2);
    return s;
}
#else
static std::string getDefaultProjectRoot() { return "."; }
#endif

int main(int argc, char** argv) {
    std::string projectRoot = getDefaultProjectRoot();
    std::string pdfDir = "test_files";
    std::string outputDir = "test/fixtures/e2e/cpp";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--project-root" && i + 1 < argc)
            projectRoot = argv[++i];
        else if (arg == "--pdf-dir" && i + 1 < argc)
            pdfDir = argv[++i];
        else if (arg == "--output" && i + 1 < argc)
            outputDir = argv[++i];
    }

    // Resolve relative paths against project root so CWD doesn't matter
    if (!fs::path(pdfDir).is_absolute())
        pdfDir = (fs::path(projectRoot) / pdfDir).lexically_normal().string();
    if (!fs::path(outputDir).is_absolute())
        outputDir = (fs::path(projectRoot) / outputDir).lexically_normal().string();

    auto cfg = PipelineConfig::Default(projectRoot);
    cfg.stages.enablePdfRender = true;
    cfg.stages.enableWiredTable = true;
    cfg.runtime.maxPages = 0;  // 0 = all pages, matching Python
    cfg.runtime.saveImages = true;
    cfg.runtime.outputDir = outputDir;

    std::cerr << "Initializing pipeline..." << std::endl;
    DocPipeline pipeline(cfg);
    if (!pipeline.initialize()) {
        std::cerr << "ERROR: Pipeline initialization failed" << std::endl;
        return 1;
    }

    std::vector<fs::path> pdfs;
    for (auto& entry : fs::directory_iterator(pdfDir)) {
        if (entry.path().extension() == ".pdf")
            pdfs.push_back(entry.path());
    }
    std::sort(pdfs.begin(), pdfs.end());
    std::cerr << "Found " << pdfs.size() << " PDFs in " << pdfDir << std::endl;

    json allSummaries = json::array();

    for (const auto& pdf : pdfs) {
        std::string stem = pdf.stem().string();
        std::string outPath = outputDir + "/" + stem;
        fs::create_directories(outPath);

        // Set per-PDF output dir so images save to the right place
        pipeline.setOutputDir(outPath);

        std::cerr << "\n============================================" << std::endl;
        std::cerr << "Processing: " << pdf.filename().string() << std::endl;
        std::cerr << "============================================" << std::endl;

        auto t0 = std::chrono::steady_clock::now();

        try {
            auto result = pipeline.processPdf(pdf.string());
            auto t1 = std::chrono::steady_clock::now();
            double totalMs = std::chrono::duration<double, std::milli>(t1 - t0).count();

            // Save markdown
            std::ofstream mdFile(outPath + "/" + stem + ".md");
            mdFile << result.markdown;

            // Save layout per page
            savePageLayouts(result, outPath);

            // Save content list
            saveContentList(result, outPath);

            // Save summary
            json summary = makeSummary(result, pdf.filename().string(), totalMs);
            std::ofstream sumFile(outPath + "/summary.json");
            sumFile << summary.dump(2);

            allSummaries.push_back(summary);

            std::cerr << "  Pages: " << result.processedPages
                      << ", MD: " << result.markdown.size() << " chars"
                      << ", Time: " << totalMs << " ms" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "  ERROR: " << e.what() << std::endl;
            allSummaries.push_back({{"pdf", pdf.filename().string()}, {"error", e.what()}});
        }
    }

    std::ofstream allFile(outputDir + "/all_summaries.json");
    allFile << allSummaries.dump(2);

    std::cerr << "\nDone! " << allSummaries.size() << " PDFs processed" << std::endl;
    std::cerr << "Results saved to: " << outputDir << std::endl;
    return 0;
}
