/**
 * @file rapid_doc_bench.cpp
 * @brief Phase 2 performance baseline runner for RapidDocCpp.
 */

#include "common/config.h"
#include "common/perf_utils.h"
#include "pipeline/doc_pipeline.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace rapid_doc;

namespace {

struct BenchmarkCaseDefinition {
    std::string name;
    std::string description;
    std::string inputPath;
    std::function<void(PipelineConfig&)> configure;
};

struct BenchmarkIterationResult {
    double totalTimeMs = 0.0;
    int processedPages = 0;
    DocumentStageStats stageStats;
    std::vector<double> pageTimesMs;
};

struct BenchmarkCaseResult {
    BenchmarkCaseDefinition definition;
    int warmupIterations = 0;
    std::string status = "ok";
    std::string error;
    std::vector<BenchmarkIterationResult> iterations;
};

std::string formatMs(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << value << " ms";
    return out.str();
}

json summaryToJson(const PercentileSummary& summary) {
    return json{
        {"samples", summary.sampleCount},
        {"min_ms", summary.minMs},
        {"mean_ms", summary.meanMs},
        {"max_ms", summary.maxMs},
        {"p50_ms", summary.p50Ms},
        {"p95_ms", summary.p95Ms},
    };
}

json stageStatsToJson(const DocumentStageStats& stats) {
    return json{
        {"pdf_render_ms", stats.pdfRenderTimeMs},
        {"layout_ms", stats.layoutTimeMs},
        {"ocr_ms", stats.ocrTimeMs},
        {"table_ms", stats.tableTimeMs},
        {"figure_ms", stats.figureTimeMs},
        {"formula_ms", stats.formulaTimeMs},
        {"unsupported_ms", stats.unsupportedTimeMs},
        {"reading_order_ms", stats.readingOrderTimeMs},
        {"output_gen_ms", stats.outputGenTimeMs},
        {"tracked_total_ms", totalTrackedStageTimeMs(stats)},
    };
}

DocumentStageStats meanStageStats(const std::vector<BenchmarkIterationResult>& iterations) {
    DocumentStageStats mean;
    if (iterations.empty()) {
        return mean;
    }

    for (const auto& iteration : iterations) {
        mean.pdfRenderTimeMs += iteration.stageStats.pdfRenderTimeMs;
        mean.layoutTimeMs += iteration.stageStats.layoutTimeMs;
        mean.ocrTimeMs += iteration.stageStats.ocrTimeMs;
        mean.tableTimeMs += iteration.stageStats.tableTimeMs;
        mean.figureTimeMs += iteration.stageStats.figureTimeMs;
        mean.formulaTimeMs += iteration.stageStats.formulaTimeMs;
        mean.unsupportedTimeMs += iteration.stageStats.unsupportedTimeMs;
        mean.readingOrderTimeMs += iteration.stageStats.readingOrderTimeMs;
        mean.outputGenTimeMs += iteration.stageStats.outputGenTimeMs;
    }

    const double denom = static_cast<double>(iterations.size());
    mean.pdfRenderTimeMs /= denom;
    mean.layoutTimeMs /= denom;
    mean.ocrTimeMs /= denom;
    mean.tableTimeMs /= denom;
    mean.figureTimeMs /= denom;
    mean.formulaTimeMs /= denom;
    mean.unsupportedTimeMs /= denom;
    mean.readingOrderTimeMs /= denom;
    mean.outputGenTimeMs /= denom;
    return mean;
}

BenchmarkIterationResult runOnce(DocPipeline& pipeline, const std::string& inputPath) {
    const auto result = pipeline.processPdf(inputPath);

    BenchmarkIterationResult iteration;
    iteration.totalTimeMs = result.totalTimeMs;
    iteration.processedPages = result.processedPages;
    iteration.stageStats = result.stats;
    iteration.pageTimesMs.reserve(result.pages.size());
    for (const auto& page : result.pages) {
        iteration.pageTimesMs.push_back(page.totalTimeMs);
    }
    return iteration;
}

BenchmarkCaseResult runBenchmarkCase(
    const BenchmarkCaseDefinition& definition,
    int warmupIterations,
    int measureIterations,
    const std::string& projectRoot,
    const std::string& outputRoot)
{
    PipelineConfig config = PipelineConfig::Default(projectRoot);
    config.runtime.outputDir = (fs::path(outputRoot) / definition.name).string();
    config.runtime.saveImages = false;
    config.runtime.saveVisualization = false;
    definition.configure(config);

    DocPipeline pipeline(config);
    if (!pipeline.initialize()) {
        throw std::runtime_error("Failed to initialize pipeline for case: " + definition.name);
    }

    BenchmarkCaseResult result;
    result.definition = definition;
    result.warmupIterations = warmupIterations;

    for (int i = 0; i < warmupIterations; ++i) {
        (void)runOnce(pipeline, definition.inputPath);
    }

    for (int i = 0; i < measureIterations; ++i) {
        result.iterations.push_back(runOnce(pipeline, definition.inputPath));
    }

    return result;
}

std::vector<BenchmarkCaseDefinition> makeCases(const std::string& projectRoot) {
    const fs::path root(projectRoot);
    return {
        {
            "single_page_pdf",
            "Single-page baseline (layout only, OCR/table disabled)",
            (root / "test_files" / "rmrb2026010601_origin.pdf").string(),
            [](PipelineConfig& cfg) {
                cfg.runtime.maxPages = 1;
                cfg.runtime.startPageId = 0;
                cfg.runtime.endPageId = 0;
                cfg.stages.enableOcr = false;
                cfg.stages.enableWiredTable = false;
                cfg.stages.enableFormula = false;
            },
        },
        {
            "multi_page_pdf",
            "Multi-page baseline (layout only, OCR/table disabled)",
            (root / "test_files" / "BVRC_Meeting_Minutes_2024-04_origin.pdf").string(),
            [](PipelineConfig& cfg) {
                cfg.stages.enableOcr = false;
                cfg.stages.enableWiredTable = false;
                cfg.stages.enableFormula = false;
            },
        },
        {
            "ocr_only",
            "Layout + OCR only on OCR-heavy document",
            (root / "test_files" / "small_ocr_origin.pdf").string(),
            [](PipelineConfig& cfg) {
                cfg.stages.enableWiredTable = false;
                cfg.stages.enableFormula = false;
                cfg.runtime.saveImages = false;
            },
        },
        {
            "table_heavy",
            "Table-heavy document (layout + table, OCR disabled)",
            (root / "test_files" / "表格0.pdf").string(),
            [](PipelineConfig& cfg) {
                cfg.stages.enableOcr = false;
                cfg.stages.enableFormula = false;
            },
        },
        {
            "layout_ocr_table_full_chain",
            "Full chain on mixed finance document (first two pages)",
            (root / "test_files" / "比亚迪财报_origin.pdf").string(),
            [](PipelineConfig& cfg) {
                cfg.runtime.maxPages = 2;
                cfg.runtime.startPageId = 0;
                cfg.runtime.endPageId = 1;
            },
        },
    };
}

json buildJsonSummary(const BenchmarkCaseResult& result) {
    if (result.status != "ok") {
        return json{
            {"name", result.definition.name},
            {"description", result.definition.description},
            {"input_path", result.definition.inputPath},
            {"warmup_iterations", result.warmupIterations},
            {"measured_iterations", result.iterations.size()},
            {"status", result.status},
            {"error", result.error},
        };
    }

    std::vector<double> documentTotals;
    std::vector<double> allPageTotals;
    for (const auto& iteration : result.iterations) {
        documentTotals.push_back(iteration.totalTimeMs);
        allPageTotals.insert(allPageTotals.end(), iteration.pageTimesMs.begin(), iteration.pageTimesMs.end());
    }

    json rawIterations = json::array();
    for (const auto& iteration : result.iterations) {
        rawIterations.push_back(json{
            {"total_time_ms", iteration.totalTimeMs},
            {"processed_pages", iteration.processedPages},
            {"page_times_ms", iteration.pageTimesMs},
            {"stage_stats", stageStatsToJson(iteration.stageStats)},
        });
    }

    return json{
        {"name", result.definition.name},
        {"description", result.definition.description},
        {"input_path", result.definition.inputPath},
        {"warmup_iterations", result.warmupIterations},
        {"measured_iterations", result.iterations.size()},
        {"status", "ok"},
        {"document_total", summaryToJson(summarizeSamples(documentTotals))},
        {"page_total", summaryToJson(summarizeSamples(allPageTotals))},
        {"mean_stage_breakdown", stageStatsToJson(meanStageStats(result.iterations))},
        {"iterations", std::move(rawIterations)},
    };
}

std::string buildHumanSummary(const json& summary) {
    std::ostringstream out;
    out << summary.value("name", "(unknown)") << "\n";
    out << "  description: " << summary.value("description", "") << "\n";
    out << "  input: " << summary.value("input_path", "") << "\n";
    if (summary.value("status", "ok") != "ok") {
        out << "  status: " << summary.value("status", "blocked") << "\n";
        out << "  error: " << summary.value("error", "(unknown)") << "\n";
        return out.str();
    }

    const auto& documentTotal = summary.at("document_total");
    const auto& pageTotal = summary.at("page_total");
    const auto& stageBreakdown = summary.at("mean_stage_breakdown");
    out << "  warmup: " << summary.value("warmup_iterations", 0)
        << ", measured_iterations: " << summary.value("measured_iterations", 0) << "\n";
    out << "  document_total: mean=" << formatMs(documentTotal.value("mean_ms", 0.0))
        << ", p50=" << formatMs(documentTotal.value("p50_ms", 0.0))
        << ", p95=" << formatMs(documentTotal.value("p95_ms", 0.0))
        << ", min=" << formatMs(documentTotal.value("min_ms", 0.0))
        << ", max=" << formatMs(documentTotal.value("max_ms", 0.0)) << "\n";
    out << "  page_total: samples=" << pageTotal.value("samples", 0)
        << ", mean=" << formatMs(pageTotal.value("mean_ms", 0.0))
        << ", p50=" << formatMs(pageTotal.value("p50_ms", 0.0))
        << ", p95=" << formatMs(pageTotal.value("p95_ms", 0.0)) << "\n";
    const auto& iterations = summary.at("iterations");
    const int pagesPerIteration = iterations.empty()
        ? 0
        : iterations.front().value("processed_pages", 0);
    out << "  pages_per_iteration: " << pagesPerIteration << "\n";
    out << "  mean_stage_breakdown:\n";
    out << "    pdf_render=" << formatMs(stageBreakdown.value("pdf_render_ms", 0.0))
        << ", layout=" << formatMs(stageBreakdown.value("layout_ms", 0.0))
        << ", ocr=" << formatMs(stageBreakdown.value("ocr_ms", 0.0))
        << ", table=" << formatMs(stageBreakdown.value("table_ms", 0.0))
        << ", figure=" << formatMs(stageBreakdown.value("figure_ms", 0.0))
        << ", formula=" << formatMs(stageBreakdown.value("formula_ms", 0.0))
        << ", unsupported=" << formatMs(stageBreakdown.value("unsupported_ms", 0.0))
        << ", reading_order=" << formatMs(stageBreakdown.value("reading_order_ms", 0.0))
        << ", output_gen=" << formatMs(stageBreakdown.value("output_gen_ms", 0.0))
        << "\n";
    return out.str();
}

const BenchmarkCaseDefinition* findCaseDefinition(
    const std::vector<BenchmarkCaseDefinition>& cases,
    const std::string& name)
{
    for (const auto& definition : cases) {
        if (definition.name == name) {
            return &definition;
        }
    }
    return nullptr;
}

int runWorkerMode(
    const std::string& caseName,
    int warmup,
    int iterations,
    const std::string& projectRoot,
    const std::string& outputDir,
    const std::string& jsonOutPath)
{
    const auto definitions = makeCases(projectRoot);
    const auto* definition = findCaseDefinition(definitions, caseName);
    if (definition == nullptr) {
        std::cerr << "Unknown worker case: " << caseName << "\n";
        return 1;
    }

    BenchmarkCaseResult result;
    try {
        result = runBenchmarkCase(*definition, warmup, iterations, projectRoot, outputDir);
    } catch (const std::exception& e) {
        result.definition = *definition;
        result.warmupIterations = warmup;
        result.status = "blocked";
        result.error = e.what();
    }

    const fs::path outputPath(jsonOutPath);
    if (!outputPath.parent_path().empty()) {
        fs::create_directories(outputPath.parent_path());
    }
    std::ofstream out(outputPath);
    out << buildJsonSummary(result).dump(2);
    return 0;
}

json runCaseInSubprocess(
    const std::string& binaryPath,
    const BenchmarkCaseDefinition& definition,
    int warmup,
    int iterations,
    const std::string& projectRoot,
    const std::string& outputDir)
{
    const fs::path workerJsonPath =
        fs::path(outputDir) / (definition.name + "_worker_summary.json");
    if (!workerJsonPath.parent_path().empty()) {
        fs::create_directories(workerJsonPath.parent_path());
    }
    fs::remove(workerJsonPath);

    std::vector<std::string> args = {
        binaryPath,
        "--worker-case", definition.name,
        "--worker-json", workerJsonPath.string(),
        "--iterations", std::to_string(iterations),
        "--warmup", std::to_string(warmup),
        "--output-dir", outputDir,
        "--project-root", projectRoot,
    };

    pid_t pid = fork();
    if (pid == 0) {
        std::vector<char*> argvExec;
        argvExec.reserve(args.size() + 1);
        for (auto& arg : args) {
            argvExec.push_back(arg.data());
        }
        argvExec.push_back(nullptr);
        execv(binaryPath.c_str(), argvExec.data());
        _exit(127);
    }
    if (pid < 0) {
        return json{
            {"name", definition.name},
            {"description", definition.description},
            {"input_path", definition.inputPath},
            {"warmup_iterations", warmup},
            {"measured_iterations", 0},
            {"status", "blocked"},
            {"error", "fork_failed"},
        };
    }

    int status = 0;
    waitpid(pid, &status, 0);

    if (WIFEXITED(status) && WEXITSTATUS(status) == 0 && fs::exists(workerJsonPath)) {
        std::ifstream in(workerJsonPath);
        return json::parse(in, nullptr, false);
    }

    std::ostringstream error;
    if (WIFSIGNALED(status)) {
        error << "worker_killed_by_signal_" << WTERMSIG(status);
    } else if (WIFEXITED(status)) {
        error << "worker_exit_code_" << WEXITSTATUS(status);
    } else {
        error << "worker_failed_without_status";
    }

    return json{
        {"name", definition.name},
        {"description", definition.description},
        {"input_path", definition.inputPath},
        {"warmup_iterations", warmup},
        {"measured_iterations", 0},
        {"status", "blocked"},
        {"error", error.str()},
    };
}

void printUsage(const char* program) {
    std::cout << "RapidDoc Phase 2 Benchmark\n\n";
    std::cout << "Usage: " << program << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --iterations <n>   Measured iterations per case (default: 5)\n";
    std::cout << "  --warmup <n>       Warmup iterations per case (default: 1)\n";
    std::cout << "  --case <name>      Run only the named case (repeatable)\n";
    std::cout << "  --json-out <path>  Write JSON summary to file\n";
    std::cout << "  --output-dir <p>   Benchmark scratch output dir (default: ./output-benchmark)\n";
    std::cout << "  --project-root <p> Override project root (internal/debug)\n";
    std::cout << "  --help             Show this help\n";
}

} // namespace

int main(int argc, char** argv) {
    std::string projectRoot = PROJECT_ROOT_DIR;
    int iterations = 5;
    int warmup = 1;
    std::string jsonOutPath;
    std::string outputDir = (fs::path(projectRoot) / "output-benchmark").string();
    std::string workerCaseName;
    std::string workerJsonPath;
    std::set<std::string> requestedCases;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::max(0, std::atoi(argv[++i]));
        } else if (arg == "--case" && i + 1 < argc) {
            requestedCases.insert(argv[++i]);
        } else if (arg == "--json-out" && i + 1 < argc) {
            jsonOutPath = argv[++i];
        } else if (arg == "--output-dir" && i + 1 < argc) {
            outputDir = argv[++i];
        } else if (arg == "--project-root" && i + 1 < argc) {
            projectRoot = argv[++i];
        } else if (arg == "--worker-case" && i + 1 < argc) {
            workerCaseName = argv[++i];
        } else if (arg == "--worker-json" && i + 1 < argc) {
            workerJsonPath = argv[++i];
        } else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    spdlog::set_level(spdlog::level::err);

    if (!workerCaseName.empty()) {
        if (workerJsonPath.empty()) {
            std::cerr << "--worker-json is required with --worker-case\n";
            return 1;
        }
        return runWorkerMode(
            workerCaseName, warmup, iterations, projectRoot, outputDir, workerJsonPath);
    }

    std::vector<BenchmarkCaseDefinition> selectedCases;
    for (const auto& definition : makeCases(projectRoot)) {
        if (requestedCases.empty() || requestedCases.count(definition.name) > 0) {
            selectedCases.push_back(definition);
        }
    }

    if (!requestedCases.empty() && selectedCases.size() != requestedCases.size()) {
        std::cerr << "Unknown benchmark case requested.\n";
        return 1;
    }

    json root = json::object();
    root["generated_at_utc_ms"] =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    root["iterations"] = iterations;
    root["warmup"] = warmup;
    root["cases"] = json::array();
    const std::string binaryPath = fs::absolute(argv[0]).string();

    for (const auto& definition : selectedCases) {
        if (!fs::exists(definition.inputPath)) {
            std::cerr << "Missing input fixture for case " << definition.name
                      << ": " << definition.inputPath << "\n";
            return 1;
        }

        const json summary = runCaseInSubprocess(
            binaryPath, definition, warmup, iterations, projectRoot, outputDir);
        std::cout << buildHumanSummary(summary) << "\n";
        root["cases"].push_back(summary);
    }

    if (!jsonOutPath.empty()) {
        const fs::path parent = fs::path(jsonOutPath).parent_path();
        if (!parent.empty()) {
            fs::create_directories(parent);
        }
        std::ofstream out(jsonOutPath);
        out << root.dump(2);
        std::cout << "Saved JSON benchmark summary: " << jsonOutPath << "\n";
    }

    return 0;
}
