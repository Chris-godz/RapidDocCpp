/**
 * @file main.cpp
 * @brief CLI entry point for RapidDoc C++
 * 
 * Usage:
 *   rapid_doc_cli --input <pdf_path> --output <dir> [options]
 * 
 * Options:
 *   --input, -i     Input PDF file path
 *   --output, -o    Output directory (default: ./output)
 *   --dpi           PDF rendering DPI (default: 200)
 *   --max-pages     Max pages to process (0 = all)
 *   --no-table      Disable table recognition
 *   --no-ocr        Disable OCR
 *   --json-only     Output JSON content list only (no Markdown)
 *   --verbose, -v   Verbose logging
 */

#include "pipeline/doc_pipeline.h"
#include "common/config.h"
#include "common/logger.h"
#include "formula/formula_recognizer.h"
#include "output/detail_report.h"
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>
#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <vector>
#include <getopt.h>

namespace fs = std::filesystem;
using json = nlohmann::json;

bool envFlagOrDefault(const char* key, bool defaultValue) {
    const char* raw = std::getenv(key);
    if (raw == nullptr) {
        return defaultValue;
    }
    std::string value(raw);
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
    }
    return defaultValue;
}

int envIntOrDefault(const char* key, int defaultValue) {
    const char* raw = std::getenv(key);
    if (raw == nullptr || *raw == '\0') {
        return defaultValue;
    }
    try {
        return std::stoi(raw);
    } catch (...) {
        return defaultValue;
    }
}

json formulaBatchRecordsToJson(const rapid_doc::FormulaRecognizer::BatchTiming& timing) {
    json batches = json::array();
    for (const auto& batch : timing.batches) {
        batches.push_back({
            {"batch_size", batch.batchSize},
            {"target_h", batch.targetH},
            {"target_w", batch.targetW},
            {"infer_ms", batch.inferMs},
            {"crop_indices", batch.cropIndices},
        });
    }
    return batches;
}

void printUsage(const char* programName) {
    std::cout << "RapidDoc C++ - Document Analysis Pipeline (DEEPX NPU)\n\n";
    std::cout << "Usage: " << programName << " -i <pdf_path> -o <dir> [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -i, --input <path>      Input PDF file path (required)\n";
    std::cout << "  -o, --output <dir>      Output directory (default: ./output)\n";
    std::cout << "  -d, --dpi <num>         PDF rendering DPI (default: 200)\n";
    std::cout << "  -m, --max-pages <num>   Max pages to process (0 = all)\n";
    std::cout << "      --no-table          Disable table recognition\n";
    std::cout << "      --no-ocr            Disable OCR\n";
    std::cout << "      --json-only         Output JSON only (no Markdown)\n";
    std::cout << "      --detail            Print and save a human-readable detail report\n";
    std::cout << "      --detail-file <p>   Override detail report output path\n";
    std::cout << "  -v, --verbose           Verbose logging\n";
    std::cout << "  -h, --help              Show this help\n";
    std::cout << "\n";
    std::cout << "Note: Formula recognition runs through ONNX Runtime when enabled.\n";
    std::cout << "      Wireless table recognition is still unsupported on DEEPX NPU.\n";
    std::cout << "      Set RAPIDDOC_FORMULA_TRACE=1 to emit formula trace sidecar JSON.\n";
    std::cout << "      Set RAPIDDOC_FORMULA_DUMP_CROPS_DIR=<dir> to save formula input crops (cpp_infer_order_*.png).\n";
}

struct CliArgs {
    std::string inputPath;
    std::string outputDir = "./output";
    int dpi = 200;
    int maxPages = 0;
    bool enableTable = true;
    bool enableOcr = true;
    bool jsonOnly = false;
    bool detail = false;
    std::string detailPath;
    bool verbose = false;
    bool showHelp = false;
    std::string formulaReplayCropDir;
    std::string formulaReplayJsonOut;
    int formulaReplayBatchSize = 1;

    bool isFormulaReplay() const {
        return !formulaReplayCropDir.empty() || !formulaReplayJsonOut.empty();
    }
};

enum LongOnlyOpt {
    OPT_NO_TABLE = 256,
    OPT_NO_OCR,
    OPT_JSON_ONLY,
    OPT_DETAIL,
    OPT_DETAIL_FILE,
    OPT_FORMULA_REPLAY_CROP_DIR,
    OPT_FORMULA_REPLAY_JSON_OUT,
    OPT_FORMULA_REPLAY_BATCH_SIZE,
};

bool parseArgs(int argc, char* argv[], CliArgs& args) {
    static const struct option longOpts[] = {
        {"input",     required_argument, nullptr, 'i'},
        {"output",    required_argument, nullptr, 'o'},
        {"dpi",       required_argument, nullptr, 'd'},
        {"max-pages", required_argument, nullptr, 'm'},
        {"no-table",  no_argument,       nullptr, OPT_NO_TABLE},
        {"no-ocr",    no_argument,       nullptr, OPT_NO_OCR},
        {"json-only", no_argument,       nullptr, OPT_JSON_ONLY},
        {"detail",    no_argument,       nullptr, OPT_DETAIL},
        {"detail-file", required_argument, nullptr, OPT_DETAIL_FILE},
        {"formula-replay-crop-dir", required_argument, nullptr, OPT_FORMULA_REPLAY_CROP_DIR},
        {"formula-replay-json-out", required_argument, nullptr, OPT_FORMULA_REPLAY_JSON_OUT},
        {"formula-replay-batch-size", required_argument, nullptr, OPT_FORMULA_REPLAY_BATCH_SIZE},
        {"verbose",   no_argument,       nullptr, 'v'},
        {"help",      no_argument,       nullptr, 'h'},
        {nullptr,     0,                 nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "i:o:d:m:vh", longOpts, nullptr)) != -1) {
        switch (opt) {
            case 'i': args.inputPath = optarg; break;
            case 'o': args.outputDir = optarg; break;
            case 'd': args.dpi = std::atoi(optarg); break;
            case 'm': args.maxPages = std::atoi(optarg); break;
            case OPT_NO_TABLE: args.enableTable = false; break;
            case OPT_NO_OCR:   args.enableOcr = false; break;
            case OPT_JSON_ONLY: args.jsonOnly = true; break;
            case OPT_DETAIL: args.detail = true; break;
            case OPT_DETAIL_FILE: args.detailPath = optarg; break;
            case OPT_FORMULA_REPLAY_CROP_DIR: args.formulaReplayCropDir = optarg; break;
            case OPT_FORMULA_REPLAY_JSON_OUT: args.formulaReplayJsonOut = optarg; break;
            case OPT_FORMULA_REPLAY_BATCH_SIZE: args.formulaReplayBatchSize = std::atoi(optarg); break;
            case 'v': args.verbose = true; break;
            case 'h':
                printUsage(argv[0]);
                args.showHelp = true;
                return false;
            default:  printUsage(argv[0]); return false;
        }
    }

    if (args.isFormulaReplay()) {
        if (args.formulaReplayCropDir.empty()) {
            std::cerr << "Error: --formula-replay-crop-dir is required for formula replay\n";
            return false;
        }
        if (args.formulaReplayJsonOut.empty()) {
            std::cerr << "Error: --formula-replay-json-out is required for formula replay\n";
            return false;
        }
        if (args.formulaReplayBatchSize <= 0) {
            std::cerr << "Error: --formula-replay-batch-size must be positive\n";
            return false;
        }
        return true;
    }

    if (args.inputPath.empty()) {
        std::cerr << "Error: --input is required\n";
        printUsage(argv[0]);
        return false;
    }

    return true;
}

bool hasImageExtension(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return ext == ".png" || ext == ".jpg" || ext == ".jpeg" ||
           ext == ".bmp" || ext == ".webp";
}

std::string resolveFormulaModelPath() {
    const char* explicitPath = std::getenv("RAPIDDOC_FORMULA_MODEL");
    if (explicitPath != nullptr && explicitPath[0] != '\0') {
        return explicitPath;
    }
    const char* explicitOnnxPath = std::getenv("RAPIDDOC_FORMULA_ONNX_MODEL");
    if (explicitOnnxPath != nullptr && explicitOnnxPath[0] != '\0') {
        return explicitOnnxPath;
    }
    return std::string(PROJECT_ROOT_DIR) +
           "/.download_cache/onnx_models/pp_formulanet_plus_m.onnx";
}

int runFormulaReplay(const CliArgs& args) {
    if (!fs::exists(args.formulaReplayCropDir) ||
        !fs::is_directory(args.formulaReplayCropDir)) {
        LOG_ERROR("Formula replay crop directory not found: {}", args.formulaReplayCropDir);
        return 1;
    }

    std::vector<fs::path> cropPaths;
    for (const auto& entry : fs::directory_iterator(args.formulaReplayCropDir)) {
        if (entry.is_regular_file() && hasImageExtension(entry.path())) {
            cropPaths.push_back(entry.path());
        }
    }
    std::sort(cropPaths.begin(), cropPaths.end());
    if (cropPaths.empty()) {
        LOG_ERROR("Formula replay crop directory contains no supported images: {}",
                  args.formulaReplayCropDir);
        return 1;
    }

    std::vector<cv::Mat> crops;
    crops.reserve(cropPaths.size());
    json cropInputs = json::array();
    for (size_t i = 0; i < cropPaths.size(); ++i) {
        cv::Mat image = cv::imread(cropPaths[i].string(), cv::IMREAD_COLOR);
        if (image.empty()) {
            LOG_ERROR("Failed to read formula replay crop: {}", cropPaths[i].string());
            return 1;
        }
        cropInputs.push_back({
            {"index", i},
            {"filename", cropPaths[i].filename().string()},
            {"path", fs::absolute(cropPaths[i]).string()},
            {"width", image.cols},
            {"height", image.rows},
        });
        crops.push_back(std::move(image));
    }

    const std::string modelPath = resolveFormulaModelPath();
    rapid_doc::FormulaRecognizerConfig config;
    config.onnxModelPath = modelPath;
    config.inputSize = 384;
    config.enableCpuMemArena =
        envFlagOrDefault("RAPIDDOC_FORMULA_CPU_MEM_ARENA", true);
    config.maxBatchSize = args.formulaReplayBatchSize;
    config.packDynamicShapes =
        envFlagOrDefault("RAPIDDOC_FORMULA_PACK_DYNAMIC_SHAPES", true);
    const char* ortProfileRaw = std::getenv("RAPIDDOC_FORMULA_ORT_PROFILE");
    const std::string ortProfile = ortProfileRaw == nullptr ? "" : std::string(ortProfileRaw);
    if (ortProfile == "constrained") {
        config.sequentialExecution = true;
        config.intraOpThreads = 1;
        config.interOpThreads = 1;
    } else if (ortProfile == "sequential") {
        config.sequentialExecution = true;
    }
    const int intraThreads = envIntOrDefault("RAPIDDOC_FORMULA_ORT_INTRA_THREADS", -1);
    const int interThreads = envIntOrDefault("RAPIDDOC_FORMULA_ORT_INTER_THREADS", -1);
    if (intraThreads > 0) {
        config.intraOpThreads = intraThreads;
    }
    if (interThreads > 0) {
        config.interOpThreads = interThreads;
    }

    rapid_doc::FormulaRecognizer recognizer(config);
    if (!recognizer.initialize()) {
        LOG_ERROR("Failed to initialize formula recognizer for replay");
        return 1;
    }

    rapid_doc::FormulaRecognizer::BatchTiming timing;
    const auto replayStart = std::chrono::steady_clock::now();
    const std::vector<std::string> latexes = recognizer.recognizeBatch(crops, &timing);
    const auto replayEnd = std::chrono::steady_clock::now();

    json outputs = json::array();
    for (size_t i = 0; i < latexes.size(); ++i) {
        outputs.push_back({
            {"index", i},
            {"filename", cropPaths[i].filename().string()},
            {"latex", latexes[i]},
        });
    }

    const double wallMs =
        std::chrono::duration<double, std::milli>(replayEnd - replayStart).count();
    json result = {
        {"mode", "formula_replay"},
        {"crop_dir", fs::absolute(args.formulaReplayCropDir).string()},
        {"crop_count", static_cast<int>(cropPaths.size())},
        {"batch_size", args.formulaReplayBatchSize},
        {"dynamic_shape_packing", config.packDynamicShapes},
        {"cpu_mem_arena", config.enableCpuMemArena},
        {"ort_profile", ortProfile.empty() ? "default" : ortProfile},
        {"ort_intra_threads", config.intraOpThreads},
        {"ort_inter_threads", config.interOpThreads},
        {"model_path", fs::absolute(modelPath).string()},
        {"pure_infer_ms", timing.inferMs},
        {"avg_infer_ms_per_crop",
         timing.cropCount > 0 ? timing.inferMs / static_cast<double>(timing.cropCount) : 0.0},
        {"preprocess_ms", timing.preprocessMs},
        {"decode_ms", timing.decodeMs},
        {"normalize_ms", timing.normalizeMs},
        {"total_ms", timing.totalMs},
        {"wall_ms", wallMs},
        {"timing", {
            {"preprocess_ms", timing.preprocessMs},
            {"infer_ms", timing.inferMs},
            {"decode_ms", timing.decodeMs},
            {"normalize_ms", timing.normalizeMs},
            {"total_ms", timing.totalMs},
            {"crop_count", timing.cropCount},
            {"batch_count", timing.batchCount},
            {"batches", formulaBatchRecordsToJson(timing)},
        }},
        {"crops", std::move(cropInputs)},
        {"outputs", std::move(outputs)},
    };

    fs::path jsonOutPath(args.formulaReplayJsonOut);
    if (!jsonOutPath.parent_path().empty()) {
        fs::create_directories(jsonOutPath.parent_path());
    }
    std::ofstream out(jsonOutPath);
    if (!out) {
        LOG_ERROR("Failed to open formula replay JSON output: {}", args.formulaReplayJsonOut);
        return 1;
    }
    out << result.dump(2) << "\n";
    LOG_INFO("Saved Formula replay JSON: {}", args.formulaReplayJsonOut);
    std::cout << result.dump(2) << "\n";
    return 0;
}

int main(int argc, char* argv[]) {
    CliArgs args;
    if (!parseArgs(argc, argv, args)) {
        return args.showHelp ? 0 : 1;
    }

    // Set log level
    if (args.verbose) {
        spdlog::set_level(spdlog::level::debug);
    } else {
        spdlog::set_level(spdlog::level::info);
    }

    if (args.isFormulaReplay()) {
        return runFormulaReplay(args);
    }

    // Check input file exists
    if (!fs::exists(args.inputPath)) {
        LOG_ERROR("Input file not found: {}", args.inputPath);
        return 1;
    }

    // Configure pipeline
    rapid_doc::PipelineConfig config = rapid_doc::PipelineConfig::Default(PROJECT_ROOT_DIR);
    config.runtime.outputDir = args.outputDir;
    config.runtime.pdfDpi = args.dpi;
    config.runtime.maxPages = args.maxPages;
    config.stages.enableWiredTable = args.enableTable;
    config.stages.enableOcr = args.enableOcr;
    config.stages.enableMarkdownOutput = !args.jsonOnly;
    config.runtime.saveVisualization = args.detail;

    // Create and initialize pipeline
    rapid_doc::DocPipeline pipeline(config);
    
    pipeline.setProgressCallback([](const std::string& stage, int current, int total) {
        std::cout << "\r[" << stage << "] " << current << "/" << total << std::flush;
    });

    if (!pipeline.initialize()) {
        LOG_ERROR("Failed to initialize pipeline");
        return 1;
    }

    // Process document
    LOG_INFO("Processing: {}", args.inputPath);
    auto result = pipeline.processPdf(args.inputPath);
    std::cout << "\n";  // New line after progress

    // Create output directory
    fs::create_directories(args.outputDir);

    // Save results
    std::string baseName = fs::path(args.inputPath).stem().string();

    // Save Markdown
    if (!args.jsonOnly && !result.markdown.empty()) {
        std::string mdPath = args.outputDir + "/" + baseName + ".md";
        std::ofstream mdFile(mdPath);
        mdFile << result.markdown;
        LOG_INFO("Saved Markdown: {}", mdPath);
    }

    // Save JSON content list
    if (!result.contentListJson.empty()) {
        std::string jsonPath = args.outputDir + "/" + baseName + "_content.json";
        std::ofstream jsonFile(jsonPath);
        jsonFile << result.contentListJson;
        LOG_INFO("Saved JSON: {}", jsonPath);
    }

    if (!result.formulaTraceJson.empty()) {
        std::string tracePath = args.outputDir + "/" + baseName + "_formula_trace.json";
        std::ofstream traceFile(tracePath);
        traceFile << result.formulaTraceJson;
        LOG_INFO("Saved Formula trace sidecar: {}", tracePath);
    }

    if (args.detail) {
        std::string detailPath = args.detailPath;
        if (detailPath.empty()) {
            detailPath = args.outputDir + "/" + baseName + "_detail.txt";
        }

        rapid_doc::DetailReportOptions detailOptions;
        detailOptions.inputPath = args.inputPath;
        detailOptions.stageConfig = config.stages;
        detailOptions.saveImages = config.runtime.saveImages;
        detailOptions.saveVisualization = config.runtime.saveVisualization;
        detailOptions.artifacts.outputDir = args.outputDir;
        if (!args.jsonOnly && !result.markdown.empty()) {
            detailOptions.artifacts.markdownPath = args.outputDir + "/" + baseName + ".md";
        }
        if (!result.contentListJson.empty()) {
            detailOptions.artifacts.contentListPath = args.outputDir + "/" + baseName + "_content.json";
        }
        if (config.runtime.saveVisualization) {
            detailOptions.artifacts.layoutDir = args.outputDir + "/layout";
        }
        if (config.runtime.saveImages) {
            detailOptions.artifacts.imagesDir = args.outputDir + "/images";
        }

        const std::string detailReport = rapid_doc::buildDetailReport(result, detailOptions);
        std::ofstream detailFile(detailPath);
        detailFile << detailReport;
        LOG_INFO("Saved detail report: {}", detailPath);
        std::cout << "\n" << detailReport;
    }

    // Print summary
    std::cout << "\n========================================\n";
    std::cout << "Processing Complete\n";
    std::cout << "========================================\n";
    std::cout << "  Pages processed: " << result.processedPages << "/" << result.totalPages << "\n";
    std::cout << "  Skipped elements: " << result.skippedElements << " (NPU unsupported)\n";
    std::cout << "  Total time: " << result.totalTimeMs << " ms\n";
    std::cout << "  Output: " << args.outputDir << "\n";
    std::cout << "========================================\n";

    return 0;
}
