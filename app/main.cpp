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
#include <iostream>
#include <string>
#include <fstream>
#include <filesystem>
#include <getopt.h>

namespace fs = std::filesystem;

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
    std::cout << "  -v, --verbose           Verbose logging\n";
    std::cout << "  -h, --help              Show this help\n";
    std::cout << "\n";
    std::cout << "Note: Formula recognition and wireless table recognition are not\n";
    std::cout << "      supported on DEEPX NPU and will be skipped.\n";
}

struct CliArgs {
    std::string inputPath;
    std::string outputDir = "./output";
    int dpi = 200;
    int maxPages = 0;
    bool enableTable = true;
    bool enableOcr = true;
    bool jsonOnly = false;
    bool verbose = false;
};

enum LongOnlyOpt {
    OPT_NO_TABLE = 256,
    OPT_NO_OCR,
    OPT_JSON_ONLY,
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
            case 'v': args.verbose = true; break;
            case 'h': printUsage(argv[0]); return false;
            default:  printUsage(argv[0]); return false;
        }
    }

    if (args.inputPath.empty()) {
        std::cerr << "Error: --input is required\n";
        printUsage(argv[0]);
        return false;
    }

    return true;
}

int main(int argc, char* argv[]) {
    CliArgs args;
    if (!parseArgs(argc, argv, args)) {
        return 1;
    }

    // Set log level
    if (args.verbose) {
        spdlog::set_level(spdlog::level::debug);
    } else {
        spdlog::set_level(spdlog::level::info);
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
