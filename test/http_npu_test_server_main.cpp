#include "server/server.h"
#include "common/config.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <string>

namespace {

int parseIntFlag(int argc, char* argv[], const std::string& flag, int defaultValue) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == flag) {
            return std::atoi(argv[i + 1]);
        }
    }
    return defaultValue;
}

std::string parseStringFlag(
    int argc,
    char* argv[],
    const std::string& flag,
    const std::string& defaultValue)
{
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == flag) {
            return argv[i + 1];
        }
    }
    return defaultValue;
}

bool hasFlag(int argc, char* argv[], const std::string& flag) {
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == flag) {
            return true;
        }
    }
    return false;
}

} // namespace

int main(int argc, char* argv[]) {
    try {
        rapid_doc::ServerConfig cfg;
        cfg.host = parseStringFlag(argc, argv, "--host", "127.0.0.1");
        cfg.port = parseIntFlag(argc, argv, "--port", 18180);
        cfg.numWorkers = parseIntFlag(argc, argv, "--workers", 2);
        cfg.uploadDir = std::string(PROJECT_ROOT_DIR) + "/test/fixtures/http_npu_server_output";

        const bool enableTable = hasFlag(argc, argv, "--enable-table");

        cfg.pipelineConfig = rapid_doc::PipelineConfig::Default(PROJECT_ROOT_DIR);
        cfg.pipelineConfig.stages.enableLayout = true;
        cfg.pipelineConfig.stages.enableOcr = true;
        cfg.pipelineConfig.stages.enableWiredTable = enableTable;
        cfg.pipelineConfig.stages.enableFormula = false;
        cfg.pipelineConfig.stages.enableReadingOrder = true;
        cfg.pipelineConfig.stages.enableMarkdownOutput = true;
        cfg.pipelineConfig.runtime.saveImages = false;
        cfg.pipelineConfig.runtime.saveVisualization = false;
        cfg.pipelineConfig.runtime.outputDir =
            std::string(PROJECT_ROOT_DIR) + "/test/fixtures/http_npu_server_output/result";

        rapid_doc::DocServer server(cfg);
        server.run();
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "rapid_doc_npu_test_server failed: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "rapid_doc_npu_test_server failed with unknown exception" << std::endl;
        return 1;
    }
}
