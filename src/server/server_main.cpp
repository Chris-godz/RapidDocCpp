/**
 * @file server_main.cpp
 * @brief HTTP server entry point
 */

#include "server/server.h"
#include "common/config.h"
#include "common/logger.h"
#include <iostream>
#include <csignal>
#include <cstdlib>
#include <getopt.h>

static rapid_doc::DocServer* g_server = nullptr;

void signalHandler(int signum) {
    LOG_INFO("Received signal {}, shutting down...", signum);
    if (g_server) {
        g_server->stop();
    }
}

void printUsage(const char* programName) {
    std::cout << "RapidDoc HTTP Server (DEEPX NPU)\n\n";
    std::cout << "Usage: " << programName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -H, --host <addr>     Bind address (default: 0.0.0.0)\n";
    std::cout << "  -p, --port <num>      Port number (default: 8080)\n";
    std::cout << "  -w, --workers <num>   Worker threads (default: 4)\n";
    std::cout << "  -h, --help            Show this help\n";
    std::cout << "\n";
    std::cout << "API Endpoints:\n";
    std::cout << "  POST /process         - Process uploaded PDF (multipart/form-data)\n";
    std::cout << "  POST /process/base64  - Process base64 encoded PDF\n";
    std::cout << "  GET  /health          - Health check\n";
    std::cout << "  GET  /status          - Server statistics\n";
}

int main(int argc, char* argv[]) {
    rapid_doc::ServerConfig config;
    config.pipelineConfig = rapid_doc::PipelineConfig::Default(PROJECT_ROOT_DIR);

    // Parse arguments using getopt_long
    static const struct option longOpts[] = {
        {"host",    required_argument, nullptr, 'H'},
        {"port",    required_argument, nullptr, 'p'},
        {"workers", required_argument, nullptr, 'w'},
        {"help",    no_argument,       nullptr, 'h'},
        {nullptr,   0,                 nullptr, 0}
    };

    int opt;
    while ((opt = getopt_long(argc, argv, "H:p:w:h", longOpts, nullptr)) != -1) {
        switch (opt) {
            case 'H': config.host = optarg; break;
            case 'p': config.port = std::atoi(optarg); break;
            case 'w': config.numWorkers = std::atoi(optarg); break;
            case 'h': printUsage(argv[0]); return 0;
            default:  printUsage(argv[0]); return 1;
        }
    }

    // Set log level
    spdlog::set_level(spdlog::level::info);

    // Setup signal handlers
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    try {
        rapid_doc::DocServer server(config);
        g_server = &server;

        LOG_INFO("===========================================");
        LOG_INFO("RapidDoc HTTP Server (DEEPX NPU Edition)");
        LOG_INFO("===========================================");
        LOG_INFO("Host: {}", config.host);
        LOG_INFO("Port: {}", config.port);
        LOG_INFO("Workers: {}", config.numWorkers);
        LOG_INFO("NPU Supported: Layout, OCR, Wired Table");
        LOG_INFO("NPU Unsupported: Formula, Wireless Table");
        LOG_INFO("===========================================");

        server.run();
    }
    catch (const std::exception& e) {
        LOG_ERROR("Server error: {}", e.what());
        return 1;
    }

    return 0;
}
