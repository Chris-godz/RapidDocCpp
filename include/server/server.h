/**
 * @file server.h
 * @brief HTTP server interface for RapidDoc
 * 
 * Provides REST API for document processing using Crow framework.
 * Follows API patterns from DXNN-OCR-cpp server module.
 */

#pragma once

#include "pipeline/doc_pipeline.h"
#include <memory>
#include <string>
#include <functional>
#include <atomic>

namespace rapid_doc {

/**
 * @brief HTTP server configuration
 */
struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    int numWorkers = 4;
    size_t maxUploadSize = 50 * 1024 * 1024;  // 50MB
    std::string uploadDir = "./uploads";
    
    // Pipeline config
    PipelineConfig pipelineConfig;
};

/**
 * @class DocServer
 * @brief HTTP server for document processing
 * 
 * REST API endpoints:
 *   POST /process           - Process uploaded PDF file
 *   POST /process/base64    - Process base64 encoded PDF
 *   GET  /health            - Health check
 *   GET  /status            - Server status and statistics
 */
class DocServer {
public:
    explicit DocServer(const ServerConfig& config);
    ~DocServer();

    /**
     * @brief Start the HTTP server (blocking)
     */
    void run();

    /**
     * @brief Stop the server
     */
    void stop();

    /**
     * @brief Check if server is running
     */
    bool isRunning() const { return running_.load(); }

private:
    ServerConfig config_;
    std::unique_ptr<DocPipeline> pipeline_;
    std::atomic<bool> running_{false};
    
    // Statistics
    std::atomic<uint64_t> requestCount_{0};
    std::atomic<uint64_t> successCount_{0};
    std::atomic<uint64_t> errorCount_{0};

    // Internal handlers
    void setupRoutes();
    std::string handleProcess(const std::string& pdfData, const std::string& filename);
    std::string buildStatusJson();
};

} // namespace rapid_doc
