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
#include <mutex>
#include <cstdint>

namespace rapid_doc {

class DocServerTestAccess;

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
 *   POST /file_parse        - Python RapidDoc-compatible batch parsing API
 *   POST /v1/images:annotate- Vision API-compatible image annotation
 *   POST /process           - Legacy single-file PDF processing
 *   POST /process/base64    - Legacy base64 PDF processing
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
    friend class DocServerTestAccess;

    ServerConfig config_;
    std::unique_ptr<DocPipeline> pipeline_;
    std::atomic<bool> running_{false};
    
    // Statistics
    std::atomic<uint64_t> requestCount_{0};
    std::atomic<uint64_t> successCount_{0};
    std::atomic<uint64_t> errorCount_{0};
    // Serializes NPU-bound pipeline inference calls only.
    // HTTP parsing, file I/O, and JSON serialization run outside this lock.
    std::mutex pipelineMutex_;
    std::atomic<uint64_t> lockSamples_{0};
    std::atomic<uint64_t> lockWaitUsTotal_{0};
    std::atomic<uint64_t> lockHoldUsTotal_{0};
    std::atomic<uint64_t> lockWaitUsMax_{0};
    std::atomic<uint64_t> lockHoldUsMax_{0};
    std::atomic<uint64_t> npuStageUsTotal_{0};
    std::atomic<uint64_t> cpuStageUsTotal_{0};

    // Internal handlers
    void setupRoutes();
    std::string handleProcess(const std::string& pdfData, const std::string& filename);
    std::string buildStatusJson();
    void recordPipelineLockStats(const DocumentResult& result);
};

} // namespace rapid_doc
