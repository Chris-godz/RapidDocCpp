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
#include <vector>

namespace rapid_doc {

class DocServerTestAccess;
class DeviceMetricsSampler;

/**
 * @brief HTTP server configuration
 */
struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    int numWorkers = 4;
    size_t maxUploadSize = 50 * 1024 * 1024;  // 50MB
    std::string uploadDir = "./uploads";
    std::string topology = "single_pipeline";
    std::string routingPolicy = "least_inflight_rr";
    std::string serverId;
    std::vector<int> deviceIds;
    
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

    struct PipelineShard {
        std::string shardId;
        int deviceId = -1;
        std::unique_ptr<DocPipeline> pipeline;
        std::mutex npuSerialMutex;
        std::mutex requestMutex;
        std::atomic<uint64_t> inflight{0};
        std::atomic<uint64_t> requestCount{0};
        std::atomic<uint64_t> busyUsTotal{0};
        std::atomic<uint64_t> npuBusyUsTotal{0};
        std::atomic<uint64_t> routeQueueUsTotal{0};
    };

    ServerConfig config_;
    std::vector<std::unique_ptr<PipelineShard>> shards_;
    std::unique_ptr<DeviceMetricsSampler> deviceMetricsSampler_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> routingCursor_{0};
    
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
    size_t selectShardIndex();
    std::string buildStatusJson();
    std::string resolvedTopology() const;
    void recordPipelineLockStats(const DocumentResult& result);
};

} // namespace rapid_doc
