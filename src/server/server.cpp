/**
 * @file server.cpp
 * @brief HTTP server implementation using Crow
 */

#include "server/server.h"
#include "common/logger.h"

// Crow HTTP framework (header-only)
#ifndef CROW_MAIN
#define CROW_MAIN
#endif
#include <crow.h>

#include <filesystem>
#include <fstream>
#include <chrono>

namespace fs = std::filesystem;

namespace rapid_doc {

// Base64 decoding helper
static std::vector<uint8_t> base64Decode(const std::string& encoded) {
    static const std::string base64Chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    
    std::vector<uint8_t> decoded;
    decoded.reserve(encoded.size() * 3 / 4);

    int val = 0, bits = -8;
    for (char c : encoded) {
        if (c == '=') break;
        auto pos = base64Chars.find(c);
        if (pos == std::string::npos) continue;
        val = (val << 6) + static_cast<int>(pos);
        bits += 6;
        if (bits >= 0) {
            decoded.push_back(static_cast<uint8_t>((val >> bits) & 0xFF));
            bits -= 8;
        }
    }
    return decoded;
}

DocServer::DocServer(const ServerConfig& config)
    : config_(config)
{
    // Create upload directory
    fs::create_directories(config_.uploadDir);

    // Initialize pipeline
    pipeline_ = std::make_unique<DocPipeline>(config_.pipelineConfig);
    if (!pipeline_->initialize()) {
        throw std::runtime_error("Failed to initialize document pipeline");
    }
}

DocServer::~DocServer() {
    stop();
}

void DocServer::run() {
    LOG_INFO("Starting RapidDoc HTTP server on {}:{}", config_.host, config_.port);

    crow::SimpleApp app;

    // GET /health - Health check
    CROW_ROUTE(app, "/health")
    ([]() {
        return crow::response(200, "OK");
    });

    // GET /status - Server statistics
    CROW_ROUTE(app, "/status")
    ([this]() {
        return crow::response(200, buildStatusJson());
    });

    // POST /process - Process uploaded PDF
    CROW_ROUTE(app, "/process").methods("POST"_method)
    ([this](const crow::request& req) {
        requestCount_++;
        
        try {
            // Check content type
            auto contentType = req.get_header_value("Content-Type");
            if (contentType.find("multipart/form-data") == std::string::npos) {
                errorCount_++;
                return crow::response(400, R"({"error": "Expected multipart/form-data"})");
            }

            // Parse multipart
            crow::multipart::message msg(req);
            
            auto it = msg.part_map.find("file");
            if (it == msg.part_map.end()) {
                errorCount_++;
                return crow::response(400, R"({"error": "No 'file' field in form"})");
            }

            const auto& filePart = it->second;
            std::string filename = "upload.pdf";
            
            // Get original filename from header
            auto cdIt = filePart.headers.find("Content-Disposition");
            if (cdIt != filePart.headers.end()) {
                auto pos = cdIt->second.params.find("filename");
                if (pos != cdIt->second.params.end()) {
                    filename = pos->second;
                }
            }

            // Process
            std::string result = handleProcess(filePart.body, filename);
            successCount_++;
            
            crow::response resp(200, result);
            resp.set_header("Content-Type", "application/json");
            return resp;
        }
        catch (const std::exception& e) {
            errorCount_++;
            LOG_ERROR("Processing error: {}", e.what());
            return crow::response(500, 
                std::string(R"({"error": ")") + e.what() + R"("})");
        }
    });

    // POST /process/base64 - Process base64 encoded PDF
    CROW_ROUTE(app, "/process/base64").methods("POST"_method)
    ([this](const crow::request& req) {
        requestCount_++;

        try {
            auto body = crow::json::load(req.body);
            if (!body) {
                errorCount_++;
                return crow::response(400, R"({"error": "Invalid JSON"})");
            }

            std::string data = body["data"].s();
            std::string filename = body.has("filename") ? body["filename"].s() : "upload.pdf";

            // Decode base64
            auto decoded = base64Decode(data);
            if (decoded.empty()) {
                errorCount_++;
                return crow::response(400, R"({"error": "Invalid base64 data"})");
            }

            // Process from memory
            auto result = pipeline_->processPdfFromMemory(decoded.data(), decoded.size());
            
            // Build response JSON
            nlohmann::json response;
            response["pages"] = result.processedPages;
            response["total_pages"] = result.totalPages;
            response["skipped"] = result.skippedElements;
            response["time_ms"] = result.totalTimeMs;
            response["markdown"] = result.markdown;
            response["content_list"] = nlohmann::json::parse(result.contentListJson);
            
            successCount_++;
            crow::response resp(200, response.dump());
            resp.set_header("Content-Type", "application/json");
            return resp;
        }
        catch (const std::exception& e) {
            errorCount_++;
            LOG_ERROR("Processing error: {}", e.what());
            return crow::response(500,
                std::string(R"({"error": ")") + e.what() + R"("})");
        }
    });

    running_ = true;
    app.bindaddr(config_.host)
       .port(config_.port)
       .concurrency(config_.numWorkers)
       .run();
    running_ = false;
}

void DocServer::stop() {
    running_ = false;
    LOG_INFO("RapidDoc HTTP server stopped");
}

std::string DocServer::handleProcess(const std::string& pdfData, const std::string& filename) {
    auto startTime = std::chrono::steady_clock::now();

    // Save to temp file
    std::string tempPath = config_.uploadDir + "/" + filename;
    {
        std::ofstream ofs(tempPath, std::ios::binary);
        ofs.write(pdfData.data(), pdfData.size());
    }

    // Process
    auto result = pipeline_->processPdf(tempPath);

    // Remove temp file
    fs::remove(tempPath);

    auto endTime = std::chrono::steady_clock::now();
    double totalMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    // Build response
    nlohmann::json response;
    response["pages"] = result.processedPages;
    response["total_pages"] = result.totalPages;
    response["skipped"] = result.skippedElements;
    response["time_ms"] = totalMs;
    response["stats"]["pdf_render_ms"] = result.stats.pdfRenderTimeMs;
    response["stats"]["layout_ms"] = result.stats.layoutTimeMs;
    response["stats"]["ocr_ms"] = result.stats.ocrTimeMs;
    response["stats"]["table_ms"] = result.stats.tableTimeMs;
    response["stats"]["output_gen_ms"] = result.stats.outputGenTimeMs;
    response["markdown"] = result.markdown;
    response["content_list"] = nlohmann::json::parse(result.contentListJson);

    return response.dump();
}

std::string DocServer::buildStatusJson() {
    nlohmann::json status;
    status["status"] = "running";
    status["requests"] = requestCount_.load();
    status["success"] = successCount_.load();
    status["errors"] = errorCount_.load();
    status["npu_support"]["layout"] = true;
    status["npu_support"]["ocr"] = true;
    status["npu_support"]["table_wired"] = true;
    status["npu_support"]["table_wireless"] = false;
    status["npu_support"]["formula"] = false;
    return status.dump();
}

} // namespace rapid_doc
