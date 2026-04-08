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

#include <dxrt/device_info_status.h>
#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <thread>
#include <unordered_map>

#include <unistd.h>

namespace fs = std::filesystem;

namespace rapid_doc {

struct DeviceMetricSample {
    int deviceId = -1;
    uint64_t memoryTotalBytes = 0;
    std::optional<uint64_t> memoryLastUsedBytes;
    std::optional<uint64_t> memoryPeakUsedBytes;
};

class DeviceMetricsSampler {
public:
    explicit DeviceMetricsSampler(std::vector<int> deviceIds)
        : deviceIds_(std::move(deviceIds))
    {
        sampleOnce();
    }

    ~DeviceMetricsSampler() {
        stop();
    }

    void start() {
        bool expected = false;
        if (!running_.compare_exchange_strong(expected, true)) {
            return;
        }

        worker_ = std::thread([this]() {
            while (running_.load(std::memory_order_relaxed)) {
                sampleOnce();
                std::this_thread::sleep_for(std::chrono::milliseconds(200));
            }
        });
    }

    void stop() {
        bool expected = true;
        if (!running_.compare_exchange_strong(expected, false)) {
            return;
        }
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    std::vector<DeviceMetricSample> snapshot() const {
        std::lock_guard<std::mutex> lock(mutex_);
        std::vector<DeviceMetricSample> result;
        result.reserve(samples_.size());
        for (const auto& [deviceId, sample] : samples_) {
            (void)deviceId;
            result.push_back(sample);
        }
        std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.deviceId < rhs.deviceId;
        });
        return result;
    }

    std::string memoryTelemetryStatus() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return memoryTelemetryStatus_;
    }

private:
    void sampleOnce() {
        std::lock_guard<std::mutex> lock(mutex_);
        for (int deviceId : deviceIds_) {
            try {
                const auto status = dxrt::DeviceStatus::GetCurrentStatus(deviceId);
                auto& sample = samples_[deviceId];
                sample.deviceId = deviceId;
                sample.memoryTotalBytes = static_cast<uint64_t>(status.MemorySize());
            } catch (...) {
                memoryTelemetryStatus_ = "blocked_memory_telemetry_unavailable";
            }
        }
    }

    std::vector<int> deviceIds_;
    mutable std::mutex mutex_;
    std::unordered_map<int, DeviceMetricSample> samples_;
    std::string memoryTelemetryStatus_ = "blocked_memory_telemetry_unavailable";
    std::atomic<bool> running_{false};
    std::thread worker_;
};

namespace {

using json = nlohmann::json;

const std::vector<std::string> kPdfSuffixes = {".pdf"};
const std::vector<std::string> kImageSuffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"};

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool isPdfExtension(const std::string& extension) {
    const std::string lower = toLower(extension);
    return std::find(kPdfSuffixes.begin(), kPdfSuffixes.end(), lower) != kPdfSuffixes.end();
}

bool isImageExtension(const std::string& extension) {
    const std::string lower = toLower(extension);
    return std::find(kImageSuffixes.begin(), kImageSuffixes.end(), lower) != kImageSuffixes.end();
}

std::string contentElementTypeToString(ContentElement::Type type) {
    switch (type) {
        case ContentElement::Type::TEXT: return "text";
        case ContentElement::Type::TITLE: return "title";
        case ContentElement::Type::IMAGE: return "image";
        case ContentElement::Type::TABLE: return "table";
        case ContentElement::Type::EQUATION: return "equation";
        case ContentElement::Type::CODE: return "code";
        case ContentElement::Type::LIST: return "list";
        case ContentElement::Type::HEADER: return "header";
        case ContentElement::Type::FOOTER: return "footer";
        case ContentElement::Type::REFERENCE: return "reference";
        default: return "unknown";
    }
}

std::string safeFilename(std::string filename) {
    filename = fs::path(filename).filename().string();
    if (filename.empty()) {
        return "upload.bin";
    }

    for (char& ch : filename) {
        if (ch == '/' || ch == '\\' || ch == ':' || ch == '\0') {
            ch = '_';
        }
    }
    return filename;
}

std::string safeStem(const std::string& filename) {
    std::string stem = fs::path(filename).stem().string();
    if (stem.empty()) {
        return "document";
    }

    for (char& ch : stem) {
        if (ch == '/' || ch == '\\' || ch == ':' || ch == '\0') {
            ch = '_';
        }
    }
    return stem;
}

bool parseBool(const std::string& rawValue, bool defaultValue) {
    if (rawValue.empty()) {
        return defaultValue;
    }

    const std::string value = toLower(rawValue);
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
    }
    return defaultValue;
}

int parseInt(const std::string& rawValue, int defaultValue) {
    if (rawValue.empty()) {
        return defaultValue;
    }

    try {
        return std::stoi(rawValue);
    }
    catch (...) {
        return defaultValue;
    }
}

std::string canonicalParseMethod(const std::string& rawValue) {
    const std::string value = toLower(rawValue);
    if (value == "ocr" || value == "txt") {
        return value;
    }
    return "auto";
}

std::string mimeTypeForPath(const fs::path& path) {
    const std::string ext = toLower(path.extension().string());
    if (ext == ".png") return "image/png";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".bmp") return "image/bmp";
    if (ext == ".tif" || ext == ".tiff") return "image/tiff";
    if (ext == ".pdf") return "application/pdf";
    return "application/octet-stream";
}

int effectiveHttpIngressConcurrency(
    const std::string& topology,
    size_t shardCount,
    int configuredWorkers)
{
    // Benchmark worker_count remains the compute-lane count. The synchronous
    // HTTP handler can still hold a Crow worker for the entire request, so the
    // ingress thread pool needs enough slack to keep c6 connections open.
    int effective = std::max(6, configuredWorkers);
    if (topology == "single_process_multi_device" && shardCount > 1) {
        effective = std::max(effective, static_cast<int>(shardCount) + 1);
    }
    return effective;
}

void writeBinaryFile(const fs::path& path, const std::string& data) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path.string());
    }
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

void writeTextFile(const fs::path& path, const std::string& data) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path.string());
    }
    out << data;
}

std::string readBinaryFile(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return {};
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::string base64Encode(const unsigned char* data, size_t len) {
    static const char kBase64Chars[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string encoded;
    encoded.reserve(((len + 2) / 3) * 4);

    for (size_t i = 0; i < len; i += 3) {
        const unsigned int octetA = data[i];
        const unsigned int octetB = (i + 1 < len) ? data[i + 1] : 0;
        const unsigned int octetC = (i + 2 < len) ? data[i + 2] : 0;
        const unsigned int triple = (octetA << 16) | (octetB << 8) | octetC;

        encoded.push_back(kBase64Chars[(triple >> 18) & 0x3F]);
        encoded.push_back(kBase64Chars[(triple >> 12) & 0x3F]);
        encoded.push_back((i + 1 < len) ? kBase64Chars[(triple >> 6) & 0x3F] : '=');
        encoded.push_back((i + 2 < len) ? kBase64Chars[triple & 0x3F] : '=');
    }

    return encoded;
}

uint64_t msToUs(double ms) {
    if (ms <= 0.0) {
        return 0;
    }
    return static_cast<uint64_t>(ms * 1000.0);
}

uint64_t countToUInt(double value) {
    if (value <= 0.0) {
        return 0;
    }
    return static_cast<uint64_t>(std::llround(value));
}

void updateAtomicMax(std::atomic<uint64_t>& target, uint64_t value) {
    uint64_t current = target.load(std::memory_order_relaxed);
    while (current < value &&
           !target.compare_exchange_weak(
               current,
               value,
               std::memory_order_relaxed,
               std::memory_order_relaxed)) {
    }
}

std::vector<int> discoverDxrtDeviceIds() {
    std::vector<int> ids;
    try {
        const int count = dxrt::DeviceStatus::GetDeviceCount();
        for (int i = 0; i < count; ++i) {
            ids.push_back(i);
        }
    } catch (...) {
    }
    return ids;
}

std::string makeServerId(const ServerConfig& config, const std::string& topology) {
    if (!config.serverId.empty()) {
        return config.serverId;
    }

    std::ostringstream out;
    out << topology << "_pid" << ::getpid();
    return out.str();
}

std::vector<uint8_t> base64Decode(const std::string& encoded) {
    static const std::string kBase64Chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::vector<uint8_t> decoded;
    decoded.reserve(encoded.size() * 3 / 4);

    int value = 0;
    int bits = -8;
    for (unsigned char ch : encoded) {
        if (std::isspace(ch)) {
            continue;
        }
        if (ch == '=') {
            break;
        }
        const auto pos = kBase64Chars.find(static_cast<char>(ch));
        if (pos == std::string::npos) {
            continue;
        }
        value = (value << 6) + static_cast<int>(pos);
        bits += 6;
        if (bits >= 0) {
            decoded.push_back(static_cast<uint8_t>((value >> bits) & 0xFF));
            bits -= 8;
        }
    }

    return decoded;
}

std::vector<crow::multipart::part> getMultipartParts(
    const crow::multipart::message& msg,
    const std::string& name) {
    std::vector<crow::multipart::part> parts;
    const auto range = msg.part_map.equal_range(name);
    for (auto it = range.first; it != range.second; ++it) {
        parts.push_back(it->second);
    }
    return parts;
}

std::string getMultipartField(
    const crow::multipart::message& msg,
    const std::string& name,
    const std::string& defaultValue = {}) {
    const auto range = msg.part_map.equal_range(name);
    if (range.first == range.second) {
        return defaultValue;
    }
    return range.first->second.body;
}

std::vector<std::string> getMultipartFieldValues(
    const crow::multipart::message& msg,
    const std::string& name) {
    std::vector<std::string> values;
    const auto range = msg.part_map.equal_range(name);
    for (auto it = range.first; it != range.second; ++it) {
        values.push_back(it->second.body);
    }
    return values;
}

json makeEngineJson(bool tableEnabled, bool formulaEnabled) {
    return json{
        {"layout", "dxengine"},
        {"ocr", "dxengine"},
        {"formula", formulaEnabled ? "image_fallback" : "disabled"},
        {"table", tableEnabled ? "dxengine" : "disabled"},
    };
}

json getenvOrNull(const char* name) {
    const char* value = std::getenv(name);
    if (value == nullptr) {
        return nullptr;
    }
    return std::string(value);
}

json maybeUIntToJson(const std::optional<uint64_t>& value) {
    if (!value.has_value()) {
        return nullptr;
    }
    return *value;
}

json makeStatsJson(
    const DocumentResult& result,
    std::optional<double> pipelineCallMs = std::nullopt,
    std::optional<double> routeQueueMs = std::nullopt,
    std::optional<double> lbProxyMs = std::nullopt)
{
    const double pagesPerSec = result.totalTimeMs > 0.0
        ? (static_cast<double>(result.processedPages) * 1000.0 / result.totalTimeMs)
        : 0.0;
    json stats{
        {"pages", result.processedPages},
        {"total_pages", result.totalPages},
        {"skipped", result.skippedElements},
        {"time_ms", result.totalTimeMs},
        {"pages_per_sec", pagesPerSec},
        {"pdf_render_ms", result.stats.pdfRenderTimeMs},
        {"layout_ms", result.stats.layoutTimeMs},
        {"ocr_ms", result.stats.ocrTimeMs},
        {"table_ms", result.stats.tableTimeMs},
        {"reading_order_ms", result.stats.readingOrderTimeMs},
        {"npu_serial_ms", result.stats.npuSerialTimeMs},
        {"cpu_only_ms", result.stats.cpuOnlyTimeMs},
        {"npu_lock_wait_ms", result.stats.npuLockWaitTimeMs},
        {"npu_lock_hold_ms", result.stats.npuLockHoldTimeMs},
        {"npu_service_ms", result.stats.npuServiceTimeMs},
        {"npu_slot_wait_ms", result.stats.npuSlotWaitTimeMs},
        {"layout_npu_service_ms", result.stats.layoutNpuServiceTimeMs},
        {"layout_npu_slot_wait_ms", result.stats.layoutNpuSlotWaitTimeMs},
        {"ocr_outer_slot_hold_ms", result.stats.ocrOuterSlotHoldTimeMs},
        {"ocr_submodule_window_ms", result.stats.ocrSubmoduleWindowTimeMs},
        {"ocr_slot_wait_ms", result.stats.ocrSlotWaitTimeMs},
        {"ocr_collect_wait_ms", result.stats.ocrCollectWaitTimeMs},
        {"ocr_inflight_peak", result.stats.ocrInflightPeak},
        {"ocr_buffered_out_of_order_count", result.stats.ocrBufferedOutOfOrderCount},
        {"table_npu_service_ms", result.stats.tableNpuServiceTimeMs},
        {"table_npu_slot_wait_ms", result.stats.tableNpuSlotWaitTimeMs},
        {"table_ocr_service_ms", result.stats.tableOcrServiceTimeMs},
        {"table_ocr_slot_wait_ms", result.stats.tableOcrSlotWaitTimeMs},
        {"cpu_pre_ms", result.stats.cpuPreTimeMs},
        {"cpu_post_ms", result.stats.cpuPostTimeMs},
        {"finalize_cpu_ms", result.stats.finalizeCpuTimeMs},
        {"table_finalize_ms", result.stats.tableFinalizeTimeMs},
        {"ocr_collect_or_merge_ms", result.stats.ocrCollectOrMergeTimeMs},
        {"layout_queue_wait_ms", result.stats.layoutQueueWaitTimeMs},
        {"plan_queue_wait_ms", result.stats.planQueueWaitTimeMs},
        {"ocr_table_queue_wait_ms", result.stats.ocrTableQueueWaitTimeMs},
        {"finalize_queue_wait_ms", result.stats.finalizeQueueWaitTimeMs},
        {"render_queue_push_block_ms", result.stats.renderQueuePushBlockTimeMs},
        {"layout_queue_push_block_ms", result.stats.layoutQueuePushBlockTimeMs},
        {"plan_queue_push_block_ms", result.stats.planQueuePushBlockTimeMs},
        {"ocr_table_queue_push_block_ms", result.stats.ocrTableQueuePushBlockTimeMs},
        {"queue_backpressure_ms", result.stats.queueBackpressureTimeMs},
        {"pipeline_overlap_factor", result.stats.pipelineOverlapFactor},
        {"pipeline_mode", result.stats.pipelineMode},
        {"output_gen_ms", result.stats.outputGenTimeMs},
        {"text_boxes_raw_count", result.stats.textBoxesRawCount},
        {"text_boxes_after_dedup_count", result.stats.textBoxesAfterDedupCount},
        {"table_boxes_raw_count", result.stats.tableBoxesRawCount},
        {"table_boxes_after_dedup_count", result.stats.tableBoxesAfterDedupCount},
        {"ocr_submit_count", result.stats.ocrSubmitCount},
        {"ocr_submit_area_sum", result.stats.ocrSubmitAreaSum},
        {"ocr_submit_area_mean", result.stats.ocrSubmitAreaMean},
        {"ocr_submit_area_p50", result.stats.ocrSubmitAreaP50},
        {"ocr_submit_area_p95", result.stats.ocrSubmitAreaP95},
        {"ocr_submit_small_count", result.stats.ocrSubmitSmallCount},
        {"ocr_submit_medium_count", result.stats.ocrSubmitMediumCount},
        {"ocr_submit_large_count", result.stats.ocrSubmitLargeCount},
        {"ocr_submit_text_count", result.stats.ocrSubmitTextCount},
        {"ocr_submit_title_count", result.stats.ocrSubmitTitleCount},
        {"ocr_submit_code_count", result.stats.ocrSubmitCodeCount},
        {"ocr_submit_list_count", result.stats.ocrSubmitListCount},
        {"ocr_dedup_skipped_count", result.stats.ocrDedupSkippedCount},
        {"table_npu_submit_count", result.stats.tableNpuSubmitCount},
        {"table_dedup_skipped_count", result.stats.tableDedupSkippedCount},
        {"ocr_timeout_count", result.stats.ocrTimeoutCount},
        {"ocr_buffered_result_hit_count", result.stats.ocrBufferedResultHitCount},
    };

    if (pipelineCallMs.has_value()) {
        stats["pipeline_call_ms"] = *pipelineCallMs;
    }
    if (routeQueueMs.has_value()) {
        stats["route_queue_ms"] = *routeQueueMs;
    }
    if (lbProxyMs.has_value()) {
        stats["lb_proxy_ms"] = *lbProxyMs;
    }
    return stats;
}

json buildContentListJson(const DocumentResult& result) {
    if (result.contentListJson.empty()) {
        return json::array();
    }

    try {
        return json::parse(result.contentListJson);
    }
    catch (...) {
        return json::array();
    }
}

json buildMiddleJson(const DocumentResult& result) {
    json pdfInfo = json::array();

    for (const auto& page : result.pages) {
        json elements = json::array();
        for (const auto& elem : page.elements) {
            json item{
                {"type", contentElementTypeToString(elem.type)},
                {"bbox", {elem.layoutBox.x0, elem.layoutBox.y0, elem.layoutBox.x1, elem.layoutBox.y1}},
                {"score", elem.confidence},
                {"page_idx", elem.pageIndex},
                {"reading_order", elem.readingOrder},
                {"skipped", elem.skipped},
            };
            if (!elem.text.empty()) {
                item["text"] = elem.text;
            }
            if (!elem.html.empty()) {
                item["html"] = elem.html;
            }
            if (!elem.imagePath.empty()) {
                item["image_path"] = elem.imagePath;
            }
            elements.push_back(std::move(item));
        }

        pdfInfo.push_back(json{
            {"page_idx", page.pageIndex},
            {"page_size", {page.pageWidth, page.pageHeight}},
            {"elements", std::move(elements)},
        });
    }

    return json{{"pdf_info", std::move(pdfInfo)}};
}

json buildModelJson(const DocumentResult& result) {
    json pages = json::array();

    for (const auto& page : result.pages) {
        json layoutDetections = json::array();
        for (const auto& box : page.layoutResult.boxes) {
            layoutDetections.push_back(json{
                {"category_id", box.clsId >= 0 ? box.clsId : static_cast<int>(box.category)},
                {"label", box.label.empty() ? layoutCategoryToString(box.category) : box.label},
                {"poly", {
                    static_cast<int>(box.x0), static_cast<int>(box.y0),
                    static_cast<int>(box.x1), static_cast<int>(box.y0),
                    static_cast<int>(box.x1), static_cast<int>(box.y1),
                    static_cast<int>(box.x0), static_cast<int>(box.y1)
                }},
                {"bbox", {box.x0, box.y0, box.x1, box.y1}},
                {"score", box.confidence},
            });
        }

        pages.push_back(json{
            {"layout_dets", std::move(layoutDetections)},
            {"page_info", {
                {"page_no", page.pageIndex},
                {"width", page.pageWidth},
                {"height", page.pageHeight},
            }},
        });
    }

    return pages;
}

json collectImagesAsDataUrls(const fs::path& imageDir) {
    json images = json::object();
    if (!fs::exists(imageDir)) {
        return images;
    }

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(imageDir)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    for (const auto& path : files) {
        const std::string raw = readBinaryFile(path);
        if (raw.empty()) {
            continue;
        }

        images[path.filename().string()] =
            "data:" + mimeTypeForPath(path) + ";base64," +
            base64Encode(reinterpret_cast<const unsigned char*>(raw.data()), raw.size());
    }

    return images;
}

json collectAbsoluteFiles(const fs::path& dir) {
    json files = json::array();
    if (!fs::exists(dir)) {
        return files;
    }

    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            paths.push_back(fs::absolute(entry.path()));
        }
    }
    std::sort(paths.begin(), paths.end());

    for (const auto& path : paths) {
        files.push_back(path.string());
    }
    return files;
}

std::string makeRequestId() {
    static std::atomic<uint64_t> counter{0};
    const auto nowMs = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    const uint64_t id = counter.fetch_add(1, std::memory_order_relaxed);
    std::ostringstream out;
    out << "req_" << nowMs << "_pid" << ::getpid() << "_" << id;
    return out.str();
}

struct FileParseOptions {
    std::string outputDir = "./output-offline";
    bool clearOutputFile = false;
    std::vector<std::string> langList = {"ch"};
    std::string backend = "pipeline";
    std::string parseMethod = "auto";
    bool formulaEnable = true;
    bool tableEnable = true;
    bool returnMd = true;
    bool returnMiddleJson = false;
    bool returnModelOutput = false;
    bool returnContentList = false;
    bool returnImages = false;
    int startPageId = 0;
    int endPageId = 99999;
    bool deepxRequested = true;
    std::string layoutEngine = "dxengine";
    std::string ocrEngine = "dxengine";
    std::string formulaEngine = "image_fallback";
    std::string tableEngine = "dxengine";
    std::string pipelineMode;
    std::string ocrOuterMode;
    size_t ocrShadowWindow = 0;
};

std::vector<std::string> collectRequestWarnings(const FileParseOptions& options) {
    std::vector<std::string> warnings;
    if (!options.deepxRequested) {
        warnings.push_back("C++ backend is DEEPX-only; 'deepx=false' was ignored.");
    }
    if (!options.layoutEngine.empty() && toLower(options.layoutEngine) != "dxengine") {
        warnings.push_back("layout_engine request ignored; C++ backend uses dxengine.");
    }
    if (!options.ocrEngine.empty() && toLower(options.ocrEngine) != "dxengine") {
        warnings.push_back("ocr_engine request ignored; C++ backend uses dxengine.");
    }
    if (!options.tableEngine.empty() && toLower(options.tableEngine) != "dxengine") {
        warnings.push_back("table_engine request ignored; C++ backend uses dxengine.");
    }
    if (!options.formulaEngine.empty() &&
        toLower(options.formulaEngine) != "onnxruntime" &&
        toLower(options.formulaEngine) != "image_fallback") {
        warnings.push_back("formula_engine request ignored; C++ backend uses image fallback.");
    }
    if (options.parseMethod == "txt") {
        warnings.push_back("parse_method=txt is not natively supported in C++; falling back to auto pipeline.");
    }
    PipelineMode ignoredMode = PipelineMode::Serial;
    if (!options.pipelineMode.empty() && !parsePipelineMode(options.pipelineMode, ignoredMode)) {
        warnings.push_back("pipeline_mode request ignored; falling back to configured pipeline mode.");
    }
    OcrOuterMode ignoredOcrOuterMode = OcrOuterMode::ImmediatePerTask;
    if (!options.ocrOuterMode.empty() &&
        !parseOcrOuterMode(options.ocrOuterMode, ignoredOcrOuterMode)) {
        warnings.push_back("ocr_outer_mode request ignored; falling back to configured OCR outer mode.");
    }
    return warnings;
}

PipelineRunOverrides makeRunOverrides(
    const DocPipeline& pipeline,
    const FileParseOptions& options,
    const fs::path& outputDir)
{
    PipelineRunOverrides overrides;
    overrides.outputDir = outputDir.string();
    overrides.saveImages = true;
    overrides.saveVisualization = true;
    overrides.startPageId = options.startPageId;
    overrides.endPageId = options.endPageId;
    PipelineMode mode = pipeline.config().runtime.pipelineMode;
    overrides.enableFormula = options.formulaEnable;
    overrides.enableWiredTable = options.tableEnable;
    overrides.enableMarkdownOutput = pipeline.config().stages.enableMarkdownOutput;
    if (parsePipelineMode(options.pipelineMode, mode)) {
        overrides.pipelineMode = mode;
    }
    OcrOuterMode ocrOuterMode = pipeline.config().runtime.ocrOuterMode;
    if (parseOcrOuterMode(options.ocrOuterMode, ocrOuterMode)) {
        overrides.ocrOuterMode = ocrOuterMode;
    }
    if (options.ocrShadowWindow > 0) {
        overrides.ocrShadowWindow = options.ocrShadowWindow;
    }
    return overrides;
}

struct ProcessedDocument {
    std::string requestId;
    std::string filename;
    fs::path requestDir;
    fs::path parseDir;
    fs::path imagesDir;
    fs::path layoutDir;
    fs::path markdownPath;
    fs::path contentListPath;
    fs::path middleJsonPath;
    fs::path modelJsonPath;
    json contentList = json::array();
    json middleJson = json::object();
    json modelJson = json::array();
    DocumentResult result;
    std::vector<std::string> warnings;
    double prepareTimeMs = 0.0;
    double pipelineCallTimeMs = 0.0;
    double assemblyTimeMs = 0.0;
};

struct DispatchMetadata {
    std::string topology = "single_pipeline";
    int deviceId = -1;
    std::string shardId;
    std::string backendId;
    double routeQueueMs = 0.0;
    double lbProxyMs = 0.0;
};

struct RoutedProcessedDocument {
    ProcessedDocument processed;
    DispatchMetadata dispatch;
};

ProcessedDocument processDocumentBytes(
    DocPipeline& pipeline,
    const std::string& bytes,
    const std::string& filename,
    const FileParseOptions& options) {
    if (options.backend != "pipeline") {
        throw std::runtime_error("Unsupported backend: " + options.backend);
    }

    const std::string cleanName = safeFilename(filename);
    const std::string stem = safeStem(cleanName);
    const std::string extension = toLower(fs::path(cleanName).extension().string());

    ProcessedDocument processed;
    const auto prepareStart = std::chrono::steady_clock::now();
    processed.requestId = makeRequestId();
    processed.filename = cleanName;
    processed.requestDir = fs::absolute(fs::path(options.outputDir) / processed.requestId);
    processed.parseDir = processed.requestDir / stem / options.parseMethod;
    processed.imagesDir = processed.parseDir / "images";
    processed.layoutDir = processed.parseDir / "layout";
    processed.markdownPath = processed.parseDir / (stem + ".md");
    processed.contentListPath = processed.parseDir / (stem + "_content_list.json");
    processed.middleJsonPath = processed.parseDir / (stem + "_middle.json");
    processed.modelJsonPath = processed.parseDir / (stem + "_model.json");
    processed.warnings = collectRequestWarnings(options);

    fs::create_directories(processed.parseDir);
    writeBinaryFile(processed.parseDir / (stem + "_origin" + extension), bytes);

    const PipelineRunOverrides overrides = makeRunOverrides(
        pipeline, options, processed.parseDir);

    const bool isPdf = isPdfExtension(extension);
    const bool isImage = isImageExtension(extension);
    cv::Mat decodedImage;
    if (!isPdf && !isImage) {
        throw std::runtime_error("Unsupported file type: " + extension);
    }
    if (isImage) {
        cv::Mat encoded(
            1,
            static_cast<int>(bytes.size()),
            CV_8UC1,
            const_cast<char*>(bytes.data()));
        decodedImage = cv::imdecode(encoded, cv::IMREAD_COLOR);
        if (decodedImage.empty()) {
            throw std::runtime_error("Failed to decode image: " + cleanName);
        }
    }
    const auto prepareEnd = std::chrono::steady_clock::now();
    processed.prepareTimeMs =
        std::chrono::duration<double, std::milli>(prepareEnd - prepareStart).count();

    const auto pipelineStart = std::chrono::steady_clock::now();
    if (isPdf) {
        processed.result = pipeline.processPdfFromMemoryWithOverrides(
            reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size(), overrides);
    } else if (isImage) {
        processed.result = pipeline.processImageDocumentWithOverrides(
            decodedImage, 0, overrides);
    }
    const auto pipelineEnd = std::chrono::steady_clock::now();
    processed.pipelineCallTimeMs =
        std::chrono::duration<double, std::milli>(pipelineEnd - pipelineStart).count();

    const auto assemblyStart = std::chrono::steady_clock::now();
    writeTextFile(processed.markdownPath, processed.result.markdown);
    writeTextFile(processed.contentListPath, processed.result.contentListJson);

    processed.contentList = buildContentListJson(processed.result);
    processed.middleJson = buildMiddleJson(processed.result);
    processed.modelJson = buildModelJson(processed.result);

    writeTextFile(processed.middleJsonPath, processed.middleJson.dump(2));
    writeTextFile(processed.modelJsonPath, processed.modelJson.dump(2));
    const auto assemblyEnd = std::chrono::steady_clock::now();
    processed.assemblyTimeMs =
        std::chrono::duration<double, std::milli>(assemblyEnd - assemblyStart).count();

    LOG_INFO(
        "request={} lock_wait={:.2f}ms lock_hold={:.2f}ms npu={:.2f}ms cpu={:.2f}ms "
        "prepare={:.2f}ms pipeline_call={:.2f}ms assemble={:.2f}ms pages={}/{}",
        processed.requestId,
        processed.result.stats.npuLockWaitTimeMs,
        processed.result.stats.npuLockHoldTimeMs,
        processed.result.stats.npuSerialTimeMs,
        processed.result.stats.cpuOnlyTimeMs,
        processed.prepareTimeMs,
        processed.pipelineCallTimeMs,
        processed.assemblyTimeMs,
        processed.result.processedPages,
        processed.result.totalPages);

    return processed;
}

json buildFileResult(
    const ProcessedDocument& processed,
    const FileParseOptions& options,
    const DispatchMetadata& dispatch)
{
    json result{
        {"filename", processed.filename},
        {"backend", options.backend},
        {"deepx", true},
        {"engines", makeEngineJson(options.tableEnable, options.formulaEnable)},
        {"topology", dispatch.topology},
        {"device_id", dispatch.deviceId},
        {"shard_id", dispatch.shardId},
        {"backend_id", dispatch.backendId},
        {"stats", makeStatsJson(
            processed.result,
            processed.pipelineCallTimeMs,
            dispatch.routeQueueMs,
            dispatch.lbProxyMs)},
        {"output_dir", processed.parseDir.string()},
        {"markdown_path", processed.markdownPath.string()},
        {"content_list_path", processed.contentListPath.string()},
        {"middle_json_path", processed.middleJsonPath.string()},
        {"model_output_path", processed.modelJsonPath.string()},
        {"layout_files", collectAbsoluteFiles(processed.layoutDir)},
    };

    if (options.returnMd) {
        result["md_content"] = processed.result.markdown;
    }
    if (options.returnMiddleJson) {
        result["middle_json"] = processed.middleJson;
    }
    if (options.returnModelOutput) {
        result["model_output"] = processed.modelJson;
    }
    if (options.returnContentList) {
        result["content_list"] = processed.contentList;
    }
    if (options.returnImages) {
        result["images"] = collectImagesAsDataUrls(processed.imagesDir);
    }
    if (!processed.warnings.empty()) {
        result["warnings"] = processed.warnings;
    }

    return result;
}

json buildHealthJson() {
    json envVars{
        {"CUSTOM_INTER_OP_THREADS_COUNT", getenvOrNull("CUSTOM_INTER_OP_THREADS_COUNT")},
        {"CUSTOM_INTRA_OP_THREADS_COUNT", getenvOrNull("CUSTOM_INTRA_OP_THREADS_COUNT")},
        {"DXRT_DYNAMIC_CPU_THREAD", getenvOrNull("DXRT_DYNAMIC_CPU_THREAD")},
        {"DXRT_TASK_MAX_LOAD", getenvOrNull("DXRT_TASK_MAX_LOAD")},
        {"NFH_INPUT_WORKER_THREADS", getenvOrNull("NFH_INPUT_WORKER_THREADS")},
        {"NFH_OUTPUT_WORKER_THREADS", getenvOrNull("NFH_OUTPUT_WORKER_THREADS")},
    };

    return json{
        {"status", "healthy"},
        {"version", "0.1.0-cpp"},
        {"api", "RapidDoc Offline API (C++/Crow)"},
        {"mode", "closed_environment"},
        {"default_engines", makeEngineJson(true, true)},
        {"environment_variables", envVars},
    };
}

FileParseOptions parseFileParseOptions(const crow::multipart::message& msg) {
    FileParseOptions options;
    options.outputDir = getMultipartField(msg, "output_dir", options.outputDir);
    options.clearOutputFile = parseBool(getMultipartField(msg, "clear_output_file"), false);
    const auto langList = getMultipartFieldValues(msg, "lang_list");
    if (!langList.empty()) {
        options.langList = langList;
    }
    options.backend = getMultipartField(msg, "backend", options.backend);
    options.parseMethod = canonicalParseMethod(
        getMultipartField(msg, "parse_method", options.parseMethod));
    options.formulaEnable = parseBool(
        getMultipartField(msg, "formula_enable"), options.formulaEnable);
    options.tableEnable = parseBool(
        getMultipartField(msg, "table_enable"), options.tableEnable);
    options.deepxRequested = parseBool(getMultipartField(msg, "deepx"), true);
    options.layoutEngine = getMultipartField(msg, "layout_engine", options.layoutEngine);
    options.ocrEngine = getMultipartField(msg, "ocr_engine", options.ocrEngine);
    options.formulaEngine = getMultipartField(msg, "formula_engine", options.formulaEngine);
    options.tableEngine = getMultipartField(msg, "table_engine", options.tableEngine);
    options.returnMd = parseBool(getMultipartField(msg, "return_md"), true);
    options.returnMiddleJson = parseBool(getMultipartField(msg, "return_middle_json"), false);
    options.returnModelOutput = parseBool(getMultipartField(msg, "return_model_output"), false);
    options.returnContentList = parseBool(getMultipartField(msg, "return_content_list"), false);
    options.returnImages = parseBool(getMultipartField(msg, "return_images"), false);
    options.startPageId = parseInt(getMultipartField(msg, "start_page_id"), 0);
    options.endPageId = parseInt(getMultipartField(msg, "end_page_id"), 99999);
    options.pipelineMode = toLower(getMultipartField(msg, "pipeline_mode", options.pipelineMode));
    options.ocrOuterMode =
        toLower(getMultipartField(msg, "ocr_outer_mode", options.ocrOuterMode));
    const std::string rawShadowWindow = getMultipartField(msg, "ocr_shadow_window");
    if (!rawShadowWindow.empty()) {
        const int parsedShadowWindow = parseInt(rawShadowWindow, 0);
        options.ocrShadowWindow =
            parsedShadowWindow > 0 ? static_cast<size_t>(parsedShadowWindow) : 1;
    }
    return options;
}

} // namespace

DocServer::DocServer(const ServerConfig& config)
    : config_(config)
{
    fs::create_directories(config_.uploadDir);

    std::string topology = toLower(config_.topology);
    if (topology.empty()) {
        topology = "single_pipeline";
    }
    if (topology == "single_pipeline" && config_.deviceIds.size() > 1) {
        topology = "single_process_multi_device";
    }
    config_.topology = topology;
    config_.serverId = makeServerId(config_, topology);

    std::vector<int> shardDeviceIds = config_.deviceIds;
    if (config_.topology == "single_process_multi_device") {
        if (shardDeviceIds.empty()) {
            shardDeviceIds = discoverDxrtDeviceIds();
        }
        if (shardDeviceIds.empty()) {
            shardDeviceIds.push_back(config_.pipelineConfig.runtime.deviceId);
        }
    } else {
        if (shardDeviceIds.empty()) {
            shardDeviceIds.push_back(config_.pipelineConfig.runtime.deviceId);
        } else {
            shardDeviceIds.resize(1);
        }
    }

    if (shardDeviceIds.empty()) {
        shardDeviceIds.push_back(-1);
    }

    std::sort(shardDeviceIds.begin(), shardDeviceIds.end());
    shardDeviceIds.erase(
        std::unique(shardDeviceIds.begin(), shardDeviceIds.end()),
        shardDeviceIds.end());

    for (size_t i = 0; i < shardDeviceIds.size(); ++i) {
        auto shard = std::make_unique<PipelineShard>();
        shard->deviceId = shardDeviceIds[i];
        shard->shardId = "shard_" + std::to_string(i);

        PipelineConfig shardConfig = config_.pipelineConfig;
        shardConfig.runtime.deviceId = shard->deviceId;
        shard->pipeline = std::make_unique<DocPipeline>(shardConfig);
        shard->pipeline->externalNpuSerialMutex_ = &shard->npuSerialMutex;
        if (!shard->pipeline->initialize()) {
            throw std::runtime_error(
                "Failed to initialize document pipeline for " + shard->shardId);
        }
        shards_.push_back(std::move(shard));
    }

    std::vector<int> telemetryDeviceIds;
    for (const auto& shard : shards_) {
        if (shard->deviceId >= 0) {
            telemetryDeviceIds.push_back(shard->deviceId);
        }
    }
    if (!telemetryDeviceIds.empty()) {
        deviceMetricsSampler_ = std::make_unique<DeviceMetricsSampler>(telemetryDeviceIds);
    }
}

DocServer::~DocServer() {
    stop();
}

size_t DocServer::selectShardIndex() {
    if (shards_.empty()) {
        throw std::runtime_error("No pipeline shards configured");
    }

    const size_t shardCount = shards_.size();
    const size_t start = static_cast<size_t>(
        routingCursor_.fetch_add(1, std::memory_order_relaxed) % shardCount);

    size_t bestIndex = start;
    uint64_t bestInflight = shards_[start]->inflight.load(std::memory_order_relaxed);
    for (size_t offset = 1; offset < shardCount; ++offset) {
        const size_t idx = (start + offset) % shardCount;
        const uint64_t inflight = shards_[idx]->inflight.load(std::memory_order_relaxed);
        if (inflight < bestInflight) {
            bestInflight = inflight;
            bestIndex = idx;
        }
    }
    return bestIndex;
}

std::string DocServer::resolvedTopology() const {
    if (!config_.topology.empty()) {
        return config_.topology;
    }
    return (shards_.size() > 1) ? "single_process_multi_device" : "single_pipeline";
}

void DocServer::run() {
    LOG_INFO("Starting RapidDoc HTTP server on {}:{}", config_.host, config_.port);
    const int httpConcurrency = effectiveHttpIngressConcurrency(
        resolvedTopology(),
        shards_.size(),
        config_.numWorkers);
    LOG_INFO(
        "HTTP ingress concurrency: {} (worker_count={}, topology={}, shard_count={})",
        httpConcurrency,
        config_.numWorkers,
        resolvedTopology(),
        shards_.size());

    crow::SimpleApp app;
    if (deviceMetricsSampler_) {
        deviceMetricsSampler_->start();
    }

    auto executeDocument = [this](
        const std::string& bytes,
        const std::string& filename,
        const FileParseOptions& options) -> RoutedProcessedDocument
    {
        const size_t shardIndex = selectShardIndex();
        auto& shard = *shards_.at(shardIndex);

        DispatchMetadata dispatch;
        dispatch.topology = resolvedTopology();
        dispatch.deviceId = shard.deviceId;
        dispatch.shardId = shard.shardId;
        dispatch.backendId =
            (dispatch.topology == "single_card_backend") ? config_.serverId : std::string();

        shard.inflight.fetch_add(1, std::memory_order_relaxed);
        try {
            const auto queueStart = std::chrono::steady_clock::now();
            std::unique_lock<std::mutex> shardLock(shard.requestMutex);
            const auto shardAcquired = std::chrono::steady_clock::now();
            dispatch.routeQueueMs =
                std::chrono::duration<double, std::milli>(shardAcquired - queueStart).count();

            RoutedProcessedDocument routed;
            routed.dispatch = dispatch;
            routed.processed = processDocumentBytes(*shard.pipeline, bytes, filename, options);

            const auto shardDone = std::chrono::steady_clock::now();
            shard.routeQueueUsTotal.fetch_add(
                msToUs(routed.dispatch.routeQueueMs),
                std::memory_order_relaxed);
            shard.busyUsTotal.fetch_add(
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                    shardDone - shardAcquired).count()),
                std::memory_order_relaxed);
            shard.npuBusyUsTotal.fetch_add(
                msToUs(routed.processed.result.stats.npuSerialTimeMs),
                std::memory_order_relaxed);
            shard.requestCount.fetch_add(1, std::memory_order_relaxed);
            recordPipelineStats(
                routed.processed.result,
                routed.processed.pipelineCallTimeMs);
            shard.inflight.fetch_sub(1, std::memory_order_relaxed);
            return routed;
        } catch (...) {
            shard.inflight.fetch_sub(1, std::memory_order_relaxed);
            throw;
        }
    };

    CROW_ROUTE(app, "/health")
    ([this]() {
        crow::response resp(200, buildHealthJson().dump());
        resp.set_header("Content-Type", "application/json");
        return resp;
    });

    CROW_ROUTE(app, "/status")
    ([this]() {
        crow::response resp(200, buildStatusJson());
        resp.set_header("Content-Type", "application/json");
        return resp;
    });

    CROW_ROUTE(app, "/process").methods("POST"_method)
    ([this, &executeDocument](const crow::request& req) {
        requestCount_++;

        try {
            const auto contentType = req.get_header_value("Content-Type");
            if (contentType.find("multipart/form-data") == std::string::npos) {
                errorCount_++;
                return crow::response(400, R"({"error":"Expected multipart/form-data"})");
            }

            crow::multipart::message msg(req);
            auto parts = getMultipartParts(msg, "file");
            if (parts.empty()) {
                parts = getMultipartParts(msg, "files");
            }
            if (parts.empty()) {
                errorCount_++;
                return crow::response(400, R"({"error":"No file field found"})");
            }

            const auto& filePart = parts.front();
            std::string filename = "upload.pdf";
            const auto disposition = filePart.get_header_object("Content-Disposition");
            const auto filenameIt = disposition.params.find("filename");
            if (filenameIt != disposition.params.end()) {
                filename = filenameIt->second;
            }

            FileParseOptions options;
            options.outputDir = fs::path(config_.uploadDir) / "legacy";
            options.returnMd = true;
            options.returnContentList = true;
            options.clearOutputFile = true;
            options.pipelineMode = toLower(getMultipartField(msg, "pipeline_mode"));
            options.ocrOuterMode = toLower(getMultipartField(msg, "ocr_outer_mode"));
            const std::string rawShadowWindow = getMultipartField(msg, "ocr_shadow_window");
            if (!rawShadowWindow.empty()) {
                const int parsedShadowWindow = parseInt(rawShadowWindow, 0);
                options.ocrShadowWindow =
                    parsedShadowWindow > 0 ? static_cast<size_t>(parsedShadowWindow) : 1;
            }

            const RoutedProcessedDocument routed = executeDocument(filePart.body, filename, options);
            const auto& processed = routed.processed;
            json legacyResponse{
                {"pages", processed.result.processedPages},
                {"total_pages", processed.result.totalPages},
                {"skipped", processed.result.skippedElements},
                {"time_ms", processed.result.totalTimeMs},
                {"topology", routed.dispatch.topology},
                {"device_id", routed.dispatch.deviceId},
                {"shard_id", routed.dispatch.shardId},
                {"backend_id", routed.dispatch.backendId},
                {"stats", makeStatsJson(
                    processed.result,
                    processed.pipelineCallTimeMs,
                    routed.dispatch.routeQueueMs,
                    routed.dispatch.lbProxyMs)},
                {"markdown", processed.result.markdown},
                {"content_list", processed.contentList},
                {"output_dir", processed.parseDir.string()},
            };
            fs::remove_all(processed.requestDir);

            successCount_++;
            crow::response resp(200, legacyResponse.dump());
            resp.set_header("Content-Type", "application/json");
            return resp;
        }
        catch (const std::exception& e) {
            errorCount_++;
            LOG_ERROR("Processing error: {}", e.what());
            return crow::response(500, json{{"error", e.what()}}.dump());
        }
    });

    CROW_ROUTE(app, "/process/base64").methods("POST"_method)
    ([this, &executeDocument](const crow::request& req) {
        requestCount_++;

        try {
            json requestBody = json::parse(req.body);
            if (!requestBody.contains("data") || !requestBody["data"].is_string()) {
                errorCount_++;
                return crow::response(400, R"({"error":"Missing 'data' field"})");
            }

            const auto decoded = base64Decode(requestBody["data"].get<std::string>());
            if (decoded.empty()) {
                errorCount_++;
                return crow::response(400, R"({"error":"Invalid base64 data"})");
            }

            const std::string filename = requestBody.value("filename", "upload.pdf");
            FileParseOptions options;
            options.outputDir = fs::path(config_.uploadDir) / "legacy";
            options.returnMd = true;
            options.returnContentList = true;
            options.clearOutputFile = true;
            options.pipelineMode = toLower(requestBody.value("pipeline_mode", std::string()));
            options.ocrOuterMode = toLower(requestBody.value("ocr_outer_mode", std::string()));
            if (requestBody.contains("ocr_shadow_window")) {
                const int parsedShadowWindow = requestBody.value("ocr_shadow_window", 0);
                options.ocrShadowWindow =
                    parsedShadowWindow > 0 ? static_cast<size_t>(parsedShadowWindow) : 1;
            }

            const RoutedProcessedDocument routed = executeDocument(
                std::string(reinterpret_cast<const char*>(decoded.data()), decoded.size()),
                filename,
                options);
            const auto& processed = routed.processed;
            json legacyResponse{
                {"pages", processed.result.processedPages},
                {"total_pages", processed.result.totalPages},
                {"skipped", processed.result.skippedElements},
                {"time_ms", processed.result.totalTimeMs},
                {"topology", routed.dispatch.topology},
                {"device_id", routed.dispatch.deviceId},
                {"shard_id", routed.dispatch.shardId},
                {"backend_id", routed.dispatch.backendId},
                {"stats", makeStatsJson(
                    processed.result,
                    processed.pipelineCallTimeMs,
                    routed.dispatch.routeQueueMs,
                    routed.dispatch.lbProxyMs)},
                {"markdown", processed.result.markdown},
                {"content_list", processed.contentList},
                {"output_dir", processed.parseDir.string()},
            };
            fs::remove_all(processed.requestDir);

            successCount_++;
            crow::response resp(200, legacyResponse.dump());
            resp.set_header("Content-Type", "application/json");
            return resp;
        }
        catch (const std::exception& e) {
            errorCount_++;
            LOG_ERROR("Base64 processing error: {}", e.what());
            return crow::response(500, json{{"error", e.what()}}.dump());
        }
    });

    CROW_ROUTE(app, "/file_parse").methods("POST"_method)
    ([this, &executeDocument](const crow::request& req) {
        requestCount_++;

        try {
            const auto contentType = req.get_header_value("Content-Type");
            if (contentType.find("multipart/form-data") == std::string::npos) {
                errorCount_++;
                return crow::response(400, R"({"error":"Expected multipart/form-data"})");
            }

            crow::multipart::message msg(req);
            auto fileParts = getMultipartParts(msg, "files");
            if (fileParts.empty()) {
                fileParts = getMultipartParts(msg, "file");
            }
            if (fileParts.empty()) {
                errorCount_++;
                return crow::response(400, R"({"error":"No files provided"})");
            }

            FileParseOptions options = parseFileParseOptions(msg);
            json results = json::array();
            int successFiles = 0;
            const auto requestWarnings = collectRequestWarnings(options);

            for (const auto& part : fileParts) {
                std::string filename = "upload.bin";
                const auto disposition = part.get_header_object("Content-Disposition");
                const auto filenameIt = disposition.params.find("filename");
                if (filenameIt != disposition.params.end()) {
                    filename = filenameIt->second;
                }

                try {
                    RoutedProcessedDocument routed = executeDocument(part.body, filename, options);
                    json fileResult = buildFileResult(routed.processed, options, routed.dispatch);
                    if (!requestWarnings.empty()) {
                        fileResult["request_warnings"] = requestWarnings;
                    }
                    results.push_back(std::move(fileResult));
                    successFiles++;

                    if (options.clearOutputFile) {
                        fs::remove_all(routed.processed.requestDir);
                    }
                }
                catch (const std::exception& e) {
                    results.push_back(json{
                        {"filename", safeFilename(filename)},
                        {"error", e.what()},
                    });
                }
            }

            json responseData{
                {"results", std::move(results)},
                {"total_files", static_cast<int>(fileParts.size())},
                {"successful_files", successFiles},
                {"mode", "closed_environment"},
                {"deepx", true},
                {"engines_used", makeEngineJson(options.tableEnable, options.formulaEnable)},
            };
            if (!requestWarnings.empty()) {
                responseData["warnings"] = requestWarnings;
            }

            successCount_++;
            crow::response resp(200, responseData.dump());
            resp.set_header("Content-Type", "application/json");
            return resp;
        }
        catch (const std::exception& e) {
            errorCount_++;
            LOG_ERROR("file_parse error: {}", e.what());
            return crow::response(500, json{{"error", e.what()}}.dump());
        }
    });

    CROW_ROUTE(app, "/v1/images:annotate").methods("POST"_method)
    ([this, &executeDocument](const crow::request& req) {
        requestCount_++;

        try {
            json requestBody = json::parse(req.body);
            if (!requestBody.contains("requests") || !requestBody["requests"].is_array()) {
                errorCount_++;
                return crow::response(400, R"({"error":{"code":400,"message":"Missing requests array","status":"INVALID_ARGUMENT"}})");
            }

            const bool globalDeepx = requestBody.value("deepx", true);
            json responses = json::array();

            size_t index = 0;
            for (const auto& requestItem : requestBody["requests"]) {
                try {
                    std::string imageBytes;
                    std::string imageName = "image_" + std::to_string(index++) + ".png";

                    if (requestItem.contains("image") &&
                        requestItem["image"].contains("content") &&
                        requestItem["image"]["content"].is_string()) {
                        const auto decoded = base64Decode(requestItem["image"]["content"].get<std::string>());
                        imageBytes.assign(reinterpret_cast<const char*>(decoded.data()), decoded.size());
                    } else if (requestItem.contains("image") &&
                               requestItem["image"].contains("source") &&
                               requestItem["image"]["source"].contains("imageUri") &&
                               requestItem["image"]["source"]["imageUri"].is_string()) {
                        const std::string uri = requestItem["image"]["source"]["imageUri"].get<std::string>();
                        if (uri.rfind("http://", 0) == 0 || uri.rfind("https://", 0) == 0) {
                            responses.push_back(json{
                                {"error", {
                                    {"code", 403},
                                    {"message", "Network access blocked in closed environment. Please use base64 content or a local path."},
                                    {"status", "PERMISSION_DENIED"},
                                }},
                            });
                            continue;
                        }
                        if (!fs::exists(uri)) {
                            responses.push_back(json{
                                {"error", {
                                    {"code", 400},
                                    {"message", "Invalid imageUri: " + uri},
                                    {"status", "INVALID_ARGUMENT"},
                                }},
                            });
                            continue;
                        }
                        imageBytes = readBinaryFile(uri);
                        imageName = fs::path(uri).filename().string();
                    } else {
                        responses.push_back(json{
                            {"error", {
                                {"code", 400},
                                {"message", "Either image.content or image.source.imageUri must be provided"},
                                {"status", "INVALID_ARGUMENT"},
                            }},
                        });
                        continue;
                    }

                    bool documentTextDetection = false;
                    bool textDetection = false;
                    if (requestItem.contains("features") && requestItem["features"].is_array()) {
                        for (const auto& feature : requestItem["features"]) {
                            const std::string type = feature.value("type", "");
                            documentTextDetection = documentTextDetection || (type == "DOCUMENT_TEXT_DETECTION");
                            textDetection = textDetection || (type == "TEXT_DETECTION");
                        }
                    }

                    if (!documentTextDetection && !textDetection) {
                        responses.push_back(json{
                            {"error", {
                                {"code", 400},
                                {"message", "Unsupported feature types. Supported: TEXT_DETECTION, DOCUMENT_TEXT_DETECTION"},
                                {"status", "INVALID_ARGUMENT"},
                            }},
                        });
                        continue;
                    }

                    FileParseOptions options;
                    options.outputDir = fs::path(config_.uploadDir) / "annotate";
                    options.parseMethod = "auto";
                    options.formulaEnable = documentTextDetection;
                    options.tableEnable = documentTextDetection;
                    options.returnMd = true;
                    options.returnMiddleJson = documentTextDetection;
                    options.returnContentList = documentTextDetection;
                    options.returnModelOutput = false;
                    options.returnImages = false;
                    options.clearOutputFile = true;
                    options.deepxRequested = requestItem.value("deepx", globalDeepx);

                    const RoutedProcessedDocument routed = executeDocument(
                        imageBytes, imageName, options);
                    const auto& processed = routed.processed;

                    json responseItem;
                    if (textDetection) {
                        responseItem["textAnnotations"] = json::array({
                            {
                                {"description", processed.result.markdown},
                                {"locale", "auto"},
                            }
                        });
                    }
                    responseItem["fullTextAnnotation"] = json{
                        {"text", processed.result.markdown},
                    };

                    if (documentTextDetection) {
                        responseItem["middleJson"] = processed.middleJson;
                        responseItem["contentList"] = processed.contentList;
                    }

                    responseItem["topology"] = routed.dispatch.topology;
                    responseItem["device_id"] = routed.dispatch.deviceId;
                    responseItem["shard_id"] = routed.dispatch.shardId;
                    responseItem["backend_id"] = routed.dispatch.backendId;
                    responseItem["stats"] = makeStatsJson(
                        processed.result,
                        processed.pipelineCallTimeMs,
                        routed.dispatch.routeQueueMs,
                        routed.dispatch.lbProxyMs);

                    responses.push_back(std::move(responseItem));
                    fs::remove_all(processed.requestDir);
                }
                catch (const std::exception& e) {
                    responses.push_back(json{
                        {"error", {
                            {"code", 500},
                            {"message", e.what()},
                            {"status", "INTERNAL"},
                        }},
                    });
                }
            }

            successCount_++;
            crow::response resp(200, json{{"responses", std::move(responses)}}.dump());
            resp.set_header("Content-Type", "application/json");
            return resp;
        }
        catch (const std::exception& e) {
            errorCount_++;
            LOG_ERROR("images:annotate error: {}", e.what());
            return crow::response(
                500,
                json{{"error", {{"code", 500}, {"message", e.what()}, {"status", "INTERNAL"}}}}.dump());
        }
    });

    running_ = true;
    app.bindaddr(config_.host)
       .port(config_.port)
       .concurrency(static_cast<uint16_t>(httpConcurrency))
       .run();
    running_ = false;
}

void DocServer::stop() {
    running_ = false;
    if (deviceMetricsSampler_) {
        deviceMetricsSampler_->stop();
    }
    LOG_INFO("RapidDoc HTTP server stopped");
}

std::string DocServer::handleProcess(const std::string& pdfData, const std::string& filename) {
    FileParseOptions options;
    options.outputDir = fs::path(config_.uploadDir) / "legacy";
    options.returnMd = true;
    options.returnContentList = true;
    options.clearOutputFile = true;

    const size_t shardIndex = selectShardIndex();
    auto& shard = *shards_.at(shardIndex);
    DispatchMetadata dispatch;
    dispatch.topology = resolvedTopology();
    dispatch.deviceId = shard.deviceId;
    dispatch.shardId = shard.shardId;
    dispatch.backendId =
        (dispatch.topology == "single_card_backend") ? config_.serverId : std::string();

    shard.inflight.fetch_add(1, std::memory_order_relaxed);
    RoutedProcessedDocument routed;
    try {
        const auto queueStart = std::chrono::steady_clock::now();
        std::unique_lock<std::mutex> lock(shard.requestMutex);
        const auto shardAcquired = std::chrono::steady_clock::now();
        dispatch.routeQueueMs =
            std::chrono::duration<double, std::milli>(shardAcquired - queueStart).count();
        routed.dispatch = dispatch;
        routed.processed = processDocumentBytes(*shard.pipeline, pdfData, filename, options);
        const auto shardDone = std::chrono::steady_clock::now();
        shard.routeQueueUsTotal.fetch_add(
            msToUs(dispatch.routeQueueMs),
            std::memory_order_relaxed);
        shard.busyUsTotal.fetch_add(
            static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                shardDone - shardAcquired).count()),
            std::memory_order_relaxed);
        shard.npuBusyUsTotal.fetch_add(
            msToUs(routed.processed.result.stats.npuSerialTimeMs),
            std::memory_order_relaxed);
        shard.requestCount.fetch_add(1, std::memory_order_relaxed);
        recordPipelineStats(
            routed.processed.result,
            routed.processed.pipelineCallTimeMs);
        shard.inflight.fetch_sub(1, std::memory_order_relaxed);
    } catch (...) {
        shard.inflight.fetch_sub(1, std::memory_order_relaxed);
        throw;
    }

    const auto& processed = routed.processed;

    json response{
        {"pages", processed.result.processedPages},
        {"total_pages", processed.result.totalPages},
        {"skipped", processed.result.skippedElements},
        {"time_ms", processed.result.totalTimeMs},
        {"topology", routed.dispatch.topology},
        {"device_id", routed.dispatch.deviceId},
        {"shard_id", routed.dispatch.shardId},
        {"backend_id", routed.dispatch.backendId},
        {"stats", makeStatsJson(
            processed.result,
            processed.pipelineCallTimeMs,
            routed.dispatch.routeQueueMs,
            routed.dispatch.lbProxyMs)},
        {"markdown", processed.result.markdown},
        {"content_list", processed.contentList},
        {"output_dir", processed.parseDir.string()},
    };

    fs::remove_all(processed.requestDir);
    return response.dump();
}

std::string DocServer::buildStatusJson() {
    const uint64_t samples = lockSamples_.load(std::memory_order_relaxed);
    const uint64_t waitUsTotal = lockWaitUsTotal_.load(std::memory_order_relaxed);
    const uint64_t holdUsTotal = lockHoldUsTotal_.load(std::memory_order_relaxed);
    const double waitAvgMs = samples == 0 ? 0.0 :
        static_cast<double>(waitUsTotal) / static_cast<double>(samples) / 1000.0;
    const double holdAvgMs = samples == 0 ? 0.0 :
        static_cast<double>(holdUsTotal) / static_cast<double>(samples) / 1000.0;

    std::unordered_map<int, DeviceMetricSample> metricsByDevice;
    std::string memoryTelemetryStatus = "blocked_memory_telemetry_unavailable";
    if (deviceMetricsSampler_) {
        memoryTelemetryStatus = deviceMetricsSampler_->memoryTelemetryStatus();
        for (const auto& sample : deviceMetricsSampler_->snapshot()) {
            metricsByDevice[sample.deviceId] = sample;
        }
    }

    std::vector<double> activeRequestCounts;
    std::vector<double> activeBusyMs;
    for (const auto& shard : shards_) {
        const double requests =
            static_cast<double>(shard->requestCount.load(std::memory_order_relaxed));
        const double busyMs =
            static_cast<double>(shard->busyUsTotal.load(std::memory_order_relaxed)) / 1000.0;
        if (requests > 0.0 || busyMs > 0.0) {
            activeRequestCounts.push_back(requests);
            activeBusyMs.push_back(busyMs);
        }
    }

    bool loadImbalance = false;
    auto maxMinRatio = [](const std::vector<double>& values) -> double {
        if (values.size() < 2) {
            return 1.0;
        }
        const auto [minIt, maxIt] = std::minmax_element(values.begin(), values.end());
        if (*minIt <= 0.0) {
            return (*maxIt > 0.0) ? std::numeric_limits<double>::infinity() : 1.0;
        }
        return *maxIt / *minIt;
    };
    loadImbalance = maxMinRatio(activeRequestCounts) > 1.20 || maxMinRatio(activeBusyMs) > 1.20;

    json perDevice = json::array();
    json configuredDeviceIds = json::array();
    for (const auto& shard : shards_) {
        configuredDeviceIds.push_back(shard->deviceId);
        const uint64_t requestCount = shard->requestCount.load(std::memory_order_relaxed);
        const uint64_t busyUs = shard->busyUsTotal.load(std::memory_order_relaxed);
        const uint64_t npuBusyUs = shard->npuBusyUsTotal.load(std::memory_order_relaxed);
        const uint64_t routeQueueUs = shard->routeQueueUsTotal.load(std::memory_order_relaxed);
        const auto metricsIt = metricsByDevice.find(shard->deviceId);

        json item{
            {"device_id", shard->deviceId},
            {"shard_id", shard->shardId},
            {"request_count", requestCount},
            {"inflight", shard->inflight.load(std::memory_order_relaxed)},
            {"busy_time_ms", static_cast<double>(busyUs) / 1000.0},
            {"npu_busy_time_ms", static_cast<double>(npuBusyUs) / 1000.0},
            {"avg_infer_ms", requestCount == 0 ? 0.0 :
                static_cast<double>(npuBusyUs) / static_cast<double>(requestCount) / 1000.0},
            {"route_queue_total_ms", static_cast<double>(routeQueueUs) / 1000.0},
            {"memory_total_bytes", nullptr},
            {"memory_last_used_bytes", nullptr},
            {"memory_peak_used_bytes", nullptr},
            {"load_imbalance_flag", loadImbalance},
        };

        if (metricsIt != metricsByDevice.end()) {
            item["memory_total_bytes"] = metricsIt->second.memoryTotalBytes;
            item["memory_last_used_bytes"] = maybeUIntToJson(metricsIt->second.memoryLastUsedBytes);
            item["memory_peak_used_bytes"] = maybeUIntToJson(metricsIt->second.memoryPeakUsedBytes);
        }
        perDevice.push_back(std::move(item));
    }

    const uint64_t ocrSubmitCountTotal = ocrSubmitCountTotal_.load(std::memory_order_relaxed);
    const double ocrSubmitAreaSumTotal =
        static_cast<double>(ocrSubmitAreaSumTotal_.load(std::memory_order_relaxed));
    const double ocrSubmitAreaMeanTotal = ocrSubmitCountTotal == 0
        ? 0.0
        : ocrSubmitAreaSumTotal / static_cast<double>(ocrSubmitCountTotal);
    const double ocrSubmitAreaP50Total = ocrSubmitCountTotal == 0
        ? 0.0
        : static_cast<double>(ocrSubmitAreaP50WeightedTotal_.load(std::memory_order_relaxed)) /
            static_cast<double>(ocrSubmitCountTotal);
    const double ocrSubmitAreaP95Total = ocrSubmitCountTotal == 0
        ? 0.0
        : static_cast<double>(ocrSubmitAreaP95WeightedTotal_.load(std::memory_order_relaxed)) /
            static_cast<double>(ocrSubmitCountTotal);
    const double overlapFactorMean = samples == 0
        ? 0.0
        : static_cast<double>(
              pipelineOverlapFactorMilliTotal_.load(std::memory_order_relaxed)) /
            static_cast<double>(samples) / 1000.0;
    const double ocrInflightPeakMean = samples == 0
        ? 0.0
        : static_cast<double>(ocrInflightPeakTotal_.load(std::memory_order_relaxed)) /
            static_cast<double>(samples);
    const uint64_t serialModeCount = serialModeCount_.load(std::memory_order_relaxed);
    const uint64_t pagePipelineModeCount =
        pagePipelineModeCount_.load(std::memory_order_relaxed);
    std::string pipelineMode = pipelineModeToString(config_.pipelineConfig.runtime.pipelineMode);
    if (serialModeCount > 0 && pagePipelineModeCount > 0) {
        pipelineMode = "mixed";
    } else if (pagePipelineModeCount > 0) {
        pipelineMode = "page_pipeline_mvp";
    } else if (serialModeCount > 0) {
        pipelineMode = "serial";
    }

    json status{
        {"status", running_.load() ? "running" : "stopped"},
        {"requests", requestCount_.load()},
        {"success", successCount_.load()},
        {"errors", errorCount_.load()},
        {"topology", {
            {"mode", resolvedTopology()},
            {"server_id", config_.serverId},
            {"routing_policy", config_.routingPolicy},
            {"worker_count", config_.numWorkers},
            {"shard_count", shards_.size()},
            {"ocr_outer_mode", ocrOuterModeToString(config_.pipelineConfig.runtime.ocrOuterMode)},
            {"ocr_shadow_window", config_.pipelineConfig.runtime.ocrShadowWindow},
            {"configured_device_ids", configuredDeviceIds},
            {"memory_telemetry_status", memoryTelemetryStatus},
        }},
        {"per_device", std::move(perDevice)},
        {"pipeline_lock", {
            {"samples", samples},
            {"wait_total_ms", static_cast<double>(waitUsTotal) / 1000.0},
            {"wait_avg_ms", waitAvgMs},
            {"wait_max_ms", static_cast<double>(lockWaitUsMax_.load(std::memory_order_relaxed)) / 1000.0},
            {"hold_total_ms", static_cast<double>(holdUsTotal) / 1000.0},
            {"hold_avg_ms", holdAvgMs},
            {"hold_max_ms", static_cast<double>(lockHoldUsMax_.load(std::memory_order_relaxed)) / 1000.0},
        }},
        {"pipeline_stage_totals", {
            {"pipeline_call_ms", static_cast<double>(pipelineCallUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"pdf_render_ms", static_cast<double>(pdfRenderUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"layout_ms", static_cast<double>(layoutStageUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"ocr_ms", static_cast<double>(ocrStageUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"table_ms", static_cast<double>(tableStageUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"reading_order_ms", static_cast<double>(readingOrderStageUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"output_gen_ms", static_cast<double>(outputGenUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"npu_serial_ms", static_cast<double>(npuStageUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"cpu_only_ms", static_cast<double>(cpuStageUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"npu_lock_wait_ms", static_cast<double>(lockWaitUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"npu_lock_hold_ms", static_cast<double>(lockHoldUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"npu_service_ms", static_cast<double>(npuServiceUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"npu_slot_wait_ms", static_cast<double>(npuSlotWaitUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"layout_npu_service_ms", static_cast<double>(layoutNpuServiceUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"layout_npu_slot_wait_ms", static_cast<double>(layoutNpuSlotWaitUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"ocr_outer_slot_hold_ms", static_cast<double>(ocrOuterSlotHoldUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"ocr_submodule_window_ms", static_cast<double>(ocrSubmoduleWindowUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"ocr_slot_wait_ms", static_cast<double>(ocrSlotWaitUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"ocr_collect_wait_ms", static_cast<double>(ocrCollectWaitUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"ocr_inflight_peak", ocrInflightPeakMean},
            {"ocr_buffered_out_of_order_count", ocrBufferedOutOfOrderCountTotal_.load(std::memory_order_relaxed)},
            {"table_npu_service_ms", static_cast<double>(tableNpuServiceUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"table_npu_slot_wait_ms", static_cast<double>(tableNpuSlotWaitUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"table_ocr_service_ms", static_cast<double>(tableOcrServiceUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"table_ocr_slot_wait_ms", static_cast<double>(tableOcrSlotWaitUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"cpu_pre_ms", static_cast<double>(cpuPreUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"cpu_post_ms", static_cast<double>(cpuPostUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"finalize_cpu_ms", static_cast<double>(finalizeCpuUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"table_finalize_ms", static_cast<double>(tableFinalizeUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"ocr_collect_or_merge_ms", static_cast<double>(ocrCollectOrMergeUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"layout_queue_wait_ms", static_cast<double>(layoutQueueWaitUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"plan_queue_wait_ms", static_cast<double>(planQueueWaitUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"ocr_table_queue_wait_ms", static_cast<double>(ocrTableQueueWaitUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"finalize_queue_wait_ms", static_cast<double>(finalizeQueueWaitUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"render_queue_push_block_ms", static_cast<double>(renderQueuePushBlockUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"layout_queue_push_block_ms", static_cast<double>(layoutQueuePushBlockUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"plan_queue_push_block_ms", static_cast<double>(planQueuePushBlockUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"ocr_table_queue_push_block_ms", static_cast<double>(ocrTableQueuePushBlockUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"queue_backpressure_ms", static_cast<double>(queueBackpressureUsTotal_.load(std::memory_order_relaxed)) / 1000.0},
            {"pipeline_overlap_factor", overlapFactorMean},
            {"pipeline_mode", pipelineMode},
            {"text_boxes_raw_count", textBoxesRawCountTotal_.load(std::memory_order_relaxed)},
            {"text_boxes_after_dedup_count", textBoxesAfterDedupCountTotal_.load(std::memory_order_relaxed)},
            {"table_boxes_raw_count", tableBoxesRawCountTotal_.load(std::memory_order_relaxed)},
            {"table_boxes_after_dedup_count", tableBoxesAfterDedupCountTotal_.load(std::memory_order_relaxed)},
            {"ocr_submit_count", ocrSubmitCountTotal},
            {"ocr_submit_area_sum", ocrSubmitAreaSumTotal},
            {"ocr_submit_area_mean", ocrSubmitAreaMeanTotal},
            {"ocr_submit_area_p50", ocrSubmitAreaP50Total},
            {"ocr_submit_area_p95", ocrSubmitAreaP95Total},
            {"ocr_submit_small_count", ocrSubmitSmallCountTotal_.load(std::memory_order_relaxed)},
            {"ocr_submit_medium_count", ocrSubmitMediumCountTotal_.load(std::memory_order_relaxed)},
            {"ocr_submit_large_count", ocrSubmitLargeCountTotal_.load(std::memory_order_relaxed)},
            {"ocr_submit_text_count", ocrSubmitTextCountTotal_.load(std::memory_order_relaxed)},
            {"ocr_submit_title_count", ocrSubmitTitleCountTotal_.load(std::memory_order_relaxed)},
            {"ocr_submit_code_count", ocrSubmitCodeCountTotal_.load(std::memory_order_relaxed)},
            {"ocr_submit_list_count", ocrSubmitListCountTotal_.load(std::memory_order_relaxed)},
            {"ocr_dedup_skipped_count", ocrDedupSkippedCountTotal_.load(std::memory_order_relaxed)},
            {"table_npu_submit_count", tableNpuSubmitCountTotal_.load(std::memory_order_relaxed)},
            {"table_dedup_skipped_count", tableDedupSkippedCountTotal_.load(std::memory_order_relaxed)},
            {"ocr_timeout_count", ocrTimeoutCountTotal_.load(std::memory_order_relaxed)},
            {"ocr_buffered_result_hit_count", ocrBufferedResultHitCountTotal_.load(std::memory_order_relaxed)},
        }},
        {"engines", makeEngineJson(
            config_.pipelineConfig.stages.enableWiredTable,
            config_.pipelineConfig.stages.enableFormula)},
        {"capabilities", {
            {"layout", true},
            {"ocr", true},
            {"wired_table", true},
            {"wireless_table", false},
            {"formula_latex", false},
            {"formula_image_fallback", true},
            {"gradio_ui", true},
            {"file_parse_api", true},
            {"vision_annotate_api", true},
        }},
    };
    return status.dump();
}

void DocServer::recordPipelineStats(const DocumentResult& result, double pipelineCallMs) {
    const uint64_t waitUs = msToUs(result.stats.npuLockWaitTimeMs);
    const uint64_t holdUs = msToUs(result.stats.npuLockHoldTimeMs);
    const uint64_t pipelineCallUs = msToUs(pipelineCallMs);
    const uint64_t pdfRenderUs = msToUs(result.stats.pdfRenderTimeMs);
    const uint64_t layoutUs = msToUs(result.stats.layoutTimeMs);
    const uint64_t ocrUs = msToUs(result.stats.ocrTimeMs);
    const uint64_t tableUs = msToUs(result.stats.tableTimeMs);
    const uint64_t readingOrderUs = msToUs(result.stats.readingOrderTimeMs);
    const uint64_t outputGenUs = msToUs(result.stats.outputGenTimeMs);
    const uint64_t npuUs = msToUs(result.stats.npuSerialTimeMs);
    const uint64_t cpuUs = msToUs(result.stats.cpuOnlyTimeMs);
    const uint64_t npuServiceUs = msToUs(result.stats.npuServiceTimeMs);
    const uint64_t npuSlotWaitUs = msToUs(result.stats.npuSlotWaitTimeMs);
    const uint64_t layoutNpuServiceUs = msToUs(result.stats.layoutNpuServiceTimeMs);
    const uint64_t layoutNpuSlotWaitUs = msToUs(result.stats.layoutNpuSlotWaitTimeMs);
    const uint64_t ocrOuterSlotHoldUs = msToUs(result.stats.ocrOuterSlotHoldTimeMs);
    const uint64_t ocrSubmoduleWindowUs = msToUs(result.stats.ocrSubmoduleWindowTimeMs);
    const uint64_t ocrSlotWaitUs = msToUs(result.stats.ocrSlotWaitTimeMs);
    const uint64_t ocrCollectWaitUs = msToUs(result.stats.ocrCollectWaitTimeMs);
    const uint64_t ocrInflightPeak = countToUInt(result.stats.ocrInflightPeak);
    const uint64_t ocrBufferedOutOfOrderCount =
        countToUInt(result.stats.ocrBufferedOutOfOrderCount);
    const uint64_t tableNpuServiceUs = msToUs(result.stats.tableNpuServiceTimeMs);
    const uint64_t tableNpuSlotWaitUs = msToUs(result.stats.tableNpuSlotWaitTimeMs);
    const uint64_t tableOcrServiceUs = msToUs(result.stats.tableOcrServiceTimeMs);
    const uint64_t tableOcrSlotWaitUs = msToUs(result.stats.tableOcrSlotWaitTimeMs);
    const uint64_t cpuPreUs = msToUs(result.stats.cpuPreTimeMs);
    const uint64_t cpuPostUs = msToUs(result.stats.cpuPostTimeMs);
    const uint64_t finalizeCpuUs = msToUs(result.stats.finalizeCpuTimeMs);
    const uint64_t tableFinalizeUs = msToUs(result.stats.tableFinalizeTimeMs);
    const uint64_t ocrCollectOrMergeUs = msToUs(result.stats.ocrCollectOrMergeTimeMs);
    const uint64_t layoutQueueWaitUs = msToUs(result.stats.layoutQueueWaitTimeMs);
    const uint64_t planQueueWaitUs = msToUs(result.stats.planQueueWaitTimeMs);
    const uint64_t ocrTableQueueWaitUs = msToUs(result.stats.ocrTableQueueWaitTimeMs);
    const uint64_t finalizeQueueWaitUs = msToUs(result.stats.finalizeQueueWaitTimeMs);
    const uint64_t renderQueuePushBlockUs = msToUs(result.stats.renderQueuePushBlockTimeMs);
    const uint64_t layoutQueuePushBlockUs = msToUs(result.stats.layoutQueuePushBlockTimeMs);
    const uint64_t planQueuePushBlockUs = msToUs(result.stats.planQueuePushBlockTimeMs);
    const uint64_t ocrTableQueuePushBlockUs = msToUs(result.stats.ocrTableQueuePushBlockTimeMs);
    const uint64_t queueBackpressureUs = msToUs(result.stats.queueBackpressureTimeMs);
    const uint64_t overlapFactorMilli =
        countToUInt(result.stats.pipelineOverlapFactor * 1000.0);
    const uint64_t textBoxesRawCount = countToUInt(result.stats.textBoxesRawCount);
    const uint64_t textBoxesAfterDedupCount = countToUInt(result.stats.textBoxesAfterDedupCount);
    const uint64_t tableBoxesRawCount = countToUInt(result.stats.tableBoxesRawCount);
    const uint64_t tableBoxesAfterDedupCount = countToUInt(result.stats.tableBoxesAfterDedupCount);
    const uint64_t ocrSubmitCount = countToUInt(result.stats.ocrSubmitCount);
    const uint64_t ocrSubmitAreaSum = countToUInt(result.stats.ocrSubmitAreaSum);
    const uint64_t ocrSubmitAreaP50Weighted =
        countToUInt(result.stats.ocrSubmitAreaP50 * std::max(0.0, result.stats.ocrSubmitCount));
    const uint64_t ocrSubmitAreaP95Weighted =
        countToUInt(result.stats.ocrSubmitAreaP95 * std::max(0.0, result.stats.ocrSubmitCount));
    const uint64_t ocrSubmitSmallCount = countToUInt(result.stats.ocrSubmitSmallCount);
    const uint64_t ocrSubmitMediumCount = countToUInt(result.stats.ocrSubmitMediumCount);
    const uint64_t ocrSubmitLargeCount = countToUInt(result.stats.ocrSubmitLargeCount);
    const uint64_t ocrSubmitTextCount = countToUInt(result.stats.ocrSubmitTextCount);
    const uint64_t ocrSubmitTitleCount = countToUInt(result.stats.ocrSubmitTitleCount);
    const uint64_t ocrSubmitCodeCount = countToUInt(result.stats.ocrSubmitCodeCount);
    const uint64_t ocrSubmitListCount = countToUInt(result.stats.ocrSubmitListCount);
    const uint64_t ocrDedupSkippedCount = countToUInt(result.stats.ocrDedupSkippedCount);
    const uint64_t tableNpuSubmitCount = countToUInt(result.stats.tableNpuSubmitCount);
    const uint64_t tableDedupSkippedCount = countToUInt(result.stats.tableDedupSkippedCount);
    const uint64_t ocrTimeoutCount = countToUInt(result.stats.ocrTimeoutCount);
    const uint64_t ocrBufferedResultHitCount =
        countToUInt(result.stats.ocrBufferedResultHitCount);

    lockSamples_.fetch_add(1, std::memory_order_relaxed);
    lockWaitUsTotal_.fetch_add(waitUs, std::memory_order_relaxed);
    lockHoldUsTotal_.fetch_add(holdUs, std::memory_order_relaxed);
    pipelineCallUsTotal_.fetch_add(pipelineCallUs, std::memory_order_relaxed);
    pdfRenderUsTotal_.fetch_add(pdfRenderUs, std::memory_order_relaxed);
    layoutStageUsTotal_.fetch_add(layoutUs, std::memory_order_relaxed);
    ocrStageUsTotal_.fetch_add(ocrUs, std::memory_order_relaxed);
    tableStageUsTotal_.fetch_add(tableUs, std::memory_order_relaxed);
    readingOrderStageUsTotal_.fetch_add(readingOrderUs, std::memory_order_relaxed);
    outputGenUsTotal_.fetch_add(outputGenUs, std::memory_order_relaxed);
    npuStageUsTotal_.fetch_add(npuUs, std::memory_order_relaxed);
    cpuStageUsTotal_.fetch_add(cpuUs, std::memory_order_relaxed);
    npuServiceUsTotal_.fetch_add(npuServiceUs, std::memory_order_relaxed);
    npuSlotWaitUsTotal_.fetch_add(npuSlotWaitUs, std::memory_order_relaxed);
    layoutNpuServiceUsTotal_.fetch_add(layoutNpuServiceUs, std::memory_order_relaxed);
    layoutNpuSlotWaitUsTotal_.fetch_add(layoutNpuSlotWaitUs, std::memory_order_relaxed);
    ocrOuterSlotHoldUsTotal_.fetch_add(ocrOuterSlotHoldUs, std::memory_order_relaxed);
    ocrSubmoduleWindowUsTotal_.fetch_add(ocrSubmoduleWindowUs, std::memory_order_relaxed);
    ocrSlotWaitUsTotal_.fetch_add(ocrSlotWaitUs, std::memory_order_relaxed);
    ocrCollectWaitUsTotal_.fetch_add(ocrCollectWaitUs, std::memory_order_relaxed);
    ocrInflightPeakTotal_.fetch_add(ocrInflightPeak, std::memory_order_relaxed);
    ocrBufferedOutOfOrderCountTotal_.fetch_add(
        ocrBufferedOutOfOrderCount, std::memory_order_relaxed);
    tableNpuServiceUsTotal_.fetch_add(tableNpuServiceUs, std::memory_order_relaxed);
    tableNpuSlotWaitUsTotal_.fetch_add(tableNpuSlotWaitUs, std::memory_order_relaxed);
    tableOcrServiceUsTotal_.fetch_add(tableOcrServiceUs, std::memory_order_relaxed);
    tableOcrSlotWaitUsTotal_.fetch_add(tableOcrSlotWaitUs, std::memory_order_relaxed);
    cpuPreUsTotal_.fetch_add(cpuPreUs, std::memory_order_relaxed);
    cpuPostUsTotal_.fetch_add(cpuPostUs, std::memory_order_relaxed);
    finalizeCpuUsTotal_.fetch_add(finalizeCpuUs, std::memory_order_relaxed);
    tableFinalizeUsTotal_.fetch_add(tableFinalizeUs, std::memory_order_relaxed);
    ocrCollectOrMergeUsTotal_.fetch_add(ocrCollectOrMergeUs, std::memory_order_relaxed);
    layoutQueueWaitUsTotal_.fetch_add(layoutQueueWaitUs, std::memory_order_relaxed);
    planQueueWaitUsTotal_.fetch_add(planQueueWaitUs, std::memory_order_relaxed);
    ocrTableQueueWaitUsTotal_.fetch_add(ocrTableQueueWaitUs, std::memory_order_relaxed);
    finalizeQueueWaitUsTotal_.fetch_add(finalizeQueueWaitUs, std::memory_order_relaxed);
    renderQueuePushBlockUsTotal_.fetch_add(renderQueuePushBlockUs, std::memory_order_relaxed);
    layoutQueuePushBlockUsTotal_.fetch_add(layoutQueuePushBlockUs, std::memory_order_relaxed);
    planQueuePushBlockUsTotal_.fetch_add(planQueuePushBlockUs, std::memory_order_relaxed);
    ocrTableQueuePushBlockUsTotal_.fetch_add(
        ocrTableQueuePushBlockUs, std::memory_order_relaxed);
    queueBackpressureUsTotal_.fetch_add(queueBackpressureUs, std::memory_order_relaxed);
    pipelineOverlapFactorMilliTotal_.fetch_add(overlapFactorMilli, std::memory_order_relaxed);
    textBoxesRawCountTotal_.fetch_add(textBoxesRawCount, std::memory_order_relaxed);
    textBoxesAfterDedupCountTotal_.fetch_add(
        textBoxesAfterDedupCount, std::memory_order_relaxed);
    tableBoxesRawCountTotal_.fetch_add(tableBoxesRawCount, std::memory_order_relaxed);
    tableBoxesAfterDedupCountTotal_.fetch_add(
        tableBoxesAfterDedupCount, std::memory_order_relaxed);
    ocrSubmitCountTotal_.fetch_add(ocrSubmitCount, std::memory_order_relaxed);
    ocrSubmitAreaSumTotal_.fetch_add(ocrSubmitAreaSum, std::memory_order_relaxed);
    ocrSubmitAreaP50WeightedTotal_.fetch_add(ocrSubmitAreaP50Weighted, std::memory_order_relaxed);
    ocrSubmitAreaP95WeightedTotal_.fetch_add(ocrSubmitAreaP95Weighted, std::memory_order_relaxed);
    ocrSubmitSmallCountTotal_.fetch_add(ocrSubmitSmallCount, std::memory_order_relaxed);
    ocrSubmitMediumCountTotal_.fetch_add(ocrSubmitMediumCount, std::memory_order_relaxed);
    ocrSubmitLargeCountTotal_.fetch_add(ocrSubmitLargeCount, std::memory_order_relaxed);
    ocrSubmitTextCountTotal_.fetch_add(ocrSubmitTextCount, std::memory_order_relaxed);
    ocrSubmitTitleCountTotal_.fetch_add(ocrSubmitTitleCount, std::memory_order_relaxed);
    ocrSubmitCodeCountTotal_.fetch_add(ocrSubmitCodeCount, std::memory_order_relaxed);
    ocrSubmitListCountTotal_.fetch_add(ocrSubmitListCount, std::memory_order_relaxed);
    ocrDedupSkippedCountTotal_.fetch_add(ocrDedupSkippedCount, std::memory_order_relaxed);
    tableNpuSubmitCountTotal_.fetch_add(tableNpuSubmitCount, std::memory_order_relaxed);
    tableDedupSkippedCountTotal_.fetch_add(tableDedupSkippedCount, std::memory_order_relaxed);
    ocrTimeoutCountTotal_.fetch_add(ocrTimeoutCount, std::memory_order_relaxed);
    ocrBufferedResultHitCountTotal_.fetch_add(
        ocrBufferedResultHitCount, std::memory_order_relaxed);
    if (result.stats.pipelineMode == "page_pipeline_mvp") {
        pagePipelineModeCount_.fetch_add(1, std::memory_order_relaxed);
    } else {
        serialModeCount_.fetch_add(1, std::memory_order_relaxed);
    }
    updateAtomicMax(lockWaitUsMax_, waitUs);
    updateAtomicMax(lockHoldUsMax_, holdUs);
}

} // namespace rapid_doc
