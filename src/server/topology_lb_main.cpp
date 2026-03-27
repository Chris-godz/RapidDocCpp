#include <crow.h>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

using json = nlohmann::json;

namespace {

struct Backend {
    std::string id;
    std::string baseUrl;
    std::atomic<uint64_t> inflight{0};
};

struct CurlResponse {
    long httpCode = 0;
    std::string body;
    std::string error;
};

std::string normalizeBaseUrl(std::string url) {
    while (!url.empty() && url.back() == '/') {
        url.pop_back();
    }
    return url;
}

size_t writeCallback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    auto* out = static_cast<std::string*>(userdata);
    out->append(ptr, size * nmemb);
    return size * nmemb;
}

CurlResponse curlGetJson(const std::string& url) {
    CurlResponse response;
    CURL* curl = curl_easy_init();
    if (curl == nullptr) {
        response.error = "curl_init_failed";
        return response;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response.body);

    const CURLcode code = curl_easy_perform(curl);
    if (code != CURLE_OK) {
        response.error = curl_easy_strerror(code);
    }
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response.httpCode);
    curl_easy_cleanup(curl);
    return response;
}

CurlResponse forwardMultipartRequest(
    const crow::multipart::message& msg,
    const std::string& url)
{
    CurlResponse response;
    CURL* curl = curl_easy_init();
    if (curl == nullptr) {
        response.error = "curl_init_failed";
        return response;
    }

    curl_mime* mime = curl_mime_init(curl);
    for (const auto& entry : msg.part_map) {
        const auto& part = entry.second;
        curl_mimepart* mimePart = curl_mime_addpart(mime);
        curl_mime_name(mimePart, entry.first.c_str());
        curl_mime_data(mimePart, part.body.data(), part.body.size());

        const auto disposition = part.get_header_object("Content-Disposition");
        const auto filenameIt = disposition.params.find("filename");
        if (filenameIt != disposition.params.end()) {
            curl_mime_filename(mimePart, filenameIt->second.c_str());
        }

        const std::string contentType = part.get_header_object("Content-Type").value;
        if (!contentType.empty()) {
            curl_mime_type(mimePart, contentType.c_str());
        }
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_MIMEPOST, mime);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response.body);

    const CURLcode code = curl_easy_perform(curl);
    if (code != CURLE_OK) {
        response.error = curl_easy_strerror(code);
    }
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &response.httpCode);
    curl_mime_free(mime);
    curl_easy_cleanup(curl);
    return response;
}

std::vector<std::string> parseRepeatableBackendUrls(int argc, char* argv[]) {
    std::vector<std::string> urls;
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::string(argv[i]) == "--backend-url") {
            urls.push_back(normalizeBaseUrl(argv[i + 1]));
        }
    }
    return urls;
}

size_t selectBackendIndex(
    const std::vector<std::unique_ptr<Backend>>& backends,
    std::atomic<uint64_t>& cursor)
{
    const size_t count = backends.size();
    const size_t start = static_cast<size_t>(
        cursor.fetch_add(1, std::memory_order_relaxed) % count);

    size_t bestIndex = start;
    uint64_t bestInflight = backends[start]->inflight.load(std::memory_order_relaxed);
    for (size_t offset = 1; offset < count; ++offset) {
        const size_t idx = (start + offset) % count;
        const uint64_t inflight = backends[idx]->inflight.load(std::memory_order_relaxed);
        if (inflight < bestInflight) {
            bestInflight = inflight;
            bestIndex = idx;
        }
    }
    return bestIndex;
}

double maxMinRatio(const std::vector<double>& values) {
    if (values.size() < 2) {
        return 1.0;
    }
    const auto [minIt, maxIt] = std::minmax_element(values.begin(), values.end());
    if (*minIt <= 0.0) {
        return (*maxIt > 0.0) ? std::numeric_limits<double>::infinity() : 1.0;
    }
    return *maxIt / *minIt;
}

void recomputeLoadImbalance(json& perDevice) {
    std::vector<double> requestCounts;
    std::vector<double> busyTimes;
    for (const auto& item : perDevice) {
        const double requests = item.value("request_count", 0.0);
        const double busyMs = item.value("busy_time_ms", 0.0);
        if (requests > 0.0 || busyMs > 0.0) {
            requestCounts.push_back(requests);
            busyTimes.push_back(busyMs);
        }
    }

    const bool loadImbalance =
        maxMinRatio(requestCounts) > 1.20 || maxMinRatio(busyTimes) > 1.20;
    for (auto& item : perDevice) {
        item["load_imbalance_flag"] = loadImbalance;
    }
}

json aggregateStatus(
    const std::vector<std::unique_ptr<Backend>>& backends,
    const std::string& serverId,
    int workers,
    const std::string& routingPolicy)
{
    uint64_t requests = 0;
    uint64_t success = 0;
    uint64_t errors = 0;
    uint64_t lockSamples = 0;
    double waitTotalMs = 0.0;
    double holdTotalMs = 0.0;
    double waitMaxMs = 0.0;
    double holdMaxMs = 0.0;
    double npuSerialMs = 0.0;
    double cpuOnlyMs = 0.0;
    json perDevice = json::array();
    json backendStatuses = json::array();
    std::string memoryStatus = "blocked_memory_telemetry_unavailable";

    for (const auto& backend : backends) {
        const CurlResponse resp = curlGetJson(backend->baseUrl + "/status");
        if (!resp.error.empty() || resp.httpCode != 200) {
            backendStatuses.push_back(json{
                {"backend_id", backend->id},
                {"base_url", backend->baseUrl},
                {"status", "error"},
                {"error", resp.error.empty() ? ("http_" + std::to_string(resp.httpCode)) : resp.error},
            });
            continue;
        }

        const json payload = json::parse(resp.body, nullptr, false);
        if (payload.is_discarded()) {
            backendStatuses.push_back(json{
                {"backend_id", backend->id},
                {"base_url", backend->baseUrl},
                {"status", "error"},
                {"error", "invalid_json"},
            });
            continue;
        }

        requests += payload.value("requests", 0ULL);
        success += payload.value("success", 0ULL);
        errors += payload.value("errors", 0ULL);
        if (payload.contains("pipeline_lock")) {
            const auto& lock = payload["pipeline_lock"];
            lockSamples += lock.value("samples", 0ULL);
            waitTotalMs += lock.value("wait_total_ms", 0.0);
            holdTotalMs += lock.value("hold_total_ms", 0.0);
            waitMaxMs = std::max(waitMaxMs, lock.value("wait_max_ms", 0.0));
            holdMaxMs = std::max(holdMaxMs, lock.value("hold_max_ms", 0.0));
        }
        if (payload.contains("pipeline_stage_totals")) {
            const auto& stages = payload["pipeline_stage_totals"];
            npuSerialMs += stages.value("npu_serial_ms", 0.0);
            cpuOnlyMs += stages.value("cpu_only_ms", 0.0);
        }
        if (payload.contains("topology")) {
            memoryStatus = payload["topology"].value(
                "memory_telemetry_status",
                memoryStatus);
        }
        if (payload.contains("per_device") && payload["per_device"].is_array()) {
            for (auto item : payload["per_device"]) {
                item["backend_id"] = backend->id;
                perDevice.push_back(std::move(item));
            }
        }

        backendStatuses.push_back(json{
            {"backend_id", backend->id},
            {"base_url", backend->baseUrl},
            {"status", "ok"},
        });
    }

    recomputeLoadImbalance(perDevice);

    json configuredDeviceIds = json::array();
    for (const auto& item : perDevice) {
        configuredDeviceIds.push_back(item.value("device_id", -1));
    }

    const double waitAvgMs = lockSamples == 0 ? 0.0 : waitTotalMs / static_cast<double>(lockSamples);
    const double holdAvgMs = lockSamples == 0 ? 0.0 : holdTotalMs / static_cast<double>(lockSamples);

    return json{
        {"status", "running"},
        {"requests", requests},
        {"success", success},
        {"errors", errors},
        {"topology", {
            {"mode", "front_lb"},
            {"server_id", serverId},
            {"routing_policy", routingPolicy},
            {"worker_count", workers},
            {"shard_count", backends.size()},
            {"configured_device_ids", configuredDeviceIds},
            {"memory_telemetry_status", memoryStatus},
        }},
        {"per_device", perDevice},
        {"pipeline_lock", {
            {"samples", lockSamples},
            {"wait_total_ms", waitTotalMs},
            {"wait_avg_ms", waitAvgMs},
            {"wait_max_ms", waitMaxMs},
            {"hold_total_ms", holdTotalMs},
            {"hold_avg_ms", holdAvgMs},
            {"hold_max_ms", holdMaxMs},
        }},
        {"pipeline_stage_totals", {
            {"npu_serial_ms", npuSerialMs},
            {"cpu_only_ms", cpuOnlyMs},
        }},
        {"backends", backendStatuses},
    };
}

void printUsage(const char* programName) {
    std::cout << "RapidDoc Topology LB\n\n";
    std::cout << "Usage: " << programName << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  -H, --host <addr>       Bind address (default: 127.0.0.1)\n";
    std::cout << "  -p, --port <num>        Port number (default: 18880)\n";
    std::cout << "  -w, --workers <num>     Worker threads (default: 1)\n";
    std::cout << "      --backend-url <u>   Backend base URL (repeatable)\n";
    std::cout << "      --server-id <id>    Stable LB identifier\n";
    std::cout << "      --routing-policy <p> Routing policy (default: least_inflight_rr)\n";
    std::cout << "  -h, --help              Show this help\n";
}

} // namespace

int main(int argc, char* argv[]) {
    std::string host = "127.0.0.1";
    int port = 18880;
    int workers = 1;
    std::string serverId = "front_lb";
    std::string routingPolicy = "least_inflight_rr";
    std::vector<std::string> backendUrls = parseRepeatableBackendUrls(argc, argv);

    static const option longOpts[] = {
        {"host", required_argument, nullptr, 'H'},
        {"port", required_argument, nullptr, 'p'},
        {"workers", required_argument, nullptr, 'w'},
        {"backend-url", required_argument, nullptr, 256},
        {"server-id", required_argument, nullptr, 257},
        {"routing-policy", required_argument, nullptr, 258},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };

    int opt = 0;
    while ((opt = getopt_long(argc, argv, "H:p:w:h", longOpts, nullptr)) != -1) {
        switch (opt) {
            case 'H': host = optarg; break;
            case 'p': port = std::atoi(optarg); break;
            case 'w': workers = std::atoi(optarg); break;
            case 256: break;
            case 257: serverId = optarg; break;
            case 258: routingPolicy = optarg; break;
            case 'h': printUsage(argv[0]); return 0;
            default: printUsage(argv[0]); return 1;
        }
    }

    if (backendUrls.empty()) {
        std::cerr << "At least one --backend-url is required\n";
        return 1;
    }

    curl_global_init(CURL_GLOBAL_DEFAULT);

    std::vector<std::unique_ptr<Backend>> backends;
    backends.reserve(backendUrls.size());
    for (size_t i = 0; i < backendUrls.size(); ++i) {
        auto backend = std::make_unique<Backend>();
        backend->id = "backend_" + std::to_string(i);
        backend->baseUrl = normalizeBaseUrl(backendUrls[i]);
        backends.push_back(std::move(backend));
    }

    std::atomic<uint64_t> cursor{0};
    crow::SimpleApp app;

    CROW_ROUTE(app, "/health")
    ([&]() {
        json payload{
            {"status", "healthy"},
            {"topology", "front_lb"},
            {"server_id", serverId},
            {"backend_count", backends.size()},
        };
        crow::response resp(200, payload.dump());
        resp.set_header("Content-Type", "application/json");
        return resp;
    });

    CROW_ROUTE(app, "/status")
    ([&]() {
        const json payload = aggregateStatus(backends, serverId, workers, routingPolicy);
        crow::response resp(200, payload.dump());
        resp.set_header("Content-Type", "application/json");
        return resp;
    });

    CROW_ROUTE(app, "/file_parse").methods("POST"_method)
    ([&](const crow::request& req) {
        const auto contentType = req.get_header_value("Content-Type");
        if (contentType.find("multipart/form-data") == std::string::npos) {
            return crow::response(400, R"({"error":"Expected multipart/form-data"})");
        }

        crow::multipart::message msg(req);
        if (msg.part_map.empty()) {
            return crow::response(400, R"({"error":"No multipart parts provided"})");
        }

        const size_t backendIndex = selectBackendIndex(backends, cursor);
        auto& backend = *backends.at(backendIndex);
        backend.inflight.fetch_add(1, std::memory_order_relaxed);

        try {
            const auto proxyStart = std::chrono::steady_clock::now();
            CurlResponse backendResp = forwardMultipartRequest(
                msg,
                backend.baseUrl + "/file_parse");
            const auto proxyEnd = std::chrono::steady_clock::now();
            backend.inflight.fetch_sub(1, std::memory_order_relaxed);

            if (!backendResp.error.empty()) {
                return crow::response(
                    502,
                    json{{"error", "backend_proxy_failed"}, {"detail", backendResp.error}}.dump());
            }

            json payload = json::parse(backendResp.body, nullptr, false);
            if (payload.is_discarded()) {
                return crow::response(
                    502,
                    json{{"error", "backend_invalid_json"}, {"body", backendResp.body}}.dump());
            }

            const double lbProxyMs =
                std::chrono::duration<double, std::milli>(proxyEnd - proxyStart).count();
            if (payload.contains("results") && payload["results"].is_array()) {
                for (auto& item : payload["results"]) {
                    item["topology"] = "front_lb";
                    item["backend_id"] = backend.id;
                    if (item.contains("stats") && item["stats"].is_object()) {
                        item["stats"]["lb_proxy_ms"] = lbProxyMs;
                    }
                }
            }
            payload["topology"] = "front_lb";
            payload["backend_id"] = backend.id;
            payload["lb_server_id"] = serverId;

            crow::response resp(static_cast<int>(backendResp.httpCode), payload.dump());
            resp.set_header("Content-Type", "application/json");
            return resp;
        } catch (const std::exception& e) {
            backend.inflight.fetch_sub(1, std::memory_order_relaxed);
            return crow::response(
                500,
                json{{"error", "lb_internal_error"}, {"detail", e.what()}}.dump());
        }
    });

    app.bindaddr(host)
       .port(static_cast<uint16_t>(port))
       .concurrency(static_cast<uint16_t>(workers))
       .run();

    curl_global_cleanup();
    return 0;
}
