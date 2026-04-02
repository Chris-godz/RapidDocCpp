#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <array>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <future>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

using json = nlohmann::json;

namespace {

const std::string kNpuServerBinary =
    std::string(RAPIDDOC_BUILD_DIR) + "/bin/rapid_doc_npu_test_server";
const std::string kPdfA =
    std::string(PROJECT_ROOT_DIR) + "/test_files/small_ocr_origin.pdf";
const std::string kPdfB =
    std::string(PROJECT_ROOT_DIR) + "/test_files/BVRC_Meeting_Minutes_2024-04_origin.pdf";

struct CurlResponse {
    int exitCode = -1;
    int httpCode = 0;
    std::string body;
    std::string rawOutput;
};

std::string shellEscape(const std::string& input) {
    std::string escaped = "'";
    for (char ch : input) {
        if (ch == '\'') {
            escaped += "'\\''";
        } else {
            escaped.push_back(ch);
        }
    }
    escaped.push_back('\'');
    return escaped;
}

std::string runCommandCapture(const std::string& cmd, int& exitCode) {
    std::array<char, 4096> buffer{};
    std::string output;

    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) {
        exitCode = -1;
        return output;
    }

    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe) != nullptr) {
        output += buffer.data();
    }

    const int status = pclose(pipe);
    if (status == -1) {
        exitCode = -1;
    } else if (WIFEXITED(status)) {
        exitCode = WEXITSTATUS(status);
    } else {
        exitCode = -1;
    }

    return output;
}

CurlResponse runCurlWithStatus(const std::string& curlCmd) {
    int exitCode = -1;
    const std::string command = curlCmd + " -w '\\n%{http_code}'";
    const std::string output = runCommandCapture(command, exitCode);

    CurlResponse response;
    response.exitCode = exitCode;
    response.rawOutput = output;
    response.body = output;

    const size_t split = output.rfind('\n');
    if (split != std::string::npos) {
        response.body = output.substr(0, split);
        const std::string codeText = output.substr(split + 1);
        try {
            response.httpCode = std::stoi(codeText);
        } catch (...) {
            response.httpCode = 0;
        }
    }
    return response;
}

bool containsAnyToken(const std::string& text, const std::vector<std::string>& tokens) {
    for (const auto& token : tokens) {
        if (text.find(token) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::string readFileText(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return {};
    }
    return std::string(
        (std::istreambuf_iterator<char>(in)),
        std::istreambuf_iterator<char>());
}

std::string shortLogSnippet(const std::string& fullLog) {
    if (fullLog.empty()) {
        return "(empty log)";
    }
    const size_t maxLen = 1600;
    if (fullLog.size() <= maxLen) {
        return fullLog;
    }
    return fullLog.substr(fullLog.size() - maxLen);
}

std::string buildFileParseCmd(
    const std::string& url,
    const std::string& pdfPath,
    const std::vector<std::string>& fields)
{
    std::string cmd =
        "curl -sS --max-time 300 -X POST "
        "-F " + shellEscape("files=@" + pdfPath + ";type=application/pdf") + " ";
    for (const auto& field : fields) {
        cmd += "-F " + shellEscape(field) + " ";
    }
    cmd += shellEscape(url);
    return cmd;
}

const json& singleResult(const json& payload) {
    return payload["results"][0];
}

uint64_t hashText(const std::string& text) {
    return static_cast<uint64_t>(std::hash<std::string>{}(text));
}

uint64_t hashJsonCanonical(const json& value) {
    return hashText(value.dump());
}

void assertPerRequestMetricsPresent(const json& stats) {
    ASSERT_TRUE(stats.contains("pdf_render_ms"));
    ASSERT_TRUE(stats.contains("layout_ms"));
    ASSERT_TRUE(stats.contains("ocr_ms"));
    ASSERT_TRUE(stats.contains("table_ms"));
    ASSERT_TRUE(stats.contains("reading_order_ms"));
    ASSERT_TRUE(stats.contains("output_gen_ms"));
    ASSERT_TRUE(stats.contains("npu_lock_wait_ms"));
    ASSERT_TRUE(stats.contains("npu_lock_hold_ms"));
    ASSERT_TRUE(stats.contains("npu_serial_ms"));
    ASSERT_TRUE(stats.contains("cpu_only_ms"));
    ASSERT_TRUE(stats.contains("pipeline_call_ms"));
    ASSERT_TRUE(stats.contains("text_boxes_raw_count"));
    ASSERT_TRUE(stats.contains("text_boxes_after_dedup_count"));
    ASSERT_TRUE(stats.contains("table_boxes_raw_count"));
    ASSERT_TRUE(stats.contains("table_boxes_after_dedup_count"));
    ASSERT_TRUE(stats.contains("ocr_submit_count"));
    ASSERT_TRUE(stats.contains("ocr_submit_area_sum"));
    ASSERT_TRUE(stats.contains("ocr_submit_area_mean"));
    ASSERT_TRUE(stats.contains("ocr_submit_area_p50"));
    ASSERT_TRUE(stats.contains("ocr_submit_area_p95"));
    ASSERT_TRUE(stats.contains("ocr_submit_small_count"));
    ASSERT_TRUE(stats.contains("ocr_submit_medium_count"));
    ASSERT_TRUE(stats.contains("ocr_submit_large_count"));
    ASSERT_TRUE(stats.contains("ocr_submit_text_count"));
    ASSERT_TRUE(stats.contains("ocr_submit_title_count"));
    ASSERT_TRUE(stats.contains("ocr_submit_code_count"));
    ASSERT_TRUE(stats.contains("ocr_submit_list_count"));
    ASSERT_TRUE(stats.contains("ocr_dedup_skipped_count"));
    ASSERT_TRUE(stats.contains("table_npu_submit_count"));
    ASSERT_TRUE(stats.contains("table_dedup_skipped_count"));
    ASSERT_TRUE(stats.contains("ocr_timeout_count"));
    ASSERT_TRUE(stats.contains("ocr_buffered_result_hit_count"));
    ASSERT_TRUE(stats.contains("route_queue_ms"));
    ASSERT_TRUE(stats.contains("lb_proxy_ms"));
    EXPECT_GE(stats.value("pdf_render_ms", -1.0), 0.0);
    EXPECT_GE(stats.value("layout_ms", -1.0), 0.0);
    EXPECT_GE(stats.value("ocr_ms", -1.0), 0.0);
    EXPECT_GE(stats.value("table_ms", -1.0), 0.0);
    EXPECT_GE(stats.value("reading_order_ms", -1.0), 0.0);
    EXPECT_GE(stats.value("output_gen_ms", -1.0), 0.0);
    EXPECT_GE(stats.value("npu_lock_wait_ms", -1.0), 0.0);
    EXPECT_GE(stats.value("npu_lock_hold_ms", -1.0), 0.0);
    EXPECT_GE(stats.value("npu_serial_ms", -1.0), 0.0);
    EXPECT_GE(stats.value("cpu_only_ms", -1.0), 0.0);
    EXPECT_GE(stats.value("pipeline_call_ms", -1.0), 0.0);
    EXPECT_GE(stats.value("text_boxes_raw_count", -1.0), 0.0);
    EXPECT_GE(stats.value("text_boxes_after_dedup_count", -1.0), 0.0);
    EXPECT_GE(stats.value("table_boxes_raw_count", -1.0), 0.0);
    EXPECT_GE(stats.value("table_boxes_after_dedup_count", -1.0), 0.0);
    EXPECT_GE(stats.value("ocr_submit_count", -1.0), 0.0);
    EXPECT_GE(stats.value("ocr_submit_area_sum", -1.0), 0.0);
    EXPECT_GE(stats.value("ocr_submit_area_mean", -1.0), 0.0);
    EXPECT_GE(stats.value("ocr_submit_area_p50", -1.0), 0.0);
    EXPECT_GE(stats.value("ocr_submit_area_p95", -1.0), 0.0);
    EXPECT_GE(stats.value("ocr_submit_small_count", -1.0), 0.0);
    EXPECT_GE(stats.value("ocr_submit_medium_count", -1.0), 0.0);
    EXPECT_GE(stats.value("ocr_submit_large_count", -1.0), 0.0);
    EXPECT_GE(stats.value("ocr_submit_text_count", -1.0), 0.0);
    EXPECT_GE(stats.value("ocr_submit_title_count", -1.0), 0.0);
    EXPECT_GE(stats.value("ocr_submit_code_count", -1.0), 0.0);
    EXPECT_GE(stats.value("ocr_submit_list_count", -1.0), 0.0);
    EXPECT_GE(stats.value("ocr_dedup_skipped_count", -1.0), 0.0);
    EXPECT_GE(stats.value("table_npu_submit_count", -1.0), 0.0);
    EXPECT_GE(stats.value("table_dedup_skipped_count", -1.0), 0.0);
    EXPECT_GE(stats.value("ocr_timeout_count", -1.0), 0.0);
    EXPECT_GE(stats.value("ocr_buffered_result_hit_count", -1.0), 0.0);
    EXPECT_GE(stats.value("route_queue_ms", -1.0), 0.0);
    EXPECT_GE(stats.value("lb_proxy_ms", -1.0), 0.0);
}

class NpuConcurrencyHttpIntegrationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        if (!std::filesystem::exists(kNpuServerBinary)) {
            skipReason_ = "Missing NPU test server binary: " + kNpuServerBinary;
            return;
        }
        if (!std::filesystem::exists(kPdfA) || !std::filesystem::exists(kPdfB)) {
            skipReason_ = "Missing PDF fixtures for NPU integration.";
            return;
        }

        port_ = 19600 + (static_cast<int>(::getpid()) % 500);
        logPath_ = std::filesystem::temp_directory_path() /
            ("rapid_doc_npu_http_" + std::to_string(::getpid()) + ".log");
        startServer();
        waitForServerReady();
    }

    static void TearDownTestSuite() {
        stopServer();
        if (!logPath_.empty()) {
            std::error_code ec;
            std::filesystem::remove(logPath_, ec);
        }
    }

    void SetUp() override {
        if (!fatalReason_.empty()) {
            FAIL() << fatalReason_;
        }
        if (!skipReason_.empty()) {
            GTEST_SKIP() << skipReason_;
        }
    }

    static void startServer() {
        const std::string portText = std::to_string(port_);
        pid_ = fork();
        if (pid_ < 0) {
            fatalReason_ = "fork() failed";
            return;
        }

        if (pid_ == 0) {
            const int fd = ::open(logPath_.c_str(), O_CREAT | O_WRONLY | O_TRUNC, 0644);
            if (fd >= 0) {
                dup2(fd, STDOUT_FILENO);
                dup2(fd, STDERR_FILENO);
                close(fd);
            }

            execl(
                kNpuServerBinary.c_str(),
                kNpuServerBinary.c_str(),
                "--host",
                "127.0.0.1",
                "--port",
                portText.c_str(),
                static_cast<char*>(nullptr));
            _exit(127);
        }
    }

    static void waitForServerReady() {
        if (pid_ <= 0 || !fatalReason_.empty() || !skipReason_.empty()) {
            return;
        }

        const std::string statusUrl =
            "http://127.0.0.1:" + std::to_string(port_) + "/status";

        for (int i = 0; i < 240; ++i) {
            int status = 0;
            const pid_t done = waitpid(pid_, &status, WNOHANG);
            if (done == pid_) {
                const std::string logText = readFileText(logPath_);
                const std::string snippet = shortLogSnippet(logText);
                if (containsAnyToken(logText, {
                        "DXRT",
                        "dxrt",
                        "SERVICE_IO",
                        "DEVICE_IO",
                        "Failed to initialize OCR pipeline",
                        "Failed to initialize layout detector",
                        "Failed to initialize document pipeline",
                    })) {
                    skipReason_ =
                        "NPU runtime unavailable for integration test. server_log_tail:\n" + snippet;
                    pid_ = -1;
                    return;
                }

                fatalReason_ =
                    "NPU test server exited before ready (status=" + std::to_string(status) +
                    "). server_log_tail:\n" + snippet;
                pid_ = -1;
                return;
            }

            const std::string cmd =
                "curl -sS --max-time 2 " + shellEscape(statusUrl);
            const CurlResponse res = runCurlWithStatus(cmd);
            if (res.exitCode == 0 && res.httpCode == 200) {
                return;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(250));
        }

        fatalReason_ = "Timed out waiting for NPU test HTTP server readiness";
        stopServer();
    }

    static void stopServer() {
        if (pid_ <= 0) {
            return;
        }

        kill(pid_, SIGTERM);
        for (int i = 0; i < 40; ++i) {
            int status = 0;
            const pid_t done = waitpid(pid_, &status, WNOHANG);
            if (done == pid_) {
                pid_ = -1;
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        kill(pid_, SIGKILL);
        int status = 0;
        waitpid(pid_, &status, 0);
        pid_ = -1;
    }

    static std::string baseUrl() {
        return "http://127.0.0.1:" + std::to_string(port_);
    }

    static int port_;
    static pid_t pid_;
    static std::filesystem::path logPath_;
    static std::string skipReason_;
    static std::string fatalReason_;
};

int NpuConcurrencyHttpIntegrationTest::port_ = 0;
pid_t NpuConcurrencyHttpIntegrationTest::pid_ = -1;
std::filesystem::path NpuConcurrencyHttpIntegrationTest::logPath_;
std::string NpuConcurrencyHttpIntegrationTest::skipReason_;
std::string NpuConcurrencyHttpIntegrationTest::fatalReason_;

TEST_F(NpuConcurrencyHttpIntegrationTest, two_concurrent_requests_report_npu_lock_metrics) {
    const std::string url = baseUrl() + "/file_parse";
    const std::string cmdA = buildFileParseCmd(
        url,
        kPdfA,
        {
            "return_content_list=true",
            "return_middle_json=true",
            "clear_output_file=true",
            "start_page_id=0",
            "end_page_id=0",
        });
    const std::string cmdB = buildFileParseCmd(
        url,
        kPdfB,
        {
            "return_content_list=true",
            "return_middle_json=true",
            "clear_output_file=true",
        });

    auto futureA = std::async(std::launch::async, [&]() {
        return runCurlWithStatus(cmdA);
    });
    auto futureB = std::async(std::launch::async, [&]() {
        return runCurlWithStatus(cmdB);
    });

    const CurlResponse resA = futureA.get();
    const CurlResponse resB = futureB.get();

    ASSERT_EQ(resA.exitCode, 0) << resA.rawOutput;
    ASSERT_EQ(resB.exitCode, 0) << resB.rawOutput;
    ASSERT_EQ(resA.httpCode, 200) << resA.body;
    ASSERT_EQ(resB.httpCode, 200) << resB.body;

    const json payloadA = json::parse(resA.body, nullptr, false);
    const json payloadB = json::parse(resB.body, nullptr, false);
    ASSERT_FALSE(payloadA.is_discarded()) << resA.body;
    ASSERT_FALSE(payloadB.is_discarded()) << resB.body;
    ASSERT_TRUE(payloadA.contains("results"));
    ASSERT_TRUE(payloadB.contains("results"));
    ASSERT_EQ(payloadA["results"].size(), 1u);
    ASSERT_EQ(payloadB["results"].size(), 1u);

    const json& resultA = singleResult(payloadA);
    const json& resultB = singleResult(payloadB);
    ASSERT_TRUE(resultA.contains("stats"));
    ASSERT_TRUE(resultB.contains("stats"));
    assertPerRequestMetricsPresent(resultA["stats"]);
    assertPerRequestMetricsPresent(resultB["stats"]);

    ASSERT_TRUE(resultA.contains("content_list"));
    ASSERT_TRUE(resultB.contains("content_list"));
    ASSERT_TRUE(resultA["content_list"].is_array());
    ASSERT_TRUE(resultB["content_list"].is_array());
    const int pagesA = resultA["stats"].value("pages", 0);
    const int pagesB = resultB["stats"].value("pages", 0);
    EXPECT_EQ(static_cast<int>(resultA["content_list"].size()), pagesA);
    EXPECT_EQ(static_cast<int>(resultB["content_list"].size()), pagesB);

    ASSERT_TRUE(resultA.contains("middle_json"));
    ASSERT_TRUE(resultB.contains("middle_json"));
    ASSERT_TRUE(resultA["middle_json"].contains("pdf_info"));
    ASSERT_TRUE(resultB["middle_json"].contains("pdf_info"));
    EXPECT_EQ(static_cast<int>(resultA["middle_json"]["pdf_info"].size()), pagesA);
    EXPECT_EQ(static_cast<int>(resultB["middle_json"]["pdf_info"].size()), pagesB);

    ASSERT_TRUE(resultA.contains("output_dir"));
    ASSERT_TRUE(resultB.contains("output_dir"));
    EXPECT_NE(resultA["output_dir"].get<std::string>(), resultB["output_dir"].get<std::string>());
    EXPECT_EQ(resultA.value("topology", ""), "single_pipeline");
    EXPECT_EQ(resultB.value("topology", ""), "single_pipeline");
    ASSERT_TRUE(resultA.contains("shard_id"));
    ASSERT_TRUE(resultB.contains("shard_id"));

    const CurlResponse statusRes = runCurlWithStatus(
        "curl -sS --max-time 30 " + shellEscape(baseUrl() + "/status"));
    ASSERT_EQ(statusRes.exitCode, 0) << statusRes.rawOutput;
    ASSERT_EQ(statusRes.httpCode, 200) << statusRes.body;
    const json status = json::parse(statusRes.body, nullptr, false);
    ASSERT_FALSE(status.is_discarded()) << statusRes.body;
    ASSERT_TRUE(status.contains("topology"));
    ASSERT_TRUE(status["topology"].is_object());
    ASSERT_TRUE(status.contains("per_device"));
    ASSERT_TRUE(status["per_device"].is_array());
    ASSERT_TRUE(status.contains("pipeline_lock"));
    ASSERT_TRUE(status["pipeline_lock"].contains("samples"));
    EXPECT_GE(status["pipeline_lock"].value("samples", 0), 2);
    ASSERT_TRUE(status.contains("pipeline_stage_totals"));
    EXPECT_GE(status["pipeline_stage_totals"].value("pipeline_call_ms", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("pdf_render_ms", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("layout_ms", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("ocr_ms", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("table_ms", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("reading_order_ms", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("output_gen_ms", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("npu_serial_ms", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("cpu_only_ms", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("npu_lock_wait_ms", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("npu_lock_hold_ms", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("text_boxes_raw_count", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("text_boxes_after_dedup_count", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("table_boxes_raw_count", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("table_boxes_after_dedup_count", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("ocr_submit_count", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("ocr_submit_area_sum", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("ocr_submit_area_mean", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("ocr_submit_area_p50", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("ocr_submit_area_p95", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("ocr_submit_small_count", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("ocr_submit_medium_count", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("ocr_submit_large_count", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("ocr_submit_text_count", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("ocr_submit_title_count", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("ocr_submit_code_count", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("ocr_submit_list_count", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("ocr_dedup_skipped_count", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("table_npu_submit_count", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("table_dedup_skipped_count", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("ocr_timeout_count", -1.0), 0.0);
    EXPECT_GE(status["pipeline_stage_totals"].value("ocr_buffered_result_hit_count", -1.0), 0.0);
}

TEST_F(NpuConcurrencyHttpIntegrationTest, concurrent_page_range_keeps_ocr_isolated_between_requests) {
    const std::string url = baseUrl() + "/file_parse";
    const std::string page0Cmd = buildFileParseCmd(
        url,
        kPdfA,
        {
            "return_content_list=true",
            "return_middle_json=true",
            "clear_output_file=true",
            "start_page_id=0",
            "end_page_id=0",
        });
    const std::string page1Cmd = buildFileParseCmd(
        url,
        kPdfA,
        {
            "return_content_list=true",
            "return_middle_json=true",
            "clear_output_file=true",
            "start_page_id=1",
            "end_page_id=1",
        });

    const CurlResponse base0Res = runCurlWithStatus(page0Cmd);
    const CurlResponse base1Res = runCurlWithStatus(page1Cmd);
    ASSERT_EQ(base0Res.exitCode, 0) << base0Res.rawOutput;
    ASSERT_EQ(base1Res.exitCode, 0) << base1Res.rawOutput;
    ASSERT_EQ(base0Res.httpCode, 200) << base0Res.body;
    ASSERT_EQ(base1Res.httpCode, 200) << base1Res.body;

    const json base0Payload = json::parse(base0Res.body, nullptr, false);
    const json base1Payload = json::parse(base1Res.body, nullptr, false);
    ASSERT_FALSE(base0Payload.is_discarded()) << base0Res.body;
    ASSERT_FALSE(base1Payload.is_discarded()) << base1Res.body;
    const json& baseline0Result = singleResult(base0Payload);
    const json& baseline1Result = singleResult(base1Payload);
    ASSERT_TRUE(baseline0Result.contains("middle_json"));
    ASSERT_TRUE(baseline1Result.contains("middle_json"));
    ASSERT_FALSE(baseline0Result.value("md_content", "").empty());
    ASSERT_FALSE(baseline1Result.value("md_content", "").empty());
    const uint64_t baseline0MdHash = hashText(baseline0Result.value("md_content", ""));
    const uint64_t baseline1MdHash = hashText(baseline1Result.value("md_content", ""));
    const uint64_t baseline0MiddleHash = hashJsonCanonical(baseline0Result["middle_json"]);
    const uint64_t baseline1MiddleHash = hashJsonCanonical(baseline1Result["middle_json"]);
    EXPECT_NE(baseline0MdHash, baseline1MdHash);
    EXPECT_NE(baseline0MiddleHash, baseline1MiddleHash);

    auto future0 = std::async(std::launch::async, [&]() {
        return runCurlWithStatus(page0Cmd);
    });
    auto future1 = std::async(std::launch::async, [&]() {
        return runCurlWithStatus(page1Cmd);
    });
    const CurlResponse concurrent0Res = future0.get();
    const CurlResponse concurrent1Res = future1.get();
    ASSERT_EQ(concurrent0Res.exitCode, 0) << concurrent0Res.rawOutput;
    ASSERT_EQ(concurrent1Res.exitCode, 0) << concurrent1Res.rawOutput;
    ASSERT_EQ(concurrent0Res.httpCode, 200) << concurrent0Res.body;
    ASSERT_EQ(concurrent1Res.httpCode, 200) << concurrent1Res.body;

    const json concurrent0Payload = json::parse(concurrent0Res.body, nullptr, false);
    const json concurrent1Payload = json::parse(concurrent1Res.body, nullptr, false);
    ASSERT_FALSE(concurrent0Payload.is_discarded()) << concurrent0Res.body;
    ASSERT_FALSE(concurrent1Payload.is_discarded()) << concurrent1Res.body;
    const json& concurrent0 = singleResult(concurrent0Payload);
    const json& concurrent1 = singleResult(concurrent1Payload);
    ASSERT_TRUE(concurrent0.contains("middle_json"));
    ASSERT_TRUE(concurrent1.contains("middle_json"));

    EXPECT_EQ(hashText(concurrent0.value("md_content", "")), baseline0MdHash);
    EXPECT_EQ(hashText(concurrent1.value("md_content", "")), baseline1MdHash);
    EXPECT_EQ(hashJsonCanonical(concurrent0["middle_json"]), baseline0MiddleHash);
    EXPECT_EQ(hashJsonCanonical(concurrent1["middle_json"]), baseline1MiddleHash);
    ASSERT_TRUE(concurrent0.contains("stats"));
    ASSERT_TRUE(concurrent1.contains("stats"));
    EXPECT_EQ(concurrent0["stats"].value("pages", 0), 1);
    EXPECT_EQ(concurrent1["stats"].value("pages", 0), 1);
    ASSERT_TRUE(concurrent0.contains("output_dir"));
    ASSERT_TRUE(concurrent1.contains("output_dir"));
    EXPECT_NE(concurrent0["output_dir"].get<std::string>(), concurrent1["output_dir"].get<std::string>());
    assertPerRequestMetricsPresent(concurrent0["stats"]);
    assertPerRequestMetricsPresent(concurrent1["stats"]);
}

TEST_F(NpuConcurrencyHttpIntegrationTest, concurrent_error_request_does_not_poison_followup) {
    const std::string url = baseUrl() + "/file_parse";
    const std::string goodCmd = buildFileParseCmd(
        url,
        kPdfA,
        {
            "return_content_list=true",
            "return_middle_json=true",
            "clear_output_file=true",
            "start_page_id=0",
            "end_page_id=0",
        });
    const std::string badCmd =
        "curl -sS --max-time 60 -X POST "
        "-F " + shellEscape("return_content_list=true") + " "
        + shellEscape(url);

    const CurlResponse baselineRes = runCurlWithStatus(goodCmd);
    ASSERT_EQ(baselineRes.exitCode, 0) << baselineRes.rawOutput;
    ASSERT_EQ(baselineRes.httpCode, 200) << baselineRes.body;
    const json baselinePayload = json::parse(baselineRes.body, nullptr, false);
    ASSERT_FALSE(baselinePayload.is_discarded()) << baselineRes.body;
    const json& baselineResult = singleResult(baselinePayload);
    ASSERT_TRUE(baselineResult.contains("middle_json"));
    ASSERT_FALSE(baselineResult.value("md_content", "").empty());
    const uint64_t baselineMdHash = hashText(baselineResult.value("md_content", ""));
    const uint64_t baselineMiddleHash = hashJsonCanonical(baselineResult["middle_json"]);

    auto badFuture = std::async(std::launch::async, [&]() {
        return runCurlWithStatus(badCmd);
    });
    auto goodFuture = std::async(std::launch::async, [&]() {
        return runCurlWithStatus(goodCmd);
    });

    const CurlResponse badRes = badFuture.get();
    const CurlResponse goodRes = goodFuture.get();
    ASSERT_EQ(badRes.exitCode, 0) << badRes.rawOutput;
    ASSERT_EQ(goodRes.exitCode, 0) << goodRes.rawOutput;
    ASSERT_EQ(badRes.httpCode, 400) << badRes.body;
    ASSERT_EQ(goodRes.httpCode, 200) << goodRes.body;

    const json goodPayload = json::parse(goodRes.body, nullptr, false);
    ASSERT_FALSE(goodPayload.is_discarded()) << goodRes.body;
    const json& goodResult = singleResult(goodPayload);
    ASSERT_TRUE(goodResult.contains("stats"));
    ASSERT_TRUE(goodResult.contains("middle_json"));
    assertPerRequestMetricsPresent(goodResult["stats"]);
    EXPECT_EQ(goodResult["stats"].value("pages", 0), 1);
    EXPECT_EQ(hashText(goodResult.value("md_content", "")), baselineMdHash);
    EXPECT_EQ(hashJsonCanonical(goodResult["middle_json"]), baselineMiddleHash);

    const CurlResponse followupRes = runCurlWithStatus(goodCmd);
    ASSERT_EQ(followupRes.exitCode, 0) << followupRes.rawOutput;
    ASSERT_EQ(followupRes.httpCode, 200) << followupRes.body;
    const json followupPayload = json::parse(followupRes.body, nullptr, false);
    ASSERT_FALSE(followupPayload.is_discarded()) << followupRes.body;
    const json& followupResult = singleResult(followupPayload);
    ASSERT_TRUE(followupResult.contains("stats"));
    ASSERT_TRUE(followupResult.contains("middle_json"));
    assertPerRequestMetricsPresent(followupResult["stats"]);
    EXPECT_EQ(followupResult["stats"].value("pages", 0), 1);
    EXPECT_EQ(hashText(followupResult.value("md_content", "")), baselineMdHash);
    EXPECT_EQ(hashJsonCanonical(followupResult["middle_json"]), baselineMiddleHash);
    ASSERT_TRUE(goodResult.contains("output_dir"));
    ASSERT_TRUE(followupResult.contains("output_dir"));
    EXPECT_NE(goodResult["output_dir"].get<std::string>(), followupResult["output_dir"].get<std::string>());
}

} // namespace
