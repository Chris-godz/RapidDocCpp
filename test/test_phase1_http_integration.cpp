#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <array>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <future>
#include <string>
#include <thread>

#include <sys/wait.h>
#include <unistd.h>

using json = nlohmann::json;

namespace {

const std::string kServerBinary =
    std::string(RAPIDDOC_BUILD_DIR) + "/bin/rapid_doc_test_server";
const std::string kMultiPagePdf =
    std::string(PROJECT_ROOT_DIR) + "/test_files/BVRC_Meeting_Minutes_2024-04_origin.pdf";
const std::array<const char*, 6> kHealthEnvVars = {
    "CUSTOM_INTER_OP_THREADS_COUNT",
    "CUSTOM_INTRA_OP_THREADS_COUNT",
    "DXRT_DYNAMIC_CPU_THREAD",
    "DXRT_TASK_MAX_LOAD",
    "NFH_INPUT_WORKER_THREADS",
    "NFH_OUTPUT_WORKER_THREADS",
};

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

class FileParseHttpIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(kServerBinary)) {
            GTEST_SKIP() << "Missing test server binary: " << kServerBinary;
        }
        if (!std::filesystem::exists(kMultiPagePdf)) {
            GTEST_SKIP() << "Missing PDF fixture: " << kMultiPagePdf;
        }

        port_ = 19080 + (static_cast<int>(::getpid()) % 500);
        startServer();
        waitForServerReady();
    }

    void TearDown() override {
        stopServer();
    }

    void startServer() {
        const std::string portText = std::to_string(port_);
        pid_ = fork();
        ASSERT_GE(pid_, 0) << "fork() failed";

        if (pid_ == 0) {
            for (const char* envName : kHealthEnvVars) {
                unsetenv(envName);
            }
            execl(
                kServerBinary.c_str(),
                kServerBinary.c_str(),
                "--host",
                "127.0.0.1",
                "--port",
                portText.c_str(),
                static_cast<char*>(nullptr));
            _exit(127);
        }
    }

    void waitForServerReady() {
        const std::string statusUrl =
            "http://127.0.0.1:" + std::to_string(port_) + "/status";

        for (int i = 0; i < 60; ++i) {
            int status = 0;
            const pid_t done = waitpid(pid_, &status, WNOHANG);
            if (done == pid_) {
                pid_ = -1;
                FAIL() << "Test HTTP server exited before ready (status=" << status << ")";
            }

            const std::string cmd =
                "curl -sS --max-time 1 " + shellEscape(statusUrl);
            const CurlResponse res = runCurlWithStatus(cmd);
            if (res.exitCode == 0 && res.httpCode == 200) {
                return;
            }

            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        FAIL() << "Timed out waiting for test HTTP server readiness";
    }

    void stopServer() {
        if (pid_ <= 0) {
            return;
        }

        kill(pid_, SIGTERM);
        for (int i = 0; i < 20; ++i) {
            int status = 0;
            const pid_t done = waitpid(pid_, &status, WNOHANG);
            if (done == pid_) {
                pid_ = -1;
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }

        kill(pid_, SIGKILL);
        int status = 0;
        waitpid(pid_, &status, 0);
        pid_ = -1;
    }

    int port_ = 0;
    pid_t pid_ = -1;
};

TEST_F(FileParseHttpIntegrationTest, health_returns_valid_json_when_env_missing) {
    const std::string url =
        "http://127.0.0.1:" + std::to_string(port_) + "/health";
    const std::string cmd =
        "curl -sS --max-time 30 " + shellEscape(url);

    const CurlResponse res = runCurlWithStatus(cmd);
    ASSERT_EQ(res.exitCode, 0) << res.rawOutput;
    ASSERT_EQ(res.httpCode, 200) << res.body;

    const json payload = json::parse(res.body, nullptr, false);
    ASSERT_FALSE(payload.is_discarded()) << res.body;
    EXPECT_EQ(payload.value("status", ""), "healthy");

    ASSERT_TRUE(payload.contains("environment_variables"));
    ASSERT_TRUE(payload["environment_variables"].is_object());
    const auto& envVars = payload["environment_variables"];
    for (const char* envName : kHealthEnvVars) {
        ASSERT_TRUE(envVars.contains(envName)) << envName;
        EXPECT_TRUE(envVars[envName].is_null())
            << envName << " should be null when missing";
    }
}

TEST_F(FileParseHttpIntegrationTest, file_parse_multipart_success_contract) {
    const std::string url =
        "http://127.0.0.1:" + std::to_string(port_) + "/file_parse";
    const std::string cmd =
        "curl -sS --max-time 90 -X POST "
        "-F " + shellEscape("files=@" + kMultiPagePdf + ";type=application/pdf") + " "
        "-F " + shellEscape("return_content_list=true") + " "
        "-F " + shellEscape("clear_output_file=true") + " "
        + shellEscape(url);

    const CurlResponse res = runCurlWithStatus(cmd);
    ASSERT_EQ(res.exitCode, 0) << res.rawOutput;
    ASSERT_EQ(res.httpCode, 200) << res.body;

    const json payload = json::parse(res.body, nullptr, false);
    ASSERT_FALSE(payload.is_discarded()) << res.body;
    ASSERT_TRUE(payload.contains("results"));
    ASSERT_TRUE(payload["results"].is_array());
    ASSERT_EQ(payload["results"].size(), 1u);

    const auto& result = payload["results"][0];
    ASSERT_TRUE(result.contains("stats"));
    ASSERT_TRUE(result["stats"].is_object());
    ASSERT_TRUE(result.contains("content_list"));
    ASSERT_TRUE(result["content_list"].is_array());
    ASSERT_TRUE(result.contains("topology"));
    ASSERT_TRUE(result.contains("device_id"));
    ASSERT_TRUE(result.contains("shard_id"));
    ASSERT_TRUE(result.contains("backend_id"));
    EXPECT_EQ(result.value("topology", ""), "single_pipeline");
    EXPECT_TRUE(result["stats"].contains("route_queue_ms"));
    EXPECT_TRUE(result["stats"].contains("lb_proxy_ms"));
    EXPECT_GE(result["stats"].value("route_queue_ms", -1.0), 0.0);
    EXPECT_GE(result["stats"].value("lb_proxy_ms", -1.0), 0.0);

    const int pages = result["stats"].value("pages", 0);
    const int totalPages = result["stats"].value("total_pages", 0);
    ASSERT_GE(pages, 2);
    EXPECT_EQ(totalPages, pages);
    EXPECT_EQ(static_cast<int>(result["content_list"].size()), pages);

    for (const auto& pageItems : result["content_list"]) {
        EXPECT_TRUE(pageItems.is_array());
    }
}

TEST_F(FileParseHttpIntegrationTest, file_parse_returns_clear_error_when_file_missing) {
    const std::string url =
        "http://127.0.0.1:" + std::to_string(port_) + "/file_parse";
    const std::string cmd =
        "curl -sS --max-time 30 -X POST "
        "-F " + shellEscape("return_content_list=true") + " "
        + shellEscape(url);

    const CurlResponse res = runCurlWithStatus(cmd);
    ASSERT_EQ(res.exitCode, 0) << res.rawOutput;
    ASSERT_EQ(res.httpCode, 400) << res.body;

    const json payload = json::parse(res.body, nullptr, false);
    ASSERT_FALSE(payload.is_discarded()) << res.body;
    ASSERT_TRUE(payload.contains("error"));
    EXPECT_EQ(payload["error"], "No files provided");
}

TEST_F(FileParseHttpIntegrationTest, file_parse_page_range_does_not_leak_between_requests) {
    const std::string url =
        "http://127.0.0.1:" + std::to_string(port_) + "/file_parse";

    const std::string firstCmd =
        "curl -sS --max-time 90 -X POST "
        "-F " + shellEscape("files=@" + kMultiPagePdf + ";type=application/pdf") + " "
        "-F " + shellEscape("return_content_list=true") + " "
        "-F " + shellEscape("clear_output_file=true") + " "
        "-F " + shellEscape("start_page_id=0") + " "
        "-F " + shellEscape("end_page_id=0") + " "
        + shellEscape(url);
    const CurlResponse firstRes = runCurlWithStatus(firstCmd);
    ASSERT_EQ(firstRes.exitCode, 0) << firstRes.rawOutput;
    ASSERT_EQ(firstRes.httpCode, 200) << firstRes.body;

    const json firstPayload = json::parse(firstRes.body, nullptr, false);
    ASSERT_FALSE(firstPayload.is_discarded()) << firstRes.body;
    ASSERT_TRUE(firstPayload.contains("results"));
    ASSERT_EQ(firstPayload["results"].size(), 1u);

    const auto& firstResult = firstPayload["results"][0];
    ASSERT_TRUE(firstResult.contains("stats"));
    const int firstPages = firstResult["stats"].value("pages", 0);
    const int firstTotalPages = firstResult["stats"].value("total_pages", 0);
    EXPECT_EQ(firstPages, 1);
    EXPECT_EQ(firstTotalPages, 1);
    ASSERT_TRUE(firstResult.contains("content_list"));
    ASSERT_TRUE(firstResult["content_list"].is_array());
    EXPECT_EQ(static_cast<int>(firstResult["content_list"].size()), 1);

    const std::string secondCmd =
        "curl -sS --max-time 90 -X POST "
        "-F " + shellEscape("files=@" + kMultiPagePdf + ";type=application/pdf") + " "
        "-F " + shellEscape("return_content_list=true") + " "
        "-F " + shellEscape("clear_output_file=true") + " "
        + shellEscape(url);
    const CurlResponse secondRes = runCurlWithStatus(secondCmd);
    ASSERT_EQ(secondRes.exitCode, 0) << secondRes.rawOutput;
    ASSERT_EQ(secondRes.httpCode, 200) << secondRes.body;

    const json secondPayload = json::parse(secondRes.body, nullptr, false);
    ASSERT_FALSE(secondPayload.is_discarded()) << secondRes.body;
    ASSERT_TRUE(secondPayload.contains("results"));
    ASSERT_EQ(secondPayload["results"].size(), 1u);

    const auto& secondResult = secondPayload["results"][0];
    ASSERT_TRUE(secondResult.contains("stats"));
    const int secondPages = secondResult["stats"].value("pages", 0);
    const int secondTotalPages = secondResult["stats"].value("total_pages", 0);
    EXPECT_GE(secondPages, 2);
    EXPECT_EQ(secondTotalPages, secondPages);
    EXPECT_GT(secondPages, firstPages);
    ASSERT_TRUE(secondResult.contains("content_list"));
    ASSERT_TRUE(secondResult["content_list"].is_array());
    EXPECT_EQ(static_cast<int>(secondResult["content_list"].size()), secondPages);
}

TEST_F(FileParseHttpIntegrationTest, file_parse_concurrent_page_range_contract) {
    const std::string url =
        "http://127.0.0.1:" + std::to_string(port_) + "/file_parse";

    const std::string rangedCmd =
        "curl -sS --max-time 90 -X POST "
        "-F " + shellEscape("files=@" + kMultiPagePdf + ";type=application/pdf") + " "
        "-F " + shellEscape("return_content_list=true") + " "
        "-F " + shellEscape("clear_output_file=true") + " "
        "-F " + shellEscape("start_page_id=0") + " "
        "-F " + shellEscape("end_page_id=0") + " "
        + shellEscape(url);
    const std::string fullCmd =
        "curl -sS --max-time 90 -X POST "
        "-F " + shellEscape("files=@" + kMultiPagePdf + ";type=application/pdf") + " "
        "-F " + shellEscape("return_content_list=true") + " "
        "-F " + shellEscape("clear_output_file=true") + " "
        + shellEscape(url);

    auto rangedFuture = std::async(std::launch::async, [&]() {
        return runCurlWithStatus(rangedCmd);
    });
    auto fullFuture = std::async(std::launch::async, [&]() {
        return runCurlWithStatus(fullCmd);
    });

    const CurlResponse rangedRes = rangedFuture.get();
    const CurlResponse fullRes = fullFuture.get();

    ASSERT_EQ(rangedRes.exitCode, 0) << rangedRes.rawOutput;
    ASSERT_EQ(rangedRes.httpCode, 200) << rangedRes.body;
    ASSERT_EQ(fullRes.exitCode, 0) << fullRes.rawOutput;
    ASSERT_EQ(fullRes.httpCode, 200) << fullRes.body;

    const json rangedPayload = json::parse(rangedRes.body, nullptr, false);
    const json fullPayload = json::parse(fullRes.body, nullptr, false);
    ASSERT_FALSE(rangedPayload.is_discarded()) << rangedRes.body;
    ASSERT_FALSE(fullPayload.is_discarded()) << fullRes.body;

    ASSERT_TRUE(rangedPayload.contains("results"));
    ASSERT_TRUE(fullPayload.contains("results"));
    ASSERT_EQ(rangedPayload["results"].size(), 1u);
    ASSERT_EQ(fullPayload["results"].size(), 1u);

    const auto& rangedResult = rangedPayload["results"][0];
    const auto& fullResult = fullPayload["results"][0];
    ASSERT_TRUE(rangedResult.contains("stats"));
    ASSERT_TRUE(fullResult.contains("stats"));
    ASSERT_TRUE(rangedResult.contains("content_list"));
    ASSERT_TRUE(fullResult.contains("content_list"));
    ASSERT_TRUE(rangedResult["content_list"].is_array());
    ASSERT_TRUE(fullResult["content_list"].is_array());

    const int rangedPages = rangedResult["stats"].value("pages", 0);
    const int fullPages = fullResult["stats"].value("pages", 0);
    EXPECT_EQ(rangedPages, 1);
    EXPECT_GE(fullPages, 2);
    EXPECT_EQ(static_cast<int>(rangedResult["content_list"].size()), rangedPages);
    EXPECT_EQ(static_cast<int>(fullResult["content_list"].size()), fullPages);

    EXPECT_TRUE(rangedResult["stats"].contains("npu_lock_wait_ms"));
    EXPECT_TRUE(rangedResult["stats"].contains("npu_lock_hold_ms"));
    EXPECT_TRUE(fullResult["stats"].contains("npu_lock_wait_ms"));
    EXPECT_TRUE(fullResult["stats"].contains("npu_lock_hold_ms"));
    EXPECT_GE(rangedResult["stats"].value("npu_lock_wait_ms", -1.0), 0.0);
    EXPECT_GE(rangedResult["stats"].value("npu_lock_hold_ms", -1.0), 0.0);
    EXPECT_GE(fullResult["stats"].value("npu_lock_wait_ms", -1.0), 0.0);
    EXPECT_GE(fullResult["stats"].value("npu_lock_hold_ms", -1.0), 0.0);
}

TEST_F(FileParseHttpIntegrationTest, file_parse_concurrent_output_dir_unique_and_cleanup_isolated) {
    const std::string url =
        "http://127.0.0.1:" + std::to_string(port_) + "/file_parse";

    const std::string cmd =
        "curl -sS --max-time 90 -X POST "
        "-F " + shellEscape("files=@" + kMultiPagePdf + ";type=application/pdf") + " "
        "-F " + shellEscape("return_content_list=true") + " "
        "-F " + shellEscape("clear_output_file=true") + " "
        + shellEscape(url);

    auto firstFuture = std::async(std::launch::async, [&]() {
        return runCurlWithStatus(cmd);
    });
    auto secondFuture = std::async(std::launch::async, [&]() {
        return runCurlWithStatus(cmd);
    });

    const CurlResponse firstRes = firstFuture.get();
    const CurlResponse secondRes = secondFuture.get();
    ASSERT_EQ(firstRes.exitCode, 0) << firstRes.rawOutput;
    ASSERT_EQ(secondRes.exitCode, 0) << secondRes.rawOutput;
    ASSERT_EQ(firstRes.httpCode, 200) << firstRes.body;
    ASSERT_EQ(secondRes.httpCode, 200) << secondRes.body;

    const json firstPayload = json::parse(firstRes.body, nullptr, false);
    const json secondPayload = json::parse(secondRes.body, nullptr, false);
    ASSERT_FALSE(firstPayload.is_discarded()) << firstRes.body;
    ASSERT_FALSE(secondPayload.is_discarded()) << secondRes.body;
    ASSERT_TRUE(firstPayload.contains("results"));
    ASSERT_TRUE(secondPayload.contains("results"));
    ASSERT_EQ(firstPayload["results"].size(), 1u);
    ASSERT_EQ(secondPayload["results"].size(), 1u);

    const std::string firstOutputDir =
        firstPayload["results"][0].value("output_dir", "");
    const std::string secondOutputDir =
        secondPayload["results"][0].value("output_dir", "");
    ASSERT_FALSE(firstOutputDir.empty());
    ASSERT_FALSE(secondOutputDir.empty());
    EXPECT_NE(firstOutputDir, secondOutputDir);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    EXPECT_FALSE(std::filesystem::exists(firstOutputDir));
    EXPECT_FALSE(std::filesystem::exists(secondOutputDir));
}

TEST_F(FileParseHttpIntegrationTest, status_reports_pipeline_lock_observability) {
    const std::string parseUrl =
        "http://127.0.0.1:" + std::to_string(port_) + "/file_parse";
    const std::string parseCmd =
        "curl -sS --max-time 90 -X POST "
        "-F " + shellEscape("files=@" + kMultiPagePdf + ";type=application/pdf") + " "
        "-F " + shellEscape("clear_output_file=true") + " "
        + shellEscape(parseUrl);
    const CurlResponse parseRes = runCurlWithStatus(parseCmd);
    ASSERT_EQ(parseRes.exitCode, 0) << parseRes.rawOutput;
    ASSERT_EQ(parseRes.httpCode, 200) << parseRes.body;

    const std::string statusUrl =
        "http://127.0.0.1:" + std::to_string(port_) + "/status";
    const CurlResponse statusRes = runCurlWithStatus(
        "curl -sS --max-time 30 " + shellEscape(statusUrl));
    ASSERT_EQ(statusRes.exitCode, 0) << statusRes.rawOutput;
    ASSERT_EQ(statusRes.httpCode, 200) << statusRes.body;

    const json status = json::parse(statusRes.body, nullptr, false);
    ASSERT_FALSE(status.is_discarded()) << statusRes.body;
    ASSERT_TRUE(status.contains("topology"));
    ASSERT_TRUE(status["topology"].is_object());
    EXPECT_EQ(status["topology"].value("mode", ""), "single_pipeline");
    ASSERT_TRUE(status.contains("per_device"));
    ASSERT_TRUE(status["per_device"].is_array());
    ASSERT_TRUE(status.contains("pipeline_lock"));
    ASSERT_TRUE(status["pipeline_lock"].is_object());
    ASSERT_TRUE(status["pipeline_lock"].contains("samples"));
    EXPECT_GE(status["pipeline_lock"].value("samples", 0), 1);
    ASSERT_TRUE(status["pipeline_lock"].contains("wait_total_ms"));
    ASSERT_TRUE(status["pipeline_lock"].contains("hold_total_ms"));
    EXPECT_GE(status["pipeline_lock"].value("wait_total_ms", -1.0), 0.0);
    EXPECT_GE(status["pipeline_lock"].value("hold_total_ms", -1.0), 0.0);
    ASSERT_TRUE(status.contains("pipeline_stage_totals"));
    ASSERT_TRUE(status["pipeline_stage_totals"].is_object());
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
}

TEST_F(FileParseHttpIntegrationTest, file_parse_concurrent_error_does_not_poison_followup) {
    const std::string url =
        "http://127.0.0.1:" + std::to_string(port_) + "/file_parse";
    const std::string goodCmd =
        "curl -sS --max-time 90 -X POST "
        "-F " + shellEscape("files=@" + kMultiPagePdf + ";type=application/pdf") + " "
        "-F " + shellEscape("return_content_list=true") + " "
        "-F " + shellEscape("clear_output_file=true") + " "
        + shellEscape(url);
    const std::string badCmd =
        "curl -sS --max-time 30 -X POST "
        "-F " + shellEscape("return_content_list=true") + " "
        + shellEscape(url);

    auto badFuture = std::async(std::launch::async, [&]() {
        return runCurlWithStatus(badCmd);
    });
    auto goodFuture = std::async(std::launch::async, [&]() {
        return runCurlWithStatus(goodCmd);
    });

    const CurlResponse badRes = badFuture.get();
    const CurlResponse goodRes = goodFuture.get();

    ASSERT_EQ(badRes.exitCode, 0) << badRes.rawOutput;
    ASSERT_EQ(badRes.httpCode, 400) << badRes.body;
    ASSERT_EQ(goodRes.exitCode, 0) << goodRes.rawOutput;
    ASSERT_EQ(goodRes.httpCode, 200) << goodRes.body;

    const json goodPayload = json::parse(goodRes.body, nullptr, false);
    ASSERT_FALSE(goodPayload.is_discarded()) << goodRes.body;
    ASSERT_TRUE(goodPayload.contains("results"));
    ASSERT_EQ(goodPayload["results"].size(), 1u);
    EXPECT_GE(goodPayload["results"][0]["stats"].value("pages", 0), 2);

    const CurlResponse followupRes = runCurlWithStatus(goodCmd);
    ASSERT_EQ(followupRes.exitCode, 0) << followupRes.rawOutput;
    ASSERT_EQ(followupRes.httpCode, 200) << followupRes.body;

    const json followupPayload = json::parse(followupRes.body, nullptr, false);
    ASSERT_FALSE(followupPayload.is_discarded()) << followupRes.body;
    ASSERT_TRUE(followupPayload.contains("results"));
    ASSERT_EQ(followupPayload["results"].size(), 1u);
    EXPECT_GE(followupPayload["results"][0]["stats"].value("pages", 0), 2);
}

} // namespace
