#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <array>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
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

} // namespace
