#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <thread>

#include <sys/wait.h>
#include <unistd.h>

using json = nlohmann::json;

namespace {

const std::string kBackendBinary =
    std::string(RAPIDDOC_BUILD_DIR) + "/bin/rapid_doc_test_server";
const std::string kLbBinary =
    std::string(RAPIDDOC_BUILD_DIR) + "/bin/rapid_doc_topology_lb";
const std::string kPdf =
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
    const std::string output = runCommandCapture(curlCmd + " -w '\\n%{http_code}'", exitCode);
    CurlResponse response;
    response.exitCode = exitCode;
    response.rawOutput = output;
    response.body = output;
    const size_t split = output.rfind('\n');
    if (split != std::string::npos) {
        response.body = output.substr(0, split);
        try {
            response.httpCode = std::stoi(output.substr(split + 1));
        } catch (...) {
            response.httpCode = 0;
        }
    }
    return response;
}

class TopologyLbIntegrationTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!std::filesystem::exists(kBackendBinary) ||
            !std::filesystem::exists(kLbBinary) ||
            !std::filesystem::exists(kPdf)) {
            GTEST_SKIP() << "Missing backend, LB binary, or PDF fixture";
        }
        backendPort_ = 19180 + (static_cast<int>(::getpid()) % 200);
        lbPort_ = backendPort_ + 1;
        startBackend();
        waitReady(backendPort_);
        startLb();
        waitReady(lbPort_);
    }

    void TearDown() override {
        stopProcess(lbPid_);
        stopProcess(backendPid_);
    }

    void startBackend() {
        backendPid_ = fork();
        ASSERT_GE(backendPid_, 0);
        if (backendPid_ == 0) {
            execl(
                kBackendBinary.c_str(),
                kBackendBinary.c_str(),
                "--host", "127.0.0.1",
                "--port", std::to_string(backendPort_).c_str(),
                "--workers", "1",
                "--server-id", "backend_0",
                static_cast<char*>(nullptr));
            _exit(127);
        }
    }

    void startLb() {
        lbPid_ = fork();
        ASSERT_GE(lbPid_, 0);
        if (lbPid_ == 0) {
            const std::string backendUrl = "http://127.0.0.1:" + std::to_string(backendPort_);
            execl(
                kLbBinary.c_str(),
                kLbBinary.c_str(),
                "--host", "127.0.0.1",
                "--port", std::to_string(lbPort_).c_str(),
                "--workers", "1",
                "--server-id", "front_lb_test",
                "--backend-url", backendUrl.c_str(),
                static_cast<char*>(nullptr));
            _exit(127);
        }
    }

    void waitReady(int port) {
        const std::string statusUrl = "http://127.0.0.1:" + std::to_string(port) + "/status";
        for (int i = 0; i < 80; ++i) {
            const CurlResponse res = runCurlWithStatus(
                "curl -sS --max-time 1 " + shellEscape(statusUrl));
            if (res.exitCode == 0 && res.httpCode == 200) {
                return;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        FAIL() << "Timed out waiting for server readiness on port " << port;
    }

    void stopProcess(pid_t& pid) {
        if (pid <= 0) {
            return;
        }
        kill(pid, SIGTERM);
        int status = 0;
        waitpid(pid, &status, 0);
        pid = -1;
    }

    int backendPort_ = 0;
    int lbPort_ = 0;
    pid_t backendPid_ = -1;
    pid_t lbPid_ = -1;
};

TEST_F(TopologyLbIntegrationTest, file_parse_and_status_report_front_lb_topology) {
    const std::string fileParseUrl =
        "http://127.0.0.1:" + std::to_string(lbPort_) + "/file_parse";
    const std::string parseCmd =
        "curl -sS --max-time 90 -X POST "
        "-F " + shellEscape("files=@" + kPdf + ";type=application/pdf") + " "
        "-F " + shellEscape("return_content_list=true") + " "
        "-F " + shellEscape("clear_output_file=true") + " "
        + shellEscape(fileParseUrl);

    const CurlResponse parseRes = runCurlWithStatus(parseCmd);
    ASSERT_EQ(parseRes.exitCode, 0) << parseRes.rawOutput;
    ASSERT_EQ(parseRes.httpCode, 200) << parseRes.body;

    const json parsePayload = json::parse(parseRes.body, nullptr, false);
    ASSERT_FALSE(parsePayload.is_discarded()) << parseRes.body;
    ASSERT_TRUE(parsePayload.contains("results"));
    ASSERT_EQ(parsePayload["results"].size(), 1u);
    const auto& result = parsePayload["results"][0];
    EXPECT_EQ(result.value("topology", ""), "front_lb");
    EXPECT_EQ(result.value("backend_id", ""), "backend_0");
    ASSERT_TRUE(result.contains("stats"));
    EXPECT_TRUE(result["stats"].contains("lb_proxy_ms"));
    EXPECT_GE(result["stats"].value("lb_proxy_ms", -1.0), 0.0);

    const std::string statusUrl =
        "http://127.0.0.1:" + std::to_string(lbPort_) + "/status";
    const CurlResponse statusRes = runCurlWithStatus(
        "curl -sS --max-time 30 " + shellEscape(statusUrl));
    ASSERT_EQ(statusRes.exitCode, 0) << statusRes.rawOutput;
    ASSERT_EQ(statusRes.httpCode, 200) << statusRes.body;

    const json status = json::parse(statusRes.body, nullptr, false);
    ASSERT_FALSE(status.is_discarded()) << statusRes.body;
    ASSERT_TRUE(status.contains("topology"));
    EXPECT_EQ(status["topology"].value("mode", ""), "front_lb");
    ASSERT_TRUE(status.contains("backends"));
    ASSERT_TRUE(status["backends"].is_array());
    ASSERT_EQ(status["backends"].size(), 1u);
}

} // namespace
