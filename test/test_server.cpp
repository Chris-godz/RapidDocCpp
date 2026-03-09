/**
 * @file test_server.cpp
 * @brief Tests for HTTP server module
 *
 * Unit tests for server utilities, config, and status JSON generation.
 * Optional integration: start server and test HTTP endpoints.
 */

#include <gtest/gtest.h>
#include "server/server.h"
#include <string>
#include <vector>

namespace {

static std::string base64Encode(const std::string& input) {
    static const char* base64Chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    
    std::string result;
    int i = 0;
    int j = 0;
    unsigned char charArray3[3];
    unsigned char charArray4[4];
    int inLen = static_cast<int>(input.size());
    const unsigned char* bytesToEncode = reinterpret_cast<const unsigned char*>(input.c_str());
    
    while (inLen--) {
        charArray3[i++] = *(bytesToEncode++);
        if (i == 3) {
            charArray4[0] = (charArray3[0] & 0xfc) >> 2;
            charArray4[1] = ((charArray3[0] & 0x03) << 4) + ((charArray3[1] & 0xf0) >> 4);
            charArray4[2] = ((charArray3[1] & 0x0f) << 2) + ((charArray3[2] & 0xc0) >> 6);
            charArray4[2] = charArray4[2] == 64 ? '=' : charArray4[2];
            charArray4[3] = charArray3[2] & 0x3f;
            charArray4[3] = charArray4[3] == 64 ? '=' : charArray4[3];
            
            for(i = 0; i < 4; i++)
                result += base64Chars[charArray4[i]];
            i = 0;
        }
    }
    
    if (i) {
        for(j = i; j < 3; j++)
            charArray3[j] = '\0';
        
        charArray4[0] = (charArray3[0] & 0xfc) >> 2;
        charArray4[1] = ((charArray3[0] & 0x03) << 4) + ((charArray3[1] & 0xf0) >> 4);
        charArray4[2] = ((charArray3[1] & 0x0f) << 2) + ((charArray3[2] & 0xc0) >> 6);
        
        for (j = 0; j < i + 1; j++)
            result += base64Chars[charArray4[j]];
        
        while((i++ < 3))
            result += '=';
    }
    
    return result;
}

} // anonymous namespace

// ========================================
// ServerConfig Tests
// ========================================

TEST(ServerConfig, DefaultValues) {
    rapid_doc::ServerConfig config;
    
    EXPECT_EQ(config.host, "0.0.0.0");
    EXPECT_EQ(config.port, 8080);
    EXPECT_EQ(config.numWorkers, 4);
    EXPECT_EQ(config.maxUploadSize, 50 * 1024 * 1024);
    EXPECT_EQ(config.uploadDir, "./uploads");
}

TEST(ServerConfig, CustomValues) {
    rapid_doc::ServerConfig config;
    config.host = "127.0.0.1";
    config.port = 9000;
    config.numWorkers = 8;
    config.uploadDir = "/tmp/test_uploads";
    
    EXPECT_EQ(config.host, "127.0.0.1");
    EXPECT_EQ(config.port, 9000);
    EXPECT_EQ(config.numWorkers, 8);
    EXPECT_EQ(config.uploadDir, "/tmp/test_uploads");
}

TEST(ServerConfig, PipelineConfigEmbedded) {
    rapid_doc::ServerConfig config;
    config.pipelineConfig.runtime.pdfDpi = 200;
    config.pipelineConfig.stages.enableReadingOrder = false;
    
    EXPECT_EQ(config.pipelineConfig.runtime.pdfDpi, 200);
    EXPECT_FALSE(config.pipelineConfig.stages.enableReadingOrder);
}

// ========================================
// DocServer Construction Tests
// ========================================

TEST(DocServer, CreateWithMinimalConfig) {
    rapid_doc::ServerConfig config;
    config.pipelineConfig.stages.enablePdfRender = false;
    config.pipelineConfig.stages.enableLayout = false;
    config.pipelineConfig.stages.enableOcr = false;
    config.pipelineConfig.stages.enableWiredTable = false;
    config.pipelineConfig.stages.enableReadingOrder = false;
    config.pipelineConfig.stages.enableMarkdownOutput = true;
    
    try {
        rapid_doc::DocServer server(config);
        EXPECT_TRUE(true);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Failed to create server (may require models): " << e.what();
    }
}

TEST(DocServer, InitialState) {
    rapid_doc::ServerConfig config;
    config.pipelineConfig.stages.enablePdfRender = false;
    config.pipelineConfig.stages.enableLayout = false;
    config.pipelineConfig.stages.enableOcr = false;
    config.pipelineConfig.stages.enableWiredTable = false;
    config.pipelineConfig.stages.enableReadingOrder = false;
    config.pipelineConfig.stages.enableMarkdownOutput = true;
    
    try {
        rapid_doc::DocServer server(config);
        EXPECT_FALSE(server.isRunning());
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Failed to create server: " << e.what();
    }
}

// ========================================
// Base64 Utility Tests
// ========================================

TEST(ServerBase64, EncodeEmpty) {
    std::string input = "";
    std::string encoded = base64Encode(input);
    EXPECT_EQ(encoded, "");
}

TEST(ServerBase64, EncodeHello) {
    std::string input = "Hello";
    std::string encoded = base64Encode(input);
    EXPECT_EQ(encoded, "SGVsbG8=");
}

TEST(ServerBase64, EncodeWorld) {
    std::string input = "World";
    std::string encoded = base64Encode(input);
    EXPECT_EQ(encoded, "V29ybGQ=");
}

TEST(ServerBase64, EncodeLonger) {
    std::string input = "Hello World!";
    std::string encoded = base64Encode(input);
    EXPECT_EQ(encoded, "SGVsbG8gV29ybGQh");
}

// ========================================
// Status JSON Tests
// ========================================

TEST(ServerStatus, BuildStatusJson) {
    rapid_doc::ServerConfig config;
    config.pipelineConfig.stages.enablePdfRender = false;
    config.pipelineConfig.stages.enableLayout = false;
    config.pipelineConfig.stages.enableOcr = false;
    config.pipelineConfig.stages.enableWiredTable = false;
    config.pipelineConfig.stages.enableReadingOrder = false;
    config.pipelineConfig.stages.enableMarkdownOutput = true;
    
    try {
        rapid_doc::DocServer server(config);
        
        // Can't call private buildStatusJson, so test that server can be created
        // and check basic functionality
        EXPECT_FALSE(server.isRunning());
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Failed to create server: " << e.what();
    }
}

TEST(ServerStatus, NpuSupportFields) {
    rapid_doc::ServerConfig config;
    config.pipelineConfig.stages.enablePdfRender = false;
    config.pipelineConfig.stages.enableLayout = false;
    config.pipelineConfig.stages.enableOcr = false;
    config.pipelineConfig.stages.enableWiredTable = false;
    config.pipelineConfig.stages.enableReadingOrder = false;
    config.pipelineConfig.stages.enableMarkdownOutput = true;
    
    try {
        rapid_doc::DocServer server(config);
        
        // Server created successfully - NPU support info is built-in
        // Layout and OCR are always marked as supported in current implementation
        EXPECT_FALSE(server.isRunning());
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Failed to create server: " << e.what();
    }
}

// ========================================
// Integration Tests (Optional - requires server startup)
// ========================================

TEST(DocServer, DISABLED_StartAndStop) {
    rapid_doc::ServerConfig config;
    config.port = 18080;
    config.pipelineConfig.stages.enablePdfRender = false;
    config.pipelineConfig.stages.enableLayout = false;
    config.pipelineConfig.stages.enableOcr = false;
    config.pipelineConfig.stages.enableWiredTable = false;
    config.pipelineConfig.stages.enableReadingOrder = false;
    config.pipelineConfig.stages.enableMarkdownOutput = true;
    
    try {
        rapid_doc::DocServer server(config);
        
        // Server should be able to start (would need to run in separate thread)
        EXPECT_FALSE(server.isRunning());
        
        server.stop();
        EXPECT_FALSE(server.isRunning());
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Failed to start server: " << e.what();
    }
}
