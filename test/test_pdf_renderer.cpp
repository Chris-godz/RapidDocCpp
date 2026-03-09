/**
 * @file test_pdf_renderer.cpp
 * @brief Tests for PdfRenderer using Poppler
 *
 * These tests validate the PDF rendering module independently
 * of model inference — no NPU or models required.
 */

#include <gtest/gtest.h>
#include "pdf/pdf_renderer.h"
#include <filesystem>
#include <fstream>
#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

namespace {

// Path to a sample PDF available via DXNN-OCR-cpp submodule
const char* SAMPLE_PDF = PROJECT_ROOT_DIR
    "/3rd-party/DXNN-OCR-cpp/server/webui/examples_pdf/3M-7770.pdf";

bool samplePdfExists() {
    return fs::exists(SAMPLE_PDF);
}

std::vector<uint8_t> readFileBytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return {};
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> data(sz);
    f.read(reinterpret_cast<char*>(data.data()), sz);
    return data;
}

} // anonymous namespace

// ========================================
// Basic Rendering Tests
// ========================================

TEST(PdfRenderer, RenderFileProducesPages) {
    if (!samplePdfExists()) GTEST_SKIP() << "Sample PDF not found";

    rapid_doc::PdfRenderConfig cfg;
    cfg.dpi = 150;
    cfg.maxPages = 3;

    rapid_doc::PdfRenderer renderer(cfg);
    auto pages = renderer.renderFile(SAMPLE_PDF);

    ASSERT_FALSE(pages.empty()) << "Expected at least 1 rendered page";

    for (size_t i = 0; i < pages.size(); ++i) {
        const auto& p = pages[i];
        EXPECT_FALSE(p.image.empty()) << "Page " << i << " image is empty";
        EXPECT_EQ(p.image.channels(), 3) << "Expected BGR image";
        EXPECT_EQ(p.pageIndex, static_cast<int>(i));
        EXPECT_GT(p.image.cols, 0);
        EXPECT_GT(p.image.rows, 0);
    }
}

TEST(PdfRenderer, RenderFromMemory) {
    if (!samplePdfExists()) GTEST_SKIP() << "Sample PDF not found";

    auto data = readFileBytes(SAMPLE_PDF);
    ASSERT_FALSE(data.empty());

    rapid_doc::PdfRenderConfig cfg;
    cfg.dpi = 100;
    cfg.maxPages = 1;

    rapid_doc::PdfRenderer renderer(cfg);
    auto pages = renderer.renderFromMemory(data.data(), data.size());

    ASSERT_EQ(pages.size(), 1u);
    EXPECT_FALSE(pages[0].image.empty());
    EXPECT_EQ(pages[0].image.channels(), 3);
}

TEST(PdfRenderer, GetPageCount) {
    if (!samplePdfExists()) GTEST_SKIP() << "Sample PDF not found";

    rapid_doc::PdfRenderer renderer;
    int count = renderer.getPageCount(SAMPLE_PDF);
    EXPECT_GT(count, 0) << "Expected positive page count";
}

TEST(PdfRenderer, DpiAffectsResolution) {
    if (!samplePdfExists()) GTEST_SKIP() << "Sample PDF not found";

    rapid_doc::PdfRenderConfig cfgLow, cfgHigh;
    cfgLow.dpi = 72;
    cfgLow.maxPages = 1;
    cfgHigh.dpi = 200;
    cfgHigh.maxPages = 1;

    rapid_doc::PdfRenderer rendererLow(cfgLow);
    rapid_doc::PdfRenderer rendererHigh(cfgHigh);

    auto pagesLow = rendererLow.renderFile(SAMPLE_PDF);
    auto pagesHigh = rendererHigh.renderFile(SAMPLE_PDF);

    ASSERT_EQ(pagesLow.size(), 1u);
    ASSERT_EQ(pagesHigh.size(), 1u);

    // Higher DPI should produce larger images
    EXPECT_GT(pagesHigh[0].image.cols, pagesLow[0].image.cols);
    EXPECT_GT(pagesHigh[0].image.rows, pagesLow[0].image.rows);
}

// ========================================
// Error Handling Tests
// ========================================

TEST(PdfRenderer, NonexistentFile) {
    rapid_doc::PdfRenderer renderer;
    auto pages = renderer.renderFile("/nonexistent/path.pdf");
    EXPECT_TRUE(pages.empty());
}

TEST(PdfRenderer, InvalidData) {
    rapid_doc::PdfRenderer renderer;
    uint8_t garbage[] = {0x00, 0xFF, 0xAB, 0xCD};
    auto pages = renderer.renderFromMemory(garbage, sizeof(garbage));
    EXPECT_TRUE(pages.empty());
}

TEST(PdfRenderer, MaxPagesLimit) {
    if (!samplePdfExists()) GTEST_SKIP() << "Sample PDF not found";

    rapid_doc::PdfRenderConfig cfg;
    cfg.dpi = 72;
    cfg.maxPages = 2;

    rapid_doc::PdfRenderer renderer(cfg);
    auto pages = renderer.renderFile(SAMPLE_PDF);

    // Should not exceed maxPages
    EXPECT_LE(static_cast<int>(pages.size()), cfg.maxPages);
}
