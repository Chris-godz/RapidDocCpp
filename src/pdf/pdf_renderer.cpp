/**
 * @file pdf_renderer.cpp
 * @brief PDF rendering implementation using Poppler
 *
 * Renders PDF pages as OpenCV BGR Mat images.
 * Reference: DXNN-OCR-cpp/server/pdf_handler.cpp
 */

#include "pdf/pdf_renderer.h"
#include "common/logger.h"

#include <poppler/cpp/poppler-document.h>
#include <poppler/cpp/poppler-page.h>
#include <poppler/cpp/poppler-page-renderer.h>
#include <poppler/cpp/poppler-image.h>

#include <fstream>
#include <filesystem>
#include <future>
#include <algorithm>

namespace fs = std::filesystem;

namespace rapid_doc {

static constexpr double POINTS_PER_INCH = 72.0;

struct PdfRenderer::Impl {
    // No persistent state needed; Poppler objects are created per-call
};

PdfRenderer::PdfRenderer(const PdfRenderConfig& config)
    : config_(config)
    , impl_(std::make_unique<Impl>())
{
}

PdfRenderer::~PdfRenderer() = default;

// ---- Single-page rendering (thread-safe) ----

static PageImage renderSinglePage(poppler::document* doc, int pageIndex,
                                   int dpi, size_t maxPixels)
{
    PageImage result;
    result.pageIndex = pageIndex;
    result.dpi = dpi;

    std::unique_ptr<poppler::page> page(doc->create_page(pageIndex));
    if (!page) {
        LOG_ERROR("Failed to load PDF page {}", pageIndex);
        return result;
    }

    poppler::rectf rect = page->page_rect();
    double widthPts  = rect.width();
    double heightPts = rect.height();

    result.pdfWidth  = static_cast<int>(widthPts);
    result.pdfHeight = static_cast<int>(heightPts);

    double scale = dpi / POINTS_PER_INCH;
    int renderW = static_cast<int>(widthPts * scale);
    int renderH = static_cast<int>(heightPts * scale);

    // Safety check: pixel count
    if (static_cast<size_t>(renderW) * renderH > maxPixels) {
        LOG_WARN("Page {} would be {}x{} ({:.1f}M px), exceeding limit — downscaling",
                 pageIndex, renderW, renderH,
                 renderW * renderH / 1e6);
        double ratio = std::sqrt(static_cast<double>(maxPixels) / (renderW * renderH));
        renderW = static_cast<int>(renderW * ratio);
        renderH = static_cast<int>(renderH * ratio);
    }

    result.scaleFactor = scale;

    // Render with Poppler
    poppler::page_renderer renderer;
    renderer.set_render_hint(poppler::page_renderer::antialiasing, true);
    renderer.set_render_hint(poppler::page_renderer::text_antialiasing, true);

    poppler::image img = renderer.render_page(page.get(), dpi, dpi);
    if (!img.is_valid()) {
        LOG_ERROR("Poppler render failed for page {}", pageIndex);
        return result;
    }

    // Convert Poppler image → OpenCV BGR Mat
    int imgW = img.width();
    int imgH = img.height();
    int bytesPerRow = img.bytes_per_row();
    const char* imgData = img.const_data();

    poppler::image::format_enum fmt = img.format();
    if (fmt == poppler::image::format_argb32) {
        cv::Mat bgra(imgH, imgW, CV_8UC4,
                     const_cast<char*>(imgData), bytesPerRow);
        cv::cvtColor(bgra, result.image, cv::COLOR_BGRA2BGR);
    } else if (fmt == poppler::image::format_rgb24) {
        cv::Mat rgb(imgH, imgW, CV_8UC3,
                    const_cast<char*>(imgData), bytesPerRow);
        cv::cvtColor(rgb, result.image, cv::COLOR_RGB2BGR);
    } else {
        // Fallback: treat as BGRA
        LOG_WARN("Unknown Poppler format {} for page {}, treating as BGRA",
                 static_cast<int>(fmt), pageIndex);
        cv::Mat raw(imgH, imgW, CV_8UC4,
                    const_cast<char*>(imgData), bytesPerRow);
        cv::cvtColor(raw, result.image, cv::COLOR_BGRA2BGR);
    }

    return result;
}

// ---- Public API ----

std::vector<PageImage> PdfRenderer::renderFile(const std::string& pdfPath) {
    LOG_INFO("PDF render: loading file {}", pdfPath);

    if (!fs::exists(pdfPath)) {
        LOG_ERROR("PDF file not found: {}", pdfPath);
        return {};
    }

    std::ifstream file(pdfPath, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Cannot open PDF file: {}", pdfPath);
        return {};
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size);

    return renderFromMemory(data.data(), data.size());
}

std::vector<PageImage> PdfRenderer::renderFromMemory(const uint8_t* data, size_t size) {
    LOG_INFO("PDF render: {} bytes, dpi={}", size, config_.dpi);

    poppler::document* doc = poppler::document::load_from_raw_data(
        reinterpret_cast<const char*>(data), static_cast<int>(size));

    if (!doc) {
        LOG_ERROR("Poppler failed to load PDF from memory ({} bytes)", size);
        return {};
    }

    if (doc->is_locked()) {
        LOG_ERROR("PDF is password-protected");
        delete doc;
        return {};
    }

    int totalPages = doc->pages();
    LOG_INFO("PDF loaded: {} pages", totalPages);

    if (totalPages <= 0) {
        delete doc;
        return {};
    }

    int pagesToRender = totalPages;
    if (config_.maxPages > 0 && config_.maxPages < totalPages) {
        pagesToRender = config_.maxPages;
        LOG_INFO("Limiting render to {} of {} pages", pagesToRender, totalPages);
    }

    // Parallel rendering using std::async
    std::vector<std::future<PageImage>> futures;
    futures.reserve(pagesToRender);

    for (int i = 0; i < pagesToRender; ++i) {
        futures.push_back(std::async(std::launch::async,
            renderSinglePage, doc, i, config_.dpi, config_.maxPixelsPerPage));
    }

    std::vector<PageImage> pages;
    pages.reserve(pagesToRender);

    for (auto& f : futures) {
        auto page = f.get();
        if (!page.image.empty()) {
            pages.push_back(std::move(page));
        } else {
            LOG_WARN("Skipping page {} — render failed", page.pageIndex);
        }
    }

    delete doc;
    LOG_INFO("PDF rendering complete: {}/{} pages", pages.size(), pagesToRender);
    return pages;
}

int PdfRenderer::getPageCount(const std::string& pdfPath) {
    if (!fs::exists(pdfPath)) {
        LOG_ERROR("PDF file not found: {}", pdfPath);
        return -1;
    }

    std::ifstream file(pdfPath, std::ios::binary);
    if (!file.is_open()) return -1;

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(size);
    file.read(reinterpret_cast<char*>(data.data()), size);

    poppler::document* doc = poppler::document::load_from_raw_data(
        reinterpret_cast<const char*>(data.data()), static_cast<int>(data.size()));

    if (!doc) return -1;

    int count = doc->pages();
    delete doc;
    return count;
}

} // namespace rapid_doc
