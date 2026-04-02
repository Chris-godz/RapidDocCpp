/**
 * @file pdf_renderer.cpp
 * @brief PDF rendering implementation using Poppler C++ API.
 *
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
#include <memory>
#include <algorithm>
#include <optional>
#include <utility>

namespace rapid_doc {

struct PdfRenderer::Impl {
    // stateless — each call creates its own poppler objects
};

namespace {

struct RenderRange {
    int totalPages = 0;
    int startPage = 0;
    int endPage = -1;
    int pagesToRender = 0;
};

std::optional<RenderRange> resolveRenderRange(
    poppler::document& doc,
    const PdfRenderConfig& config)
{
    RenderRange range;
    range.totalPages = doc.pages();
    range.startPage = std::max(0, config.startPageId);
    if (range.startPage >= range.totalPages) {
        LOG_WARN("Start page {} exceeds total pages {}", range.startPage, range.totalPages);
        return std::nullopt;
    }

    range.endPage = (config.endPageId < 0)
                        ? (range.totalPages - 1)
                        : std::min(config.endPageId, range.totalPages - 1);
    if (range.endPage < range.startPage) {
        LOG_WARN("Invalid page range: start={}, end={}", range.startPage, range.endPage);
        return std::nullopt;
    }

    if (config.maxPages > 0) {
        range.endPage = std::min(range.endPage, range.startPage + config.maxPages - 1);
    }

    range.pagesToRender = range.endPage - range.startPage + 1;
    return range;
}

std::optional<PageImage> renderPageImage(
    poppler::document& doc,
    const PdfRenderConfig& config,
    int pageNo)
{
    auto renderStart = std::chrono::steady_clock::now();
    std::unique_ptr<poppler::page> page(doc.create_page(pageNo));
    if (!page) {
        LOG_WARN("Failed to create page {}", pageNo);
        return std::nullopt;
    }

    poppler::rectf rect = page->page_rect();

    poppler::page_renderer renderer;
    renderer.set_render_hint(poppler::page_renderer::antialiasing, true);
    renderer.set_render_hint(poppler::page_renderer::text_antialiasing, true);

    poppler::image img = renderer.render_page(page.get(), config.dpi, config.dpi);
    if (!img.is_valid()) {
        LOG_WARN("Failed to render page {}", pageNo);
        return std::nullopt;
    }

    int imgW = img.width();
    int imgH = img.height();
    int bpr  = img.bytes_per_row();
    const char* raw = img.const_data();
    auto fmt = img.format();

    cv::Mat bgr;
    if (fmt == poppler::image::format_argb32) {
        cv::Mat bgra(imgH, imgW, CV_8UC4, const_cast<char*>(raw), bpr);
        cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
    } else if (fmt == poppler::image::format_rgb24) {
        cv::Mat rgb(imgH, imgW, CV_8UC3, const_cast<char*>(raw), bpr);
        cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    } else {
        cv::Mat bgra(imgH, imgW, CV_8UC4, const_cast<char*>(raw), bpr);
        cv::cvtColor(bgra, bgr, cv::COLOR_BGRA2BGR);
    }

    PageImage pi;
    pi.image = std::move(bgr);
    pi.pageIndex = pageNo;
    pi.dpi = config.dpi;
    pi.scaleFactor = config.dpi / 72.0;
    pi.pdfWidth = static_cast<int>(rect.width());
    pi.pdfHeight = static_cast<int>(rect.height());
    pi.renderTimeMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - renderStart).count();

    LOG_DEBUG("Page {}: {}x{} px (pdf {}x{} pt)", pageNo, imgW, imgH,
              pi.pdfWidth, pi.pdfHeight);
    return pi;
}

} // namespace

PdfRenderer::PdfRenderer(const PdfRenderConfig& config)
    : impl_(std::make_unique<Impl>())
    , config_(config)
{}

PdfRenderer::~PdfRenderer() = default;

std::vector<PageImage> PdfRenderer::renderFile(const std::string& pdfPath) {
    std::vector<PageImage> results;
    renderFileStreaming(pdfPath, [&results](PageImage&& page) {
        results.push_back(std::move(page));
        return true;
    });
    return results;
}

bool PdfRenderer::renderFileStreaming(const std::string& pdfPath, const PageVisitor& visitor) {
    LOG_INFO("PDF render: loading file {}", pdfPath);

    if (!std::filesystem::exists(pdfPath)) {
        LOG_ERROR("PDF file not found: {}", pdfPath);
        return false;
    }

    std::ifstream file(pdfPath, std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Cannot open PDF file: {}", pdfPath);
        return false;
    }

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<uint8_t> data(size);
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));

    return renderFromMemoryStreaming(data.data(), data.size(), visitor);
}

std::vector<PageImage> PdfRenderer::renderFromMemory(const uint8_t* data, size_t size) {
    std::vector<PageImage> results;
    renderFromMemoryStreaming(data, size, [&results](PageImage&& page) {
        results.push_back(std::move(page));
        return true;
    });
    return results;
}

bool PdfRenderer::renderFromMemoryStreaming(
    const uint8_t* data,
    size_t size,
    const PageVisitor& visitor)
{
    LOG_INFO("PDF render: {} bytes, dpi={}", size, config_.dpi);

    std::unique_ptr<poppler::document> doc(
        poppler::document::load_from_raw_data(
            reinterpret_cast<const char*>(data), static_cast<int>(size)));

    if (!doc) {
        LOG_ERROR("Failed to load PDF document");
        return false;
    }

    if (doc->is_locked()) {
        LOG_ERROR("PDF is password protected");
        return false;
    }

    const auto range = resolveRenderRange(*doc, config_);
    if (!range.has_value()) {
        return false;
    }

    LOG_INFO("PDF: {} total pages, rendering {} pages ({}-{})",
             range->totalPages,
             range->pagesToRender,
             range->startPage,
             range->endPage);

    for (int pageNo = range->startPage; pageNo <= range->endPage; ++pageNo) {
        auto page = renderPageImage(*doc, config_, pageNo);
        if (!page.has_value()) {
            continue;
        }
        if (!visitor(std::move(*page))) {
            return false;
        }
    }

    return true;
}

int PdfRenderer::getPageCount(const std::string& pdfPath) {
    std::ifstream file(pdfPath, std::ios::binary);
    if (!file.is_open()) return -1;

    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> data(size);
    file.read(data.data(), static_cast<std::streamsize>(size));

    std::unique_ptr<poppler::document> doc(
        poppler::document::load_from_raw_data(data.data(), static_cast<int>(size)));
    return doc ? doc->pages() : -1;
}

} // namespace rapid_doc
