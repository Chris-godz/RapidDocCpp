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

namespace rapid_doc {

struct PdfRenderer::Impl {
    // stateless — each call creates its own poppler objects
};

PdfRenderer::PdfRenderer(const PdfRenderConfig& config)
    : config_(config)
    , impl_(std::make_unique<Impl>())
{}

PdfRenderer::~PdfRenderer() = default;

std::vector<PageImage> PdfRenderer::renderFile(const std::string& pdfPath) {
    LOG_INFO("PDF render: loading file {}", pdfPath);

    if (!std::filesystem::exists(pdfPath)) {
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
    file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(size));

    return renderFromMemory(data.data(), data.size());
}

std::vector<PageImage> PdfRenderer::renderFromMemory(const uint8_t* data, size_t size) {
    LOG_INFO("PDF render: {} bytes, dpi={}", size, config_.dpi);

    std::unique_ptr<poppler::document> doc(
        poppler::document::load_from_raw_data(
            reinterpret_cast<const char*>(data), static_cast<int>(size)));

    if (!doc) {
        LOG_ERROR("Failed to load PDF document");
        return {};
    }

    if (doc->is_locked()) {
        LOG_ERROR("PDF is password protected");
        return {};
    }

    int totalPages = doc->pages();
    int startPage = std::max(0, config_.startPageId);
    if (startPage >= totalPages) {
        LOG_WARN("Start page {} exceeds total pages {}", startPage, totalPages);
        return {};
    }

    int endPage = (config_.endPageId < 0)
                      ? (totalPages - 1)
                      : std::min(config_.endPageId, totalPages - 1);
    if (endPage < startPage) {
        LOG_WARN("Invalid page range: start={}, end={}", startPage, endPage);
        return {};
    }

    if (config_.maxPages > 0) {
        endPage = std::min(endPage, startPage + config_.maxPages - 1);
    }

    int pagesToRender = endPage - startPage + 1;

    LOG_INFO("PDF: {} total pages, rendering {} pages ({}-{})",
             totalPages, pagesToRender, startPage, endPage);

    std::vector<PageImage> results;
    results.reserve(pagesToRender);

    for (int pageNo = startPage; pageNo <= endPage; ++pageNo) {
        std::unique_ptr<poppler::page> page(doc->create_page(pageNo));
        if (!page) {
            LOG_WARN("Failed to create page {}", pageNo);
            continue;
        }

        poppler::rectf rect = page->page_rect();

        poppler::page_renderer renderer;
        renderer.set_render_hint(poppler::page_renderer::antialiasing, true);
        renderer.set_render_hint(poppler::page_renderer::text_antialiasing, true);

        poppler::image img = renderer.render_page(page.get(), config_.dpi, config_.dpi);
        if (!img.is_valid()) {
            LOG_WARN("Failed to render page {}", pageNo);
            continue;
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
        pi.image       = bgr.clone();
        pi.pageIndex   = pageNo;
        pi.dpi         = config_.dpi;
        pi.scaleFactor = config_.dpi / 72.0;
        pi.pdfWidth    = static_cast<int>(rect.width());
        pi.pdfHeight   = static_cast<int>(rect.height());

        results.push_back(std::move(pi));
        LOG_DEBUG("Page {}: {}x{} px (pdf {}x{} pt)", pageNo, imgW, imgH,
                  pi.pdfWidth, pi.pdfHeight);
    }

    return results;
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
