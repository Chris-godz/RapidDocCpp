#pragma once

/**
 * @file pdf_renderer.h
 * @brief PDF to image rendering using Poppler
 * 
 * Renders PDF pages as OpenCV Mat images for pipeline processing.
 * Supports parallel page rendering with concurrency control.
 * Reuses Poppler integration pattern from DXNN-OCR-cpp server.
 */

#include "common/types.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <functional>

namespace rapid_doc {

/**
 * @brief PDF rendering configuration
 */
struct PdfRenderConfig {
    int dpi = 200;                  // Rendering resolution
    int maxPages = 0;               // Max pages to render (0 = all)
    int startPageId = 0;            // Inclusive start page (0-based)
    int endPageId = -1;             // Inclusive end page (-1 = all)
    int maxConcurrentRenders = 4;   // Parallel rendering limit
    int maxDpi = 300;               // Safety limit
    size_t maxPixelsPerPage = 25000000; // 25M pixels safety limit
};

/**
 * @brief PDF page renderer using Poppler
 */
class PdfRenderer {
public:
    using PageVisitor = std::function<bool(PageImage&&)>;

    explicit PdfRenderer(const PdfRenderConfig& config = {});
    ~PdfRenderer();

    /**
     * @brief Render all pages from a PDF file
     * @param pdfPath Path to PDF file
     * @return Vector of rendered page images
     */
    std::vector<PageImage> renderFile(const std::string& pdfPath);
    bool renderFileStreaming(const std::string& pdfPath, const PageVisitor& visitor);

    /**
     * @brief Render all pages from PDF data in memory
     * @param data Raw PDF bytes
     * @param size Data size in bytes
     * @return Vector of rendered page images
     */
    std::vector<PageImage> renderFromMemory(const uint8_t* data, size_t size);
    bool renderFromMemoryStreaming(const uint8_t* data, size_t size, const PageVisitor& visitor);

    /**
     * @brief Get total page count without rendering
     * @param pdfPath Path to PDF file
     * @return Number of pages, or -1 on error
     */
    int getPageCount(const std::string& pdfPath);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    PdfRenderConfig config_;
};

} // namespace rapid_doc
