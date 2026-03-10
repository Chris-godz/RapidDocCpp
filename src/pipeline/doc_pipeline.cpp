/**
 * @file doc_pipeline.cpp
 * @brief Main document processing pipeline implementation
 * 
 * Orchestrates PDF rendering → Layout → OCR → Table → Reading Order → Output
 * Uses DXNN-OCR-cpp via submodule for OCR functionality.
 */

#include "pipeline/doc_pipeline.h"
#include "common/logger.h"
#include <filesystem>
#include <chrono>
#include <thread>
#include <algorithm>

namespace fs = std::filesystem;

namespace rapid_doc {

DocPipeline::DocPipeline(const PipelineConfig& config)
    : config_(config)
{
}

DocPipeline::~DocPipeline() {
    if (ocrPipeline_) {
        ocrPipeline_->stop();
    }
}

bool DocPipeline::initialize() {
    LOG_INFO("Initializing RapidDoc pipeline...");
    config_.show();

    // Validate configuration
    std::string err = config_.validate();
    if (!err.empty()) {
        LOG_ERROR("Configuration validation failed: {}", err);
        return false;
    }

    // Initialize PDF renderer
    if (config_.stages.enablePdfRender) {
        PdfRenderConfig pdfCfg;
        pdfCfg.dpi = config_.runtime.pdfDpi;
        pdfCfg.maxPages = config_.runtime.maxPages;
        pdfCfg.maxConcurrentRenders = config_.runtime.maxConcurrentPages;
        pdfRenderer_ = std::make_unique<PdfRenderer>(pdfCfg);
        LOG_INFO("PDF renderer initialized");
    }

    // Initialize Layout detector
    if (config_.stages.enableLayout) {
        LayoutDetectorConfig layoutCfg;
        layoutCfg.dxnnModelPath = config_.models.layoutDxnnModel;
        layoutCfg.onnxSubModelPath = config_.models.layoutOnnxSubModel;
        layoutCfg.inputSize = config_.runtime.layoutInputSize;
        layoutCfg.confThreshold = config_.runtime.layoutConfThreshold;
        layoutDetector_ = std::make_unique<LayoutDetector>(layoutCfg);
        if (!layoutDetector_->initialize()) {
            LOG_ERROR("Failed to initialize layout detector");
            return false;
        }
        LOG_INFO("Layout detector initialized");
    }

    // Initialize Table recognizer (wired tables only)
    if (config_.stages.enableWiredTable) {
        TableRecognizerConfig tableCfg;
        tableCfg.unetDxnnModelPath = config_.models.tableUnetDxnnModel;
        tableCfg.threshold = config_.runtime.tableConfThreshold;
        tableRecognizer_ = std::make_unique<TableRecognizer>(tableCfg);
        if (!tableRecognizer_->initialize()) {
            LOG_ERROR("Failed to initialize table recognizer");
            return false;
        }
        LOG_INFO("Table recognizer initialized (wired tables only)");
    }

    // Initialize OCR pipeline (from DXNN-OCR-cpp)
    if (config_.stages.enableOcr) {
        ocr::OCRPipelineConfig ocrCfg;

        // Detection model paths — use only 640 model to conserve NPU memory
        ocrCfg.detectorConfig.model640Path = config_.models.ocrModelDir + "/det_v5_640.dxnn";
        ocrCfg.detectorConfig.model960Path = "";
        ocrCfg.detectorConfig.sizeThreshold = 99999;

        // Recognition model paths
        std::string mdir = config_.models.ocrModelDir;
        ocrCfg.recognizerConfig.modelPaths = {
            {3,  mdir + "/rec_v5_ratio_3.dxnn"},
            {5,  mdir + "/rec_v5_ratio_5.dxnn"},
            {10, mdir + "/rec_v5_ratio_10.dxnn"},
            {15, mdir + "/rec_v5_ratio_15.dxnn"},
            {25, mdir + "/rec_v5_ratio_25.dxnn"},
            {35, mdir + "/rec_v5_ratio_35.dxnn"},
        };
        ocrCfg.recognizerConfig.dictPath = config_.models.ocrDictPath;

        // Disable heavy document-level preprocessing for per-region OCR
        ocrCfg.useDocPreprocessing = false;
        ocrCfg.useClassification = false;
        ocrCfg.enableVisualization = false;

        ocrPipeline_ = std::make_unique<ocr::OCRPipeline>(ocrCfg);
        if (!ocrPipeline_->initialize()) {
            LOG_ERROR("Failed to initialize OCR pipeline");
            return false;
        }
        ocrPipeline_->start();
        LOG_INFO("OCR pipeline initialized (DXNN-OCR-cpp)");
    }

    // Create output directory
    if (!fs::exists(config_.runtime.outputDir)) {
        fs::create_directories(config_.runtime.outputDir);
    }

    initialized_ = true;
    LOG_INFO("RapidDoc pipeline initialized successfully");
    return true;
}

DocumentResult DocPipeline::processPdf(const std::string& pdfPath) {
    LOG_INFO("Processing PDF: {}", pdfPath);
    auto startTime = std::chrono::steady_clock::now();

    DocumentResult result;

    if (!initialized_) {
        LOG_ERROR("Pipeline not initialized");
        return result;
    }

    // Step 1: Render PDF pages
    reportProgress("PDF Render", 0, 1);
    auto renderStart = std::chrono::steady_clock::now();
    
    std::vector<PageImage> pageImages;
    if (pdfRenderer_) {
        pageImages = pdfRenderer_->renderFile(pdfPath);
    }
    
    auto renderEnd = std::chrono::steady_clock::now();
    result.stats.pdfRenderTimeMs = std::chrono::duration<double, std::milli>(renderEnd - renderStart).count();
    result.totalPages = static_cast<int>(pageImages.size());

    if (pageImages.empty()) {
        LOG_WARN("No pages rendered from PDF");
        return result;
    }

    LOG_INFO("Rendered {} pages from PDF", pageImages.size());

    // Step 2: Process each page
    for (size_t i = 0; i < pageImages.size(); i++) {
        reportProgress("Processing", static_cast<int>(i + 1), static_cast<int>(pageImages.size()));
        
        PageResult pageResult = processPage(pageImages[i]);
        result.pages.push_back(std::move(pageResult));
        result.processedPages++;
    }

    // Step 3: Generate output
    reportProgress("Output", 0, 1);
    auto outputStart = std::chrono::steady_clock::now();

    if (config_.stages.enableMarkdownOutput) {
        result.markdown = markdownWriter_.generate(result);
    }
    result.contentListJson = contentListWriter_.generate(result);

    auto outputEnd = std::chrono::steady_clock::now();
    result.stats.outputGenTimeMs = std::chrono::duration<double, std::milli>(outputEnd - outputStart).count();

    // Calculate total time
    auto endTime = std::chrono::steady_clock::now();
    result.totalTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    // Count skipped elements
    for (const auto& page : result.pages) {
        for (const auto& elem : page.elements) {
            if (elem.skipped) result.skippedElements++;
        }
    }

    LOG_INFO("Document processing complete: {} pages, {} skipped elements, {:.1f}ms",
             result.processedPages, result.skippedElements, result.totalTimeMs);

    return result;
}

DocumentResult DocPipeline::processPdfFromMemory(const uint8_t* data, size_t size) {
    LOG_INFO("Processing PDF from memory: {} bytes", size);
    
    DocumentResult result;
    if (!initialized_) {
        LOG_ERROR("Pipeline not initialized");
        return result;
    }

    // Render from memory
    std::vector<PageImage> pageImages;
    if (pdfRenderer_) {
        pageImages = pdfRenderer_->renderFromMemory(data, size);
    }
    
    result.totalPages = static_cast<int>(pageImages.size());

    for (size_t i = 0; i < pageImages.size(); i++) {
        PageResult pageResult = processPage(pageImages[i]);
        result.pages.push_back(std::move(pageResult));
        result.processedPages++;
    }

    if (config_.stages.enableMarkdownOutput) {
        result.markdown = markdownWriter_.generate(result);
    }
    result.contentListJson = contentListWriter_.generate(result);

    return result;
}

PageResult DocPipeline::processImage(const cv::Mat& image, int pageIndex) {
    LOG_INFO("Processing image: {}x{}, page {}", image.cols, image.rows, pageIndex);

    PageImage pageImage;
    pageImage.image = image.clone();
    pageImage.pageIndex = pageIndex;
    pageImage.dpi = config_.runtime.pdfDpi;
    pageImage.scaleFactor = 1.0;
    pageImage.pdfWidth = image.cols;
    pageImage.pdfHeight = image.rows;

    return processPage(pageImage);
}

PageResult DocPipeline::processPage(const PageImage& pageImage) {
    auto startTime = std::chrono::steady_clock::now();
    PageResult result;
    result.pageIndex = pageImage.pageIndex;

    const cv::Mat& image = pageImage.image;
    int pageWidth = image.cols;
    int pageHeight = image.rows;

    // Step 1: Layout detection
    if (layoutDetector_ && config_.stages.enableLayout) {
        auto layoutStart = std::chrono::steady_clock::now();
        result.layoutResult = layoutDetector_->detect(image);
        auto layoutEnd = std::chrono::steady_clock::now();
        result.layoutResult.inferenceTimeMs = 
            std::chrono::duration<double, std::milli>(layoutEnd - layoutStart).count();

        LOG_DEBUG("Page {}: detected {} layout boxes", 
                  pageImage.pageIndex, result.layoutResult.boxes.size());
    }

    // Step 2: Process each layout category
    auto textBoxes = result.layoutResult.getTextBoxes();
    auto tableBoxes = result.layoutResult.getTableBoxes();
    auto figureBoxes = result.layoutResult.getBoxesByCategory(LayoutCategory::FIGURE);
    // Equation boxes: saved as images like Python (no ONNX model on NPU, use image fallback)
    auto equationBoxes = result.layoutResult.getEquationBoxes();
    auto unsupportedBoxes = result.layoutResult.getUnsupportedBoxes();

    // OCR on text regions
    if (ocrPipeline_ && config_.stages.enableOcr) {
        auto ocrElements = runOcrOnRegions(image, textBoxes);
        result.elements.insert(result.elements.end(), ocrElements.begin(), ocrElements.end());
    }

    // Table recognition
    if (tableRecognizer_ && config_.stages.enableWiredTable) {
        auto tableElements = runTableRecognition(image, tableBoxes);
        result.elements.insert(result.elements.end(), tableElements.begin(), tableElements.end());
    }

    // Handle figure/image regions
    saveExtractedImages(image, figureBoxes, pageImage.pageIndex, result.elements);

    // Formula: save crop as image (matches Python — no LaTeX, just image fallback)
    // Python renders formulas as ![]() even with formulanet enabled
    saveFormulaImages(image, equationBoxes, pageImage.pageIndex, result.elements);

    // Handle truly unsupported elements (non-formula)
    auto skipElements = handleUnsupportedElements(unsupportedBoxes);
    result.elements.insert(result.elements.end(), skipElements.begin(), skipElements.end());

    // Step 3: Reading order sort
    if (config_.stages.enableReadingOrder && !result.elements.empty()) {
        // Extract layout boxes from elements for sorting
        std::vector<LayoutBox> sortBoxes;
        for (const auto& elem : result.elements) {
            sortBoxes.push_back(elem.layoutBox);
        }

        auto sortedIndices = xycutPlusSort(sortBoxes, pageWidth, pageHeight);
        
        // Reorder elements
        std::vector<ContentElement> sortedElements;
        sortedElements.reserve(result.elements.size());
        for (int i = 0; i < static_cast<int>(sortedIndices.size()); i++) {
            int idx = sortedIndices[i];
            result.elements[idx].readingOrder = i;
            sortedElements.push_back(result.elements[idx]);
        }
        result.elements = std::move(sortedElements);
    }

    auto endTime = std::chrono::steady_clock::now();
    result.totalTimeMs = std::chrono::duration<double, std::milli>(endTime - startTime).count();

    return result;
}

// ---------------------------------------------------------------------------
// OCR helper: submit one crop and block-wait for result
// ---------------------------------------------------------------------------
std::string DocPipeline::ocrOnCrop(const cv::Mat& crop, int64_t taskId) {
    if (!ocrPipeline_ || crop.empty()) return "";

    if (!ocrPipeline_->pushTask(crop, taskId))
        return "";

    std::vector<ocr::PipelineOCRResult> ocrResults;
    int64_t resultId = -1;
    bool success = false;
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
    while (!ocrPipeline_->getResult(ocrResults, resultId, nullptr, &success)) {
        if (std::chrono::steady_clock::now() > deadline) {
            LOG_WARN("OCR timeout for task {}", taskId);
            return "";
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    if (!success || ocrResults.empty()) return "";

    std::string combined;
    for (const auto& r : ocrResults) {
        if (!combined.empty()) combined += "\n";
        combined += r.text;
    }
    return combined;
}

// ---------------------------------------------------------------------------
// OCR on text regions detected by layout
// ---------------------------------------------------------------------------
std::vector<ContentElement> DocPipeline::runOcrOnRegions(
    const cv::Mat& image,
    const std::vector<LayoutBox>& textBoxes)
{
    std::vector<ContentElement> elements;

    for (size_t bi = 0; bi < textBoxes.size(); ++bi) {
        const auto& box = textBoxes[bi];
        ContentElement elem;
        elem.type = (box.category == LayoutCategory::TITLE)
                    ? ContentElement::Type::TITLE
                    : ContentElement::Type::TEXT;
        elem.layoutBox = box;
        elem.confidence = box.confidence;

        cv::Rect roi = box.toRect() & cv::Rect(0, 0, image.cols, image.rows);
        if (roi.width <= 0 || roi.height <= 0) {
            elem.skipped = true;
            elements.push_back(elem);
            continue;
        }

        elem.text = ocrOnCrop(image(roi).clone(), static_cast<int64_t>(bi));
        elements.push_back(elem);
    }

    return elements;
}

// ---------------------------------------------------------------------------
// Table recognition with cell-level OCR
// ---------------------------------------------------------------------------
std::vector<ContentElement> DocPipeline::runTableRecognition(
    const cv::Mat& image,
    const std::vector<LayoutBox>& tableBoxes)
{
    std::vector<ContentElement> elements;
    int64_t cellTaskId = 10000;

    for (const auto& box : tableBoxes) {
        ContentElement elem;
        elem.type = ContentElement::Type::TABLE;
        elem.layoutBox = box;

        cv::Rect roi = box.toRect() & cv::Rect(0, 0, image.cols, image.rows);
        if (roi.width <= 0 || roi.height <= 0) {
            elem.skipped = true;
            elements.push_back(elem);
            continue;
        }

        cv::Mat tableCrop = image(roi);

        // In UNET-only mode (matching Python), run UNET on ALL tables
        // without wired/wireless classification.
        if (tableRecognizer_) {
            TableResult tableResult = tableRecognizer_->recognize(tableCrop);

            // Match approach (like Python match_ocr_cell):
            // 1. Run OCR on the FULL table image once
            // 2. Match each OCR text box to the nearest cell by spatial overlap
            if (ocrPipeline_ && config_.stages.enableOcr && !tableResult.cells.empty()) {
                std::string tableOcrText = ocrOnCrop(tableCrop.clone(), cellTaskId++);

                // Get the raw OCR results with bounding boxes
                // Re-submit for detailed box-level results
                ocrPipeline_->pushTask(tableCrop.clone(), cellTaskId);
                std::vector<ocr::PipelineOCRResult> ocrBoxes;
                int64_t rid = -1;
                bool ok = false;
                auto dl = std::chrono::steady_clock::now() + std::chrono::seconds(30);
                while (!ocrPipeline_->getResult(ocrBoxes, rid, nullptr, &ok)) {
                    if (std::chrono::steady_clock::now() > dl) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(5));
                }
                cellTaskId++;

                if (ok && !ocrBoxes.empty()) {
                    // Match each OCR box to cells by containment/overlap
                    for (const auto& ocrRes : ocrBoxes) {
                        if (ocrRes.text.empty()) continue;
                        // Get OCR box bounding rect
                        cv::Rect ocrRect = ocrRes.getBoundingRect();

                        // Find the best matching cell
                        int bestCell = -1;
                        float bestOverlap = 0.0f;
                        for (size_t ci = 0; ci < tableResult.cells.size(); ++ci) {
                            auto& c = tableResult.cells[ci];
                            cv::Rect cellRect(static_cast<int>(c.x0), static_cast<int>(c.y0),
                                              static_cast<int>(c.x1 - c.x0),
                                              static_cast<int>(c.y1 - c.y0));
                            cv::Rect inter = ocrRect & cellRect;
                            if (inter.width <= 0 || inter.height <= 0) continue;
                            float interArea = static_cast<float>(inter.area());
                            float ocrArea = static_cast<float>(std::max(1, ocrRect.area()));
                            float overlap = interArea / ocrArea;
                            if (overlap > bestOverlap) {
                                bestOverlap = overlap;
                                bestCell = static_cast<int>(ci);
                            }
                        }
                        if (bestCell >= 0 && bestOverlap > 0.3f) {
                            auto& cell = tableResult.cells[bestCell];
                            if (!cell.content.empty()) cell.content += "\n";
                            cell.content += ocrRes.text;
                        }
                    }
                }

                tableResult.html = tableRecognizer_->generateHtml(tableResult.cells);
            }

            elem.html = tableResult.html;
            elem.skipped = !tableResult.supported;
        }

        elements.push_back(elem);
    }

    return elements;
}

std::vector<ContentElement> DocPipeline::handleUnsupportedElements(
    const std::vector<LayoutBox>& unsupportedBoxes)
{
    std::vector<ContentElement> elements;

    for (const auto& box : unsupportedBoxes) {
        // Equations are handled by saveFormulaImages(), skip here
        if (box.category == LayoutCategory::EQUATION ||
            box.category == LayoutCategory::INTERLINE_EQUATION)
            continue;

        ContentElement elem;
        elem.layoutBox = box;
        elem.skipped = true;
        elem.type = ContentElement::Type::UNKNOWN;
        elem.text = "[Unsupported element type]";

        LOG_DEBUG("Skipping unsupported element: {} at ({}, {})",
                  layoutCategoryToString(box.category), box.x0, box.y0);

        elements.push_back(elem);
    }

    return elements;
}

void DocPipeline::saveExtractedImages(
    const cv::Mat& image,
    const std::vector<LayoutBox>& figureBoxes,
    int pageIndex,
    std::vector<ContentElement>& elements)
{
    for (size_t i = 0; i < figureBoxes.size(); i++) {
        const auto& box = figureBoxes[i];
        cv::Rect roi = box.toRect() & cv::Rect(0, 0, image.cols, image.rows);
        if (roi.width <= 0 || roi.height <= 0) continue;

        std::string filename = "images/page" + std::to_string(pageIndex) +
                               "_fig" + std::to_string(i) + ".png";

        if (config_.runtime.saveImages) {
            cv::Mat figureCrop = image(roi);
            std::string filepath = config_.runtime.outputDir + "/" + filename;
            std::filesystem::create_directories(
                std::filesystem::path(filepath).parent_path());
            cv::imwrite(filepath, figureCrop);
        }

        ContentElement elem;
        elem.type = ContentElement::Type::IMAGE;
        elem.layoutBox = box;
        elem.imagePath = filename;
        elem.pageIndex = pageIndex;
        elements.push_back(elem);
    }
}

void DocPipeline::saveFormulaImages(
    const cv::Mat& image,
    const std::vector<LayoutBox>& equationBoxes,
    int pageIndex,
    std::vector<ContentElement>& elements)
{
    // Python behavior: formula regions are saved as images, rendered as ![]()
    // No LaTeX recognition (onnx model not available on NPU).
    for (size_t i = 0; i < equationBoxes.size(); i++) {
        const auto& box = equationBoxes[i];
        cv::Rect roi = box.toRect() & cv::Rect(0, 0, image.cols, image.rows);
        if (roi.width <= 0 || roi.height <= 0) continue;

        std::string filename = "images/page" + std::to_string(pageIndex) +
                               "_eq" + std::to_string(i) + ".png";

        if (config_.runtime.saveImages) {
            cv::Mat crop = image(roi);
            std::string filepath = config_.runtime.outputDir + "/" + filename;
            std::filesystem::create_directories(
                std::filesystem::path(filepath).parent_path());
            cv::imwrite(filepath, crop);
        }

        ContentElement elem;
        elem.type = ContentElement::Type::EQUATION;
        elem.layoutBox = box;
        elem.imagePath = filename;  // store path for image-based formula output
        elem.pageIndex = pageIndex;
        elements.push_back(elem);
    }
}

void DocPipeline::reportProgress(const std::string& stage, int current, int total) {
    if (progressCallback_) {
        progressCallback_(stage, current, total);
    }
}

} // namespace rapid_doc
