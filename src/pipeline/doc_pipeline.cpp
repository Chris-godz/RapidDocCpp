/**
 * @file doc_pipeline.cpp
 * @brief Main document processing pipeline implementation
 * 
 * Orchestrates PDF rendering → Layout → OCR → Table → Reading Order → Output
 * Uses DXNN-OCR-cpp via submodule for OCR functionality.
 */

#include "pipeline/doc_pipeline.h"
#include "../../3rd-party/DXNN-OCR-cpp/include/pipeline/ocr_pipeline.h"  // From DXNN-OCR-cpp

// Undefine OCR logger macros to use RapidDocCpp logger
#undef LOG_TRACE
#undef LOG_DEBUG
#undef LOG_INFO
#undef LOG_WARN
#undef LOG_ERROR

#include "common/logger.h"
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <thread>
#include <unordered_map>
#include <map>
#include <sstream>

namespace fs = std::filesystem;

namespace rapid_doc {

namespace {

// Generate HTML table from TableCell grid, using row/col spans and OCR content.
std::string generateTableHtmlFromCells(const std::vector<TableCell>& cells) {
    if (cells.empty()) return "";

    int maxRow = 0;
    int maxCol = 0;
    for (const auto& cell : cells) {
        maxRow = std::max(maxRow, cell.row + cell.rowSpan);
        maxCol = std::max(maxCol, cell.col + cell.colSpan);
    }
    if (maxRow <= 0 || maxCol <= 0) return "";

    std::vector<std::vector<bool>> occupied(maxRow, std::vector<bool>(maxCol, false));
    std::map<std::pair<int, int>, const TableCell*> cellMap;
    for (const auto& cell : cells) {
        if (cell.row < 0 || cell.col < 0) continue;
        cellMap[{cell.row, cell.col}] = &cell;
    }

    std::ostringstream html;
    html << "<table border=\"1\">\n";

    for (int r = 0; r < maxRow; ++r) {
        html << "  <tr>\n";
        for (int c = 0; c < maxCol; ++c) {
            if (occupied[r][c]) continue;

            auto it = cellMap.find({r, c});
            if (it != cellMap.end()) {
                const TableCell* cell = it->second;
                int rowSpan = std::max(1, cell->rowSpan);
                int colSpan = std::max(1, cell->colSpan);

                for (int dr = 0; dr < rowSpan; ++dr) {
                    for (int dc = 0; dc < colSpan; ++dc) {
                        int nr = r + dr;
                        int nc = c + dc;
                        if (nr < maxRow && nc < maxCol) {
                            occupied[nr][nc] = true;
                        }
                    }
                }

                html << "    <td";
                if (rowSpan > 1) html << " rowspan=\"" << rowSpan << "\"";
                if (colSpan > 1) html << " colspan=\"" << colSpan << "\"";
                html << ">";

                // Escape HTML special characters in content
                std::string escapedText;
                for (char ch : cell->content) {
                    switch (ch) {
                        case '<': escapedText += "&lt;"; break;
                        case '>': escapedText += "&gt;"; break;
                        case '&': escapedText += "&amp;"; break;
                        case '"': escapedText += "&quot;"; break;
                        default: escapedText += ch;
                    }
                }
                html << escapedText;
                html << "</td>\n";
            } else {
                occupied[r][c] = true;
                html << "    <td></td>\n";
            }
        }
        html << "  </tr>\n";
    }

    html << "</table>";
    return html.str();
}

} // namespace

DocPipeline::DocPipeline(const PipelineConfig& config)
    : config_(config)
{
}

DocPipeline::~DocPipeline() {
    if (ocrPipeline_ && ocrRunning_) {
        ocrPipeline_->stop();
        ocrRunning_ = false;
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

    // Initialize Table recognizer (wired & wireless tables)
    if (config_.stages.enableWiredTable || config_.stages.enableWirelessTable) {
        TableRecognizerConfig tableCfg;
        tableCfg.unetDxnnModelPath = config_.models.tableUnetDxnnModel;
        tableCfg.tableClsOnnxModelPath = config_.models.tableClsOnnxModel;
        tableCfg.tableSlanetOnnxModelPath = config_.models.tableSlanetOnnxModel;
        tableCfg.tableSlanetDictPath = config_.models.tableSlanetDictPath;
        tableCfg.enableTableClassify = config_.stages.enableTableClassify;
        tableCfg.enableWirelessTable = config_.stages.enableWirelessTable;

        tableRecognizer_ = std::make_unique<TableRecognizer>(tableCfg);
        if (!tableRecognizer_->initialize()) {
            LOG_ERROR("Failed to initialize table recognizer");
            return false;
        }
        LOG_INFO("Table recognizer initialized");
    }

    // Initialize Formula recognizer
    if (config_.stages.enableFormula) {
        FormulaRecognizerConfig formulaCfg;
        formulaCfg.onnxModelPath = config_.models.formulaOnnxModel;
        formulaCfg.dictPath = config_.models.formulaDictPath;
        formulaRecognizer_ = std::make_unique<FormulaRecognizer>(formulaCfg);
        if (!formulaRecognizer_->initialize()) {
            LOG_ERROR("Failed to initialize formula recognizer");
            return false;
        }
        LOG_INFO("Formula recognizer initialized");
    }

    // Initialize OCR pipeline (from DXNN-OCR-cpp)
    if (config_.stages.enableOcr) {
        ocr::OCRPipelineConfig ocrCfg;
        // Configure OCR model paths
        ocrCfg.detectorConfig.model640Path = config_.models.ocrModelDir + "/det_v5_640.dxnn";
        ocrCfg.detectorConfig.model960Path = config_.models.ocrModelDir + "/det_v5_960.dxnn";
        
        ocrCfg.recognizerConfig.modelPaths.clear();
        ocrCfg.recognizerConfig.modelPaths[3] = config_.models.ocrModelDir + "/rec_v5_ratio_3.dxnn";
        ocrCfg.recognizerConfig.modelPaths[5] = config_.models.ocrModelDir + "/rec_v5_ratio_5.dxnn";
        ocrCfg.recognizerConfig.modelPaths[10] = config_.models.ocrModelDir + "/rec_v5_ratio_10.dxnn";
        ocrCfg.recognizerConfig.modelPaths[15] = config_.models.ocrModelDir + "/rec_v5_ratio_15.dxnn";
        ocrCfg.recognizerConfig.modelPaths[25] = config_.models.ocrModelDir + "/rec_v5_ratio_25.dxnn";
        ocrCfg.recognizerConfig.modelPaths[35] = config_.models.ocrModelDir + "/rec_v5_ratio_35.dxnn";
        ocrCfg.recognizerConfig.dictPath = config_.models.ocrDictPath;

        ocrCfg.classifierConfig.modelPath = config_.models.ocrModelDir + "/textline_ori.dxnn";

        ocrCfg.docPreprocessingConfig.orientationConfig.modelPath = config_.models.ocrModelDir + "/doc_ori_fixed.dxnn";
        ocrCfg.docPreprocessingConfig.uvdocConfig.modelPath = config_.models.ocrModelDir + "/UVDoc_pruned_p3.dxnn";
        
        ocrPipeline_ = std::make_unique<ocr::OCRPipeline>(ocrCfg);
        if (!ocrPipeline_->initialize()) {
            LOG_ERROR("Failed to initialize OCR pipeline");
            return false;
        }
        ocrPipeline_->start();
        ocrRunning_ = true;
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
        result.layoutResult = layoutDetector_->detect(image);

        LOG_DEBUG("Page {}: detected {} layout boxes ({:.2f} ms)", 
                  pageImage.pageIndex, result.layoutResult.boxes.size(),
                  result.layoutResult.inferenceTimeMs);
    }

    // Step 2: Process each layout category
    auto textBoxes = result.layoutResult.getTextBoxes();
    auto tableBoxes = result.layoutResult.getTableBoxes();
    auto figureBoxes = result.layoutResult.getBoxesByCategory(LayoutCategory::IMAGE);
    auto chartBoxes = result.layoutResult.getBoxesByCategory(LayoutCategory::CHART);
    figureBoxes.insert(figureBoxes.end(), chartBoxes.begin(), chartBoxes.end());
    auto unsupportedBoxes = result.layoutResult.getUnsupportedBoxes();

    // Split unsupported boxes into formula and others
    std::vector<LayoutBox> formulaBoxes;
    std::vector<LayoutBox> trulyUnsupportedBoxes;
    for (const auto& box : unsupportedBoxes) {
        if (box.category == LayoutCategory::FORMULA || 
            box.category == LayoutCategory::FORMULA_NUMBER) {
            formulaBoxes.push_back(box);
        } else {
            trulyUnsupportedBoxes.push_back(box);
        }
    }

    // OCR on text regions
    if (ocrPipeline_ && config_.stages.enableOcr) {
        auto ocrElements = runOcrOnRegions(image, textBoxes);
        result.elements.insert(result.elements.end(), ocrElements.begin(), ocrElements.end());
    }

    // Table recognition
    if (tableRecognizer_ && config_.stages.enableWiredTable) {
        auto tableElements = runTableRecognition(image, tableBoxes, formulaBoxes, figureBoxes);
        result.elements.insert(result.elements.end(), tableElements.begin(), tableElements.end());
    }

    // Save extracted figures
    if (config_.runtime.saveImages) {
        saveExtractedImages(image, figureBoxes, pageImage.pageIndex, result.elements);
    }

    // Formula recognition
    if (config_.stages.enableFormula) {
        auto formulaElements = runFormulaRecognition(image, formulaBoxes);
        result.elements.insert(result.elements.end(), formulaElements.begin(), formulaElements.end());
    } else {
        // If disabled, just treat them as unsupported
        trulyUnsupportedBoxes.insert(trulyUnsupportedBoxes.end(), formulaBoxes.begin(), formulaBoxes.end());
    }

    // Handle unsupported elements
    auto skipElements = handleUnsupportedElements(trulyUnsupportedBoxes);
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

std::vector<ContentElement> DocPipeline::runOcrOnRegions(
    const cv::Mat& image,
    const std::vector<LayoutBox>& textBoxes)
{
    std::vector<ContentElement> elements;
    if (textBoxes.empty() || !ocrPipeline_) return elements;

    // To keep track of which element corresponds to which task
    std::unordered_map<int64_t, ContentElement> taskToElement;

    int64_t taskId = 0;
    for (const auto& box : textBoxes) {
        cv::Rect roi = box.toRect() & cv::Rect(0, 0, image.cols, image.rows);
        if (roi.width <= 0 || roi.height <= 0) continue;

        cv::Mat crop = image(roi);

        ContentElement elem;
        elem.type = (box.category == LayoutCategory::PARAGRAPH_TITLE ||
                    box.category == LayoutCategory::DOC_TITLE) 
                    ? ContentElement::Type::TITLE 
                    : ContentElement::Type::TEXT;
        elem.layoutBox = box;
        elem.confidence = 0.0f; // To be updated by OCR

        // Submit task (wait if queue is full)
        while (!ocrPipeline_->pushTask(crop, taskId)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }

        taskToElement[taskId] = elem;
        taskId++;
    }

    size_t pendingTasks = taskToElement.size();
    while (pendingTasks > 0) {
        std::vector<ocr::PipelineOCRResult> ocrResults;
        int64_t completedTaskId = -1;
        bool success = false;
        
        if (ocrPipeline_->getResult(ocrResults, completedTaskId, nullptr, &success)) {
            if (taskToElement.find(completedTaskId) != taskToElement.end()) {
                ContentElement& elem = taskToElement[completedTaskId];
                
                if (success && !ocrResults.empty()) {
                    std::string combinedText;
                    float sumConf = 0.0f;
                    for (size_t i = 0; i < ocrResults.size(); ++i) {
                        if (i > 0) combinedText += "\n";
                        combinedText += ocrResults[i].text;
                        sumConf += ocrResults[i].confidence;
                    }
                    elem.text = combinedText;
                    elem.confidence = sumConf / ocrResults.size();
                }
                
                elements.push_back(elem);
                pendingTasks--;
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }

    return elements;
}

void DocPipeline::runTableCellOcr(
    const cv::Mat& tableImage,
    const cv::Rect& tableRoi,
    TableResult& tableResult,
    const std::vector<LayoutBox>& formulaBoxes,
    const std::vector<LayoutBox>& figureBoxes)
{
    if (!ocrPipeline_ || tableImage.empty() || tableResult.cells.empty()) {
        return;
    }

    // 1. Create a copy of the table image and paint white over formulas and figures
    cv::Mat maskedImage = tableImage.clone();
    
    // Helper to paint a layout box white if it intersects with table
    auto applyMask = [&](const std::vector<LayoutBox>& boxes) {
        for (const auto& b : boxes) {
            cv::Rect boxRect = b.toRect();
            cv::Rect intersect = boxRect & tableRoi;
            if (intersect.area() > 0) {
                // Convert intersection rect from page coordinates to table crop coordinates
                cv::Rect localRect(
                    intersect.x - tableRoi.x,
                    intersect.y - tableRoi.y,
                    intersect.width,
                    intersect.height
                );
                cv::rectangle(maskedImage, localRect, cv::Scalar(255, 255, 255), cv::FILLED);
            }
        }
    };
    
    applyMask(formulaBoxes);
    applyMask(figureBoxes);

    // 2. Submit a single OCR task for the masked whole table crop
    int64_t taskId = 0;
    while (!ocrPipeline_->pushTask(maskedImage, taskId)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }

    size_t pendingTasks = 1;
    std::vector<ocr::PipelineOCRResult> ocrResults;

    while (pendingTasks > 0) {
        ocrResults.clear();
        int64_t completedTaskId = -1;
        bool success = false;

        if (ocrPipeline_->getResult(ocrResults, completedTaskId, nullptr, &success)) {
            if (completedTaskId == taskId) {
                pendingTasks--;

                if (!success || ocrResults.empty()) {
                    return;
                }

                // Precompute cell rectangles (table-local coordinates)
                std::vector<cv::Rect> cellRects;
                cellRects.reserve(tableResult.cells.size());
                for (const auto& cell : tableResult.cells) {
                    int x = static_cast<int>(cell.x0);
                    int y = static_cast<int>(cell.y0);
                    int w = static_cast<int>(cell.x1 - cell.x0);
                    int h = static_cast<int>(cell.y1 - cell.y0);
                    cv::Rect rect(x, y, w, h);
                    rect &= cv::Rect(0, 0, tableImage.cols, tableImage.rows);
                    cellRects.push_back(rect);
                }

                struct CellTextPiece {
                    float y;
                    float x;
                    std::string text;
                };

                std::vector<std::vector<CellTextPiece>> cellPieces(tableResult.cells.size());

                // Assign each OCR box to the best matching cell
                const float containThresh = 0.6f;
                for (const auto& res : ocrResults) {
                    if (res.text.empty()) continue;

                    cv::Rect ocrRect = res.getBoundingRect();
                    ocrRect &= cv::Rect(0, 0, tableImage.cols, tableImage.rows);
                    if (ocrRect.width <= 0 || ocrRect.height <= 0) continue;

                    int bestIndex = -1;
                    float bestRatio = 0.0f;

                    for (size_t i = 0; i < cellRects.size(); ++i) {
                        const cv::Rect& cellRect = cellRects[i];
                        if (cellRect.width <= 0 || cellRect.height <= 0) continue;

                        cv::Rect inter = ocrRect & cellRect;
                        if (inter.width <= 0 || inter.height <= 0) continue;

                        float ratio = static_cast<float>(inter.area()) /
                                      static_cast<float>(std::max(1, ocrRect.area()));
                        if (ratio >= containThresh && ratio > bestRatio) {
                            bestRatio = ratio;
                            bestIndex = static_cast<int>(i);
                        }
                    }

                    if (bestIndex >= 0) {
                        cv::Point2f center = res.getCenter();
                        cellPieces[static_cast<size_t>(bestIndex)].push_back(
                            CellTextPiece{center.y, center.x, res.text}
                        );
                    }
                }

                // Sort and join text pieces within each cell
                for (size_t i = 0; i < tableResult.cells.size(); ++i) {
                    auto& pieces = cellPieces[i];
                    if (pieces.empty()) continue;

                    std::sort(pieces.begin(), pieces.end(),
                              [](const CellTextPiece& a, const CellTextPiece& b) {
                                  if (std::abs(a.y - b.y) > 5.0f) {
                                      return a.y < b.y;
                                  }
                                  return a.x < b.x;
                              });

                    std::string merged;
                    for (size_t k = 0; k < pieces.size(); ++k) {
                        if (k > 0) {
                            if (std::abs(pieces[k].y - pieces[k - 1].y) > 5.0f) {
                                merged += "\n";
                            } else {
                                merged += " ";
                            }
                        }
                        merged += pieces[k].text;
                    }

                    tableResult.cells[i].content = std::move(merged);
                }
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    }
}

std::vector<ContentElement> DocPipeline::runTableRecognition(
    const cv::Mat& image,
    const std::vector<LayoutBox>& tableBoxes,
    const std::vector<LayoutBox>& formulaBoxes,
    const std::vector<LayoutBox>& figureBoxes)
{
    std::vector<ContentElement> elements;

    for (const auto& box : tableBoxes) {
        ContentElement elem;
        elem.type = ContentElement::Type::TABLE;
        elem.layoutBox = box;

        // Estimate table type
        cv::Rect roi = box.toRect() & cv::Rect(0, 0, image.cols, image.rows);
        cv::Mat tableCrop = image(roi);
        
        TableType type;
        if (tableRecognizer_ && config_.stages.enableTableClassify) {
            type = tableRecognizer_->classifyTableType(tableCrop);
        } else {
            type = TableRecognizer::estimateTableType(tableCrop);
        }
        
        if (type == TableType::WIRELESS) {
            if (config_.stages.enableWirelessTable && tableRecognizer_) {
                TableResult tableResult = tableRecognizer_->recognizeWireless(tableCrop);
                elem.skipped = !tableResult.supported;

                if (tableResult.supported &&
                    config_.stages.enableOcr &&
                    config_.runtime.enableTableCellOcr &&
                    ocrPipeline_) {
                    runTableCellOcr(tableCrop, roi, tableResult, formulaBoxes, figureBoxes);
                }

                if (!tableResult.cells.empty()) {
                    std::string htmlWithContent = generateTableHtmlFromCells(tableResult.cells);
                    elem.html = htmlWithContent.empty() ? tableResult.html : std::move(htmlWithContent);
                } else {
                    elem.html = tableResult.html;
                }
            } else {
                // Wireless table not supported — skip with placeholder
                elem.skipped = true;
                elem.html = "<!-- Wireless table: NPU not supported -->";
                LOG_WARN("Skipping wireless table at ({}, {}) — NPU not supported", box.x0, box.y0);
            }
        } else {
            // Wired table — run recognition
            if (tableRecognizer_) {
                TableResult tableResult = tableRecognizer_->recognize(tableCrop);
                elem.skipped = !tableResult.supported;

                // Optionally run OCR on table cells and regenerate HTML with text
                if (tableResult.supported &&
                    config_.stages.enableOcr &&
                    config_.runtime.enableTableCellOcr &&
                    ocrPipeline_) {
                    runTableCellOcr(tableCrop, roi, tableResult, formulaBoxes, figureBoxes);
                }

                if (!tableResult.cells.empty()) {
                    std::string htmlWithContent = generateTableHtmlFromCells(tableResult.cells);
                    elem.html = htmlWithContent.empty() ? tableResult.html : std::move(htmlWithContent);
                } else {
                    elem.html = tableResult.html;
                }
            }
        }

        elements.push_back(elem);
    }

    return elements;
}

std::vector<ContentElement> DocPipeline::runFormulaRecognition(
    const cv::Mat& image,
    const std::vector<LayoutBox>& formulaBoxes)
{
    std::vector<ContentElement> elements;

    for (const auto& box : formulaBoxes) {
        ContentElement elem;
        elem.layoutBox = box;
        elem.type = ContentElement::Type::EQUATION;
        
        if (formulaRecognizer_ && config_.stages.enableFormula) {
            // Expand crop slightly
            int pad = 5;
            int x = std::max(0, static_cast<int>(box.x0) - pad);
            int y = std::max(0, static_cast<int>(box.y0) - pad);
            int w = std::min(image.cols - x, static_cast<int>(box.x1 - box.x0) + 2 * pad);
            int h = std::min(image.rows - y, static_cast<int>(box.y1 - box.y0) + 2 * pad);
            
            cv::Rect roi(x, y, w, h);
            cv::Mat crop = image(roi).clone();

            FormulaResult formulaResult = formulaRecognizer_->recognize(crop);
            if (formulaResult.success && !formulaResult.latex.empty()) {
                elem.text = "$$" + formulaResult.latex + "$$";
                elem.skipped = false;
            } else {
                elem.text = "[Formula recognition failed]";
                elem.skipped = true;
            }
        } else {
            elem.text = "[Formula: ONNX Runtime not available or disabled]";
            elem.skipped = true;
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
        ContentElement elem;
        elem.layoutBox = box;
        elem.skipped = true;
        
        if (box.category == LayoutCategory::FORMULA || 
            box.category == LayoutCategory::FORMULA_NUMBER) {
            elem.type = ContentElement::Type::EQUATION;
            elem.text = "[Formula: DEEPX NPU does not support formula recognition]";
        } else {
            elem.type = ContentElement::Type::UNKNOWN;
            elem.text = "[Unsupported element type]";
        }

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

        cv::Mat figureCrop = image(roi);
        
        std::string filename = "page" + std::to_string(pageIndex) + 
                               "_fig" + std::to_string(i) + ".png";
        std::string filepath = config_.runtime.outputDir + "/" + filename;
        
        cv::imwrite(filepath, figureCrop);

        ContentElement elem;
        elem.type = ContentElement::Type::IMAGE;
        elem.layoutBox = box;
        elem.imagePath = filename;
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
