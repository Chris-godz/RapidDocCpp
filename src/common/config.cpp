#include "common/config.h"
#include "common/logger.h"
#include <filesystem>
#include <algorithm>
#include <cctype>

namespace fs = std::filesystem;

namespace rapid_doc {

const char* pipelineModeToString(PipelineMode mode) {
    switch (mode) {
        case PipelineMode::Serial:
            return "serial";
        case PipelineMode::PagePipelineMvp:
            return "page_pipeline_mvp";
        default:
            return "serial";
    }
}

bool parsePipelineMode(const std::string& raw, PipelineMode& mode) {
    std::string value = raw;
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (value == "serial") {
        mode = PipelineMode::Serial;
        return true;
    }
    if (value == "page_pipeline_mvp") {
        mode = PipelineMode::PagePipelineMvp;
        return true;
    }
    return false;
}

const char* ocrOuterModeToString(OcrOuterMode mode) {
    switch (mode) {
        case OcrOuterMode::ImmediatePerTask:
            return "immediate_per_task";
        case OcrOuterMode::ShadowWindowedCollect:
            return "shadow_windowed_collect";
        default:
            return "immediate_per_task";
    }
}

bool parseOcrOuterMode(const std::string& raw, OcrOuterMode& mode) {
    std::string value = raw;
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (value == "immediate_per_task") {
        mode = OcrOuterMode::ImmediatePerTask;
        return true;
    }
    if (value == "shadow_windowed_collect") {
        mode = OcrOuterMode::ShadowWindowedCollect;
        return true;
    }
    return false;
}

PipelineConfig PipelineConfig::Default(const std::string& projectRoot) {
    PipelineConfig cfg;

    // Layout models
    cfg.models.layoutDxnnModel = projectRoot + "/engine/model_files/layout/pp_doclayout_l_part1.dxnn";
    cfg.models.layoutOnnxSubModel = projectRoot + "/engine/model_files/layout/pp_doclayout_l_part2.onnx";

    // Table models (wired only)
    cfg.models.tableUnetDxnnModel = projectRoot + "/engine/model_files/table/unet.dxnn";

    // OCR models (managed by DXNN-OCR-cpp)
    cfg.models.ocrModelDir = projectRoot + "/3rd-party/DXNN-OCR-cpp/engine/model_files/server";
    cfg.models.ocrDictPath = projectRoot + "/3rd-party/DXNN-OCR-cpp/engine/model_files/ppocrv5_dict.txt";

    // Layout input size for pp_doclayout_l model
    cfg.runtime.layoutInputSize = 640;

    return cfg;
}

std::string PipelineConfig::validate() const {
    if (runtime.stageQueueRendered == 0 ||
        runtime.stageQueuePlanned == 0 ||
        runtime.stageQueueOcrTable == 0 ||
        runtime.stageQueueFinalize == 0) {
        return "Pipeline stage queue capacities must be positive";
    }
    if (runtime.ocrShadowWindow == 0) {
        return "OCR shadow window must be positive";
    }

    // Layout model check
    if (stages.enableLayout) {
        if (!fs::exists(models.layoutDxnnModel))
            return "Layout DXNN model not found: " + models.layoutDxnnModel;
        if (!fs::exists(models.layoutOnnxSubModel))
            return "Layout ONNX sub-model not found: " + models.layoutOnnxSubModel;
    }

    // Table model check
    if (stages.enableWiredTable) {
        if (!fs::exists(models.tableUnetDxnnModel))
            return "Table UNET model not found: " + models.tableUnetDxnnModel;
    }

    // OCR model directory check
    if (stages.enableOcr) {
        if (!fs::exists(models.ocrModelDir))
            return "OCR model directory not found: " + models.ocrModelDir;
        if (!fs::exists(models.ocrDictPath))
            return "OCR dictionary not found: " + models.ocrDictPath;
    }

    return "";  // Valid
}

void PipelineConfig::show() const {
    LOG_INFO("========================================");
    LOG_INFO("RapidDoc Pipeline Configuration");
    LOG_INFO("========================================");
    LOG_INFO("Stages:");
    LOG_INFO("  PDF Render:       {}", stages.enablePdfRender ? "ON" : "OFF");
    LOG_INFO("  Layout:           {}", stages.enableLayout ? "ON" : "OFF");
    LOG_INFO("  OCR:              {}", stages.enableOcr ? "ON" : "OFF");
    LOG_INFO("  Wired Table:      {}", stages.enableWiredTable ? "ON" : "OFF");
    LOG_INFO("  Reading Order:    {}", stages.enableReadingOrder ? "ON" : "OFF");
    LOG_INFO("  Markdown Output:  {}", stages.enableMarkdownOutput ? "ON" : "OFF");
    LOG_INFO("  Formula Fallback: {}", stages.enableFormula ? "ON" : "OFF");
    LOG_INFO("  Wireless Table:   {}", stages.enableWirelessTable ? "ON" : "OFF");
    LOG_INFO("  Table Classify:   {}", stages.enableTableClassify ? "ON" : "OFF");
    LOG_INFO("Models:");
    LOG_INFO("  Layout DXNN:      {}", models.layoutDxnnModel);
    LOG_INFO("  Layout ONNX post: {}", models.layoutOnnxSubModel);
    LOG_INFO("  Table UNET:       {}", models.tableUnetDxnnModel);
    LOG_INFO("  OCR model dir:    {}", models.ocrModelDir);
    LOG_INFO("Runtime:");
    LOG_INFO("  PDF DPI:          {}", runtime.pdfDpi);
    LOG_INFO("  Max pages:        {}", runtime.maxPages);
    LOG_INFO("  Start page:       {}", runtime.startPageId);
    LOG_INFO("  End page:         {}", runtime.endPageId);
    LOG_INFO("  Device ID:        {}", runtime.deviceId);
    LOG_INFO("  Pipeline mode:    {}", pipelineModeToString(runtime.pipelineMode));
    LOG_INFO("  OCR outer mode:   {}", ocrOuterModeToString(runtime.ocrOuterMode));
    LOG_INFO("  OCR shadow win:   {}", runtime.ocrShadowWindow);
    LOG_INFO("  Queue rendered:   {}", runtime.stageQueueRendered);
    LOG_INFO("  Queue planned:    {}", runtime.stageQueuePlanned);
    LOG_INFO("  Queue ocr/table:  {}", runtime.stageQueueOcrTable);
    LOG_INFO("  Queue finalize:   {}", runtime.stageQueueFinalize);
    LOG_INFO("  Output dir:       {}", runtime.outputDir);
    LOG_INFO("========================================");
}

} // namespace rapid_doc
