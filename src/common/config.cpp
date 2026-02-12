#include "common/config.h"
#include "common/logger.h"
#include <filesystem>

namespace fs = std::filesystem;

namespace rapid_doc {

PipelineConfig PipelineConfig::Default(const std::string& projectRoot) {
    PipelineConfig cfg;

    // Layout models
    cfg.models.layoutDxnnModel = projectRoot + "/engine/model_files/layout/pp_doclayout_plus_l.dxnn";
    cfg.models.layoutOnnxSubModel = projectRoot + "/engine/model_files/layout/pp_doclayout_plus_l_post.onnx";

    // Table models (wired only)
    cfg.models.tableUnetDxnnModel = projectRoot + "/engine/model_files/table/unet.dxnn";

    // OCR models (managed by DXNN-OCR-cpp)
    cfg.models.ocrModelDir = projectRoot + "/3rd-party/DXNN-OCR-cpp/engine/model_files/server";
    cfg.models.ocrDictPath = cfg.models.ocrModelDir + "/ppocrv5_dict.txt";

    return cfg;
}

std::string PipelineConfig::validate() const {
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
    LOG_INFO("  Formula (NPU N/A):{}", stages.enableFormula ? "ON" : "OFF");
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
    LOG_INFO("  Output dir:       {}", runtime.outputDir);
    LOG_INFO("========================================");
}

} // namespace rapid_doc
