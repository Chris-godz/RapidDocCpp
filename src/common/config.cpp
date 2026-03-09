#include "common/config.h"
#include "common/logger.h"
#include <filesystem>

namespace fs = std::filesystem;

namespace rapid_doc {

PipelineConfig PipelineConfig::Default(const std::string& projectRoot) {
    PipelineConfig cfg;

    // Layout models (PP-DocLayout-L split model)
    // Python uses: dxnn_models/pp_doclayout_l_part1.dxnn + onnx_models/pp_doclayout_l_part2.onnx
    cfg.models.layoutDxnnModel = projectRoot + "/engine/model_files/layout/pp_doclayout_l_part1.dxnn";
    cfg.models.layoutOnnxSubModel = projectRoot + "/engine/model_files/layout/pp_doclayout_l_part2.onnx";

    // Table models (wired only)
    cfg.models.tableUnetDxnnModel = projectRoot + "/engine/model_files/table/unet.dxnn";

    // Table classification & wireless table
    cfg.models.tableClsOnnxModel = projectRoot + "/engine/model_files/table/paddle_cls.onnx";
    cfg.models.tableSlanetOnnxModel = projectRoot + "/engine/model_files/table/slanet-plus.onnx";
    cfg.models.tableSlanetDictPath = projectRoot + "/engine/model_files/table/slanet_dict.txt";

    // Formula recognition
    cfg.models.formulaOnnxModel = projectRoot + "/engine/model_files/formula/pp_formulanet_plus_m.onnx";
    cfg.models.formulaDictPath = projectRoot + "/engine/model_files/formula/formula_dict.txt";

    // OCR models (managed by DXNN-OCR-cpp)
    cfg.models.ocrModelDir = projectRoot + "/3rd-party/DXNN-OCR-cpp/engine/model_files/server";
    cfg.models.ocrDictPath = projectRoot + "/3rd-party/DXNN-OCR-cpp/engine/model_files/ppocrv5_dict.txt";

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
    
    if (stages.enableTableClassify) {
        if (!fs::exists(models.tableClsOnnxModel))
            return "Table Classification ONNX model not found: " + models.tableClsOnnxModel;
    }

    if (stages.enableWirelessTable) {
        if (!fs::exists(models.tableSlanetOnnxModel))
            return "Wireless Table ONNX model not found: " + models.tableSlanetOnnxModel;
        if (!fs::exists(models.tableSlanetDictPath))
            return "Wireless Table Dictionary not found: " + models.tableSlanetDictPath;
    }

    if (stages.enableFormula) {
        if (!fs::exists(models.formulaOnnxModel))
            return "Formula ONNX model not found: " + models.formulaOnnxModel;
        if (!fs::exists(models.formulaDictPath))
            return "Formula Dictionary not found: " + models.formulaDictPath;
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
    LOG_INFO("  Table cell OCR:   {}", runtime.enableTableCellOcr ? "ON" : "OFF");
    LOG_INFO("========================================");
}

} // namespace rapid_doc
