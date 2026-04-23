#include "common/config.h"
#include "common/logger.h"
#include <filesystem>

namespace fs = std::filesystem;

namespace rapid_doc {

PipelineConfig PipelineConfig::Default(const std::string& projectRoot) {
    PipelineConfig cfg;

    // Layout models (align with Python README lane DXNN/ONNX cache)
    cfg.models.layoutDxnnModel = projectRoot + "/.download_cache/dxnn_models/pp_doclayout_l_part1.dxnn";
    cfg.models.layoutOnnxSubModel = projectRoot + "/.download_cache/onnx_models/pp_doclayout_l_part2.onnx";

    // Table models (wired only; align with Python README lane DXNN cache)
    cfg.models.tableUnetDxnnModel = projectRoot + "/.download_cache/dxnn_models/unet.dxnn";

    // OCR models (align with Python README lane DXNN cache)
    cfg.models.ocrModelDir = projectRoot + "/.download_cache/dxnn_models";
    cfg.models.ocrDictPath = projectRoot + "/3rd-party/DXNN-OCR-cpp/engine/model_files/ppocrv5_dict.txt";

    // Formula model (same README lane as Python demo_offline.py --finegrained)
    cfg.models.formulaOnnxModel = projectRoot + "/.download_cache/onnx_models/pp_formulanet_plus_m.onnx";

    // Table classifier (PaddleCls) — wired/wireless routing (mirrors Python
    // rapid_table_self/table_cls/main.py). Optional: if the file is missing
    // at runtime, TableRecognizer falls back to the legacy lineRatio heuristic.
    cfg.models.tableClsOnnxModel = projectRoot + "/.download_cache/onnx_models/paddle_cls.onnx";

    // SLANet+ wireless table structure model (mirrors Python
    // rapid_table_self/model/slanet-plus.onnx). Optional: if missing at
    // runtime, the wireless backend is disabled and pipeline falls back to
    // the "[Unsupported table]" placeholder (same behavior as before port).
    cfg.models.slanetPlusOnnxModel = projectRoot + "/.download_cache/onnx_models/slanet-plus.onnx";

    // Layout input size for pp_doclayout_l model
    cfg.runtime.layoutInputSize = 640;

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

    // Formula model check
    if (stages.enableFormula) {
        if (!fs::exists(models.formulaOnnxModel))
            return "Formula ONNX model not found: " + models.formulaOnnxModel;
    }

    // Table classifier model check (non-fatal if missing when enabled: the
    // recognizer falls back to the geometric heuristic). We therefore only
    // emit an informational log upstream, not a validation error here.

    // SLANet+ wireless table model check (non-fatal if missing when enabled:
    // the wireless backend is disabled and the pipeline falls back to the
    // unsupported-table placeholder — same behavior as before the port).

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
    LOG_INFO("  Formula:          {}", stages.enableFormula ? "ON" : "OFF");
    LOG_INFO("  Wireless Table:   {}", stages.enableWirelessTable ? "ON" : "OFF");
    LOG_INFO("  Table Classify:   {}", stages.enableTableClassify ? "ON" : "OFF");
    LOG_INFO("Models:");
    LOG_INFO("  Layout DXNN:      {}", models.layoutDxnnModel);
    LOG_INFO("  Layout ONNX post: {}", models.layoutOnnxSubModel);
    LOG_INFO("  Table UNET:       {}", models.tableUnetDxnnModel);
    LOG_INFO("  OCR model dir:    {}", models.ocrModelDir);
    LOG_INFO("  Formula ONNX:     {}", models.formulaOnnxModel);
    LOG_INFO("  TableCls ONNX:    {}", models.tableClsOnnxModel);
    LOG_INFO("  SLANet+ ONNX:     {}", models.slanetPlusOnnxModel);
    LOG_INFO("Runtime:");
    LOG_INFO("  PDF DPI:          {}", runtime.pdfDpi);
    LOG_INFO("  Max pages:        {}", runtime.maxPages);
    LOG_INFO("  Start page:       {}", runtime.startPageId);
    LOG_INFO("  End page:         {}", runtime.endPageId);
    LOG_INFO("  Device ID:        {}", runtime.deviceId);
    LOG_INFO("  Output dir:       {}", runtime.outputDir);
    LOG_INFO("========================================");
}

} // namespace rapid_doc
