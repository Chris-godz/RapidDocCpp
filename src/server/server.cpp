/**
 * @file server.cpp
 * @brief HTTP server implementation using Crow
 */

#include "server/server.h"
#include "common/logger.h"

// Crow HTTP framework (header-only)
#ifndef CROW_MAIN
#define CROW_MAIN
#endif
#include <crow.h>

#include <nlohmann/json.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <chrono>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace fs = std::filesystem;

namespace rapid_doc {

namespace {

using json = nlohmann::json;

const std::vector<std::string> kPdfSuffixes = {".pdf"};
const std::vector<std::string> kImageSuffixes = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"};

std::string toLower(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool isPdfExtension(const std::string& extension) {
    const std::string lower = toLower(extension);
    return std::find(kPdfSuffixes.begin(), kPdfSuffixes.end(), lower) != kPdfSuffixes.end();
}

bool isImageExtension(const std::string& extension) {
    const std::string lower = toLower(extension);
    return std::find(kImageSuffixes.begin(), kImageSuffixes.end(), lower) != kImageSuffixes.end();
}

std::string contentElementTypeToString(ContentElement::Type type) {
    switch (type) {
        case ContentElement::Type::TEXT: return "text";
        case ContentElement::Type::TITLE: return "title";
        case ContentElement::Type::IMAGE: return "image";
        case ContentElement::Type::TABLE: return "table";
        case ContentElement::Type::EQUATION: return "equation";
        case ContentElement::Type::CODE: return "code";
        case ContentElement::Type::LIST: return "list";
        case ContentElement::Type::HEADER: return "header";
        case ContentElement::Type::FOOTER: return "footer";
        case ContentElement::Type::REFERENCE: return "reference";
        default: return "unknown";
    }
}

std::string safeFilename(std::string filename) {
    filename = fs::path(filename).filename().string();
    if (filename.empty()) {
        return "upload.bin";
    }

    for (char& ch : filename) {
        if (ch == '/' || ch == '\\' || ch == ':' || ch == '\0') {
            ch = '_';
        }
    }
    return filename;
}

std::string safeStem(const std::string& filename) {
    std::string stem = fs::path(filename).stem().string();
    if (stem.empty()) {
        return "document";
    }

    for (char& ch : stem) {
        if (ch == '/' || ch == '\\' || ch == ':' || ch == '\0') {
            ch = '_';
        }
    }
    return stem;
}

bool parseBool(const std::string& rawValue, bool defaultValue) {
    if (rawValue.empty()) {
        return defaultValue;
    }

    const std::string value = toLower(rawValue);
    if (value == "1" || value == "true" || value == "yes" || value == "on") {
        return true;
    }
    if (value == "0" || value == "false" || value == "no" || value == "off") {
        return false;
    }
    return defaultValue;
}

int parseInt(const std::string& rawValue, int defaultValue) {
    if (rawValue.empty()) {
        return defaultValue;
    }

    try {
        return std::stoi(rawValue);
    }
    catch (...) {
        return defaultValue;
    }
}

std::string canonicalParseMethod(const std::string& rawValue) {
    const std::string value = toLower(rawValue);
    if (value == "ocr" || value == "txt") {
        return value;
    }
    return "auto";
}

std::string mimeTypeForPath(const fs::path& path) {
    const std::string ext = toLower(path.extension().string());
    if (ext == ".png") return "image/png";
    if (ext == ".jpg" || ext == ".jpeg") return "image/jpeg";
    if (ext == ".bmp") return "image/bmp";
    if (ext == ".tif" || ext == ".tiff") return "image/tiff";
    if (ext == ".pdf") return "application/pdf";
    return "application/octet-stream";
}

void writeBinaryFile(const fs::path& path, const std::string& data) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path, std::ios::binary);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path.string());
    }
    out.write(data.data(), static_cast<std::streamsize>(data.size()));
}

void writeTextFile(const fs::path& path, const std::string& data) {
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path.string());
    }
    out << data;
}

std::string readBinaryFile(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        return {};
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    return buffer.str();
}

std::string base64Encode(const unsigned char* data, size_t len) {
    static const char kBase64Chars[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::string encoded;
    encoded.reserve(((len + 2) / 3) * 4);

    for (size_t i = 0; i < len; i += 3) {
        const unsigned int octetA = data[i];
        const unsigned int octetB = (i + 1 < len) ? data[i + 1] : 0;
        const unsigned int octetC = (i + 2 < len) ? data[i + 2] : 0;
        const unsigned int triple = (octetA << 16) | (octetB << 8) | octetC;

        encoded.push_back(kBase64Chars[(triple >> 18) & 0x3F]);
        encoded.push_back(kBase64Chars[(triple >> 12) & 0x3F]);
        encoded.push_back((i + 1 < len) ? kBase64Chars[(triple >> 6) & 0x3F] : '=');
        encoded.push_back((i + 2 < len) ? kBase64Chars[triple & 0x3F] : '=');
    }

    return encoded;
}

std::vector<uint8_t> base64Decode(const std::string& encoded) {
    static const std::string kBase64Chars =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

    std::vector<uint8_t> decoded;
    decoded.reserve(encoded.size() * 3 / 4);

    int value = 0;
    int bits = -8;
    for (unsigned char ch : encoded) {
        if (std::isspace(ch)) {
            continue;
        }
        if (ch == '=') {
            break;
        }
        const auto pos = kBase64Chars.find(static_cast<char>(ch));
        if (pos == std::string::npos) {
            continue;
        }
        value = (value << 6) + static_cast<int>(pos);
        bits += 6;
        if (bits >= 0) {
            decoded.push_back(static_cast<uint8_t>((value >> bits) & 0xFF));
            bits -= 8;
        }
    }

    return decoded;
}

std::vector<crow::multipart::part> getMultipartParts(
    const crow::multipart::message& msg,
    const std::string& name) {
    std::vector<crow::multipart::part> parts;
    const auto range = msg.part_map.equal_range(name);
    for (auto it = range.first; it != range.second; ++it) {
        parts.push_back(it->second);
    }
    return parts;
}

std::string getMultipartField(
    const crow::multipart::message& msg,
    const std::string& name,
    const std::string& defaultValue = {}) {
    const auto range = msg.part_map.equal_range(name);
    if (range.first == range.second) {
        return defaultValue;
    }
    return range.first->second.body;
}

std::vector<std::string> getMultipartFieldValues(
    const crow::multipart::message& msg,
    const std::string& name) {
    std::vector<std::string> values;
    const auto range = msg.part_map.equal_range(name);
    for (auto it = range.first; it != range.second; ++it) {
        values.push_back(it->second.body);
    }
    return values;
}

json makeEngineJson(bool tableEnabled, bool formulaEnabled) {
    return json{
        {"layout", "dxengine"},
        {"ocr", "dxengine"},
        {"formula", formulaEnabled ? "image_fallback" : "disabled"},
        {"table", tableEnabled ? "dxengine" : "disabled"},
    };
}

json makeStatsJson(const DocumentResult& result) {
    return json{
        {"pages", result.processedPages},
        {"total_pages", result.totalPages},
        {"skipped", result.skippedElements},
        {"time_ms", result.totalTimeMs},
        {"pdf_render_ms", result.stats.pdfRenderTimeMs},
        {"layout_ms", result.stats.layoutTimeMs},
        {"ocr_ms", result.stats.ocrTimeMs},
        {"table_ms", result.stats.tableTimeMs},
        {"output_gen_ms", result.stats.outputGenTimeMs},
    };
}

json buildContentListJson(const DocumentResult& result) {
    if (result.contentListJson.empty()) {
        return json::array();
    }

    try {
        return json::parse(result.contentListJson);
    }
    catch (...) {
        return json::array();
    }
}

json buildMiddleJson(const DocumentResult& result) {
    json pdfInfo = json::array();

    for (const auto& page : result.pages) {
        json elements = json::array();
        for (const auto& elem : page.elements) {
            json item{
                {"type", contentElementTypeToString(elem.type)},
                {"bbox", {elem.layoutBox.x0, elem.layoutBox.y0, elem.layoutBox.x1, elem.layoutBox.y1}},
                {"score", elem.confidence},
                {"page_idx", elem.pageIndex},
                {"reading_order", elem.readingOrder},
                {"skipped", elem.skipped},
            };
            if (!elem.text.empty()) {
                item["text"] = elem.text;
            }
            if (!elem.html.empty()) {
                item["html"] = elem.html;
            }
            if (!elem.imagePath.empty()) {
                item["image_path"] = elem.imagePath;
            }
            elements.push_back(std::move(item));
        }

        pdfInfo.push_back(json{
            {"page_idx", page.pageIndex},
            {"page_size", {page.pageWidth, page.pageHeight}},
            {"elements", std::move(elements)},
        });
    }

    return json{{"pdf_info", std::move(pdfInfo)}};
}

json buildModelJson(const DocumentResult& result) {
    json pages = json::array();

    for (const auto& page : result.pages) {
        json layoutDetections = json::array();
        for (const auto& box : page.layoutResult.boxes) {
            layoutDetections.push_back(json{
                {"category_id", box.clsId >= 0 ? box.clsId : static_cast<int>(box.category)},
                {"label", box.label.empty() ? layoutCategoryToString(box.category) : box.label},
                {"poly", {
                    static_cast<int>(box.x0), static_cast<int>(box.y0),
                    static_cast<int>(box.x1), static_cast<int>(box.y0),
                    static_cast<int>(box.x1), static_cast<int>(box.y1),
                    static_cast<int>(box.x0), static_cast<int>(box.y1)
                }},
                {"bbox", {box.x0, box.y0, box.x1, box.y1}},
                {"score", box.confidence},
            });
        }

        pages.push_back(json{
            {"layout_dets", std::move(layoutDetections)},
            {"page_info", {
                {"page_no", page.pageIndex},
                {"width", page.pageWidth},
                {"height", page.pageHeight},
            }},
        });
    }

    return pages;
}

json collectImagesAsDataUrls(const fs::path& imageDir) {
    json images = json::object();
    if (!fs::exists(imageDir)) {
        return images;
    }

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(imageDir)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    for (const auto& path : files) {
        const std::string raw = readBinaryFile(path);
        if (raw.empty()) {
            continue;
        }

        images[path.filename().string()] =
            "data:" + mimeTypeForPath(path) + ";base64," +
            base64Encode(reinterpret_cast<const unsigned char*>(raw.data()), raw.size());
    }

    return images;
}

json collectAbsoluteFiles(const fs::path& dir) {
    json files = json::array();
    if (!fs::exists(dir)) {
        return files;
    }

    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (entry.is_regular_file()) {
            paths.push_back(fs::absolute(entry.path()));
        }
    }
    std::sort(paths.begin(), paths.end());

    for (const auto& path : paths) {
        files.push_back(path.string());
    }
    return files;
}

struct FileParseOptions {
    std::string outputDir = "./output-offline";
    bool clearOutputFile = false;
    std::vector<std::string> langList = {"ch"};
    std::string backend = "pipeline";
    std::string parseMethod = "auto";
    bool formulaEnable = true;
    bool tableEnable = true;
    bool returnMd = true;
    bool returnMiddleJson = false;
    bool returnModelOutput = false;
    bool returnContentList = false;
    bool returnImages = false;
    int startPageId = 0;
    int endPageId = 99999;
    bool deepxRequested = true;
    std::string layoutEngine = "dxengine";
    std::string ocrEngine = "dxengine";
    std::string formulaEngine = "image_fallback";
    std::string tableEngine = "dxengine";
};

std::vector<std::string> collectRequestWarnings(const FileParseOptions& options) {
    std::vector<std::string> warnings;
    if (!options.deepxRequested) {
        warnings.push_back("C++ backend is DEEPX-only; 'deepx=false' was ignored.");
    }
    if (!options.layoutEngine.empty() && toLower(options.layoutEngine) != "dxengine") {
        warnings.push_back("layout_engine request ignored; C++ backend uses dxengine.");
    }
    if (!options.ocrEngine.empty() && toLower(options.ocrEngine) != "dxengine") {
        warnings.push_back("ocr_engine request ignored; C++ backend uses dxengine.");
    }
    if (!options.tableEngine.empty() && toLower(options.tableEngine) != "dxengine") {
        warnings.push_back("table_engine request ignored; C++ backend uses dxengine.");
    }
    if (!options.formulaEngine.empty() &&
        toLower(options.formulaEngine) != "onnxruntime" &&
        toLower(options.formulaEngine) != "image_fallback") {
        warnings.push_back("formula_engine request ignored; C++ backend uses image fallback.");
    }
    if (options.parseMethod == "txt") {
        warnings.push_back("parse_method=txt is not natively supported in C++; falling back to auto pipeline.");
    }
    return warnings;
}

class PipelineStateGuard {
public:
    explicit PipelineStateGuard(DocPipeline& pipeline)
        : pipeline_(pipeline)
        , originalConfig_(pipeline.config()) {}

    ~PipelineStateGuard() {
        pipeline_.setStageOptions(originalConfig_.stages);
        pipeline_.setOutputDir(originalConfig_.runtime.outputDir);
        pipeline_.setSaveImages(originalConfig_.runtime.saveImages);
        pipeline_.setSaveVisualization(originalConfig_.runtime.saveVisualization);
        pipeline_.setPageRange(originalConfig_.runtime.startPageId, originalConfig_.runtime.endPageId);
        pipeline_.setMaxPages(originalConfig_.runtime.maxPages);
    }

private:
    DocPipeline& pipeline_;
    PipelineConfig originalConfig_;
};

struct ProcessedDocument {
    std::string filename;
    fs::path parseDir;
    fs::path imagesDir;
    fs::path layoutDir;
    fs::path markdownPath;
    fs::path contentListPath;
    fs::path middleJsonPath;
    fs::path modelJsonPath;
    json contentList = json::array();
    json middleJson = json::object();
    json modelJson = json::array();
    DocumentResult result;
    std::vector<std::string> warnings;
};

ProcessedDocument processDocumentBytes(
    DocPipeline& pipeline,
    const std::string& bytes,
    const std::string& filename,
    const FileParseOptions& options) {
    if (options.backend != "pipeline") {
        throw std::runtime_error("Unsupported backend: " + options.backend);
    }

    const std::string cleanName = safeFilename(filename);
    const std::string stem = safeStem(cleanName);
    const std::string extension = toLower(fs::path(cleanName).extension().string());

    ProcessedDocument processed;
    processed.filename = cleanName;
    processed.parseDir = fs::absolute(fs::path(options.outputDir) / stem / options.parseMethod);
    processed.imagesDir = processed.parseDir / "images";
    processed.layoutDir = processed.parseDir / "layout";
    processed.markdownPath = processed.parseDir / (stem + ".md");
    processed.contentListPath = processed.parseDir / (stem + "_content_list.json");
    processed.middleJsonPath = processed.parseDir / (stem + "_middle.json");
    processed.modelJsonPath = processed.parseDir / (stem + "_model.json");
    processed.warnings = collectRequestWarnings(options);

    fs::create_directories(processed.parseDir);
    writeBinaryFile(processed.parseDir / (stem + "_origin" + extension), bytes);

    PipelineStateGuard guard(pipeline);

    PipelineStages stages = pipeline.config().stages;
    stages.enableFormula = options.formulaEnable;
    stages.enableWiredTable = options.tableEnable;
    pipeline.setStageOptions(stages);
    pipeline.setOutputDir(processed.parseDir.string());
    pipeline.setSaveImages(true);
    pipeline.setSaveVisualization(true);
    pipeline.setPageRange(options.startPageId, options.endPageId);

    if (isPdfExtension(extension)) {
        processed.result = pipeline.processPdfFromMemory(
            reinterpret_cast<const uint8_t*>(bytes.data()), bytes.size());
    } else if (isImageExtension(extension)) {
        std::vector<uint8_t> buffer(bytes.begin(), bytes.end());
        cv::Mat image = cv::imdecode(buffer, cv::IMREAD_COLOR);
        if (image.empty()) {
            throw std::runtime_error("Failed to decode image: " + cleanName);
        }
        processed.result = pipeline.processImageDocument(image, 0);
    } else {
        throw std::runtime_error("Unsupported file type: " + extension);
    }

    writeTextFile(processed.markdownPath, processed.result.markdown);
    writeTextFile(processed.contentListPath, processed.result.contentListJson);

    processed.contentList = buildContentListJson(processed.result);
    processed.middleJson = buildMiddleJson(processed.result);
    processed.modelJson = buildModelJson(processed.result);

    writeTextFile(processed.middleJsonPath, processed.middleJson.dump(2));
    writeTextFile(processed.modelJsonPath, processed.modelJson.dump(2));

    return processed;
}

json buildFileResult(const ProcessedDocument& processed, const FileParseOptions& options) {
    json result{
        {"filename", processed.filename},
        {"backend", options.backend},
        {"deepx", true},
        {"engines", makeEngineJson(options.tableEnable, options.formulaEnable)},
        {"stats", makeStatsJson(processed.result)},
        {"output_dir", processed.parseDir.string()},
        {"markdown_path", processed.markdownPath.string()},
        {"content_list_path", processed.contentListPath.string()},
        {"middle_json_path", processed.middleJsonPath.string()},
        {"model_output_path", processed.modelJsonPath.string()},
        {"layout_files", collectAbsoluteFiles(processed.layoutDir)},
    };

    if (options.returnMd) {
        result["md_content"] = processed.result.markdown;
    }
    if (options.returnMiddleJson) {
        result["middle_json"] = processed.middleJson;
    }
    if (options.returnModelOutput) {
        result["model_output"] = processed.modelJson;
    }
    if (options.returnContentList) {
        result["content_list"] = processed.contentList;
    }
    if (options.returnImages) {
        result["images"] = collectImagesAsDataUrls(processed.imagesDir);
    }
    if (!processed.warnings.empty()) {
        result["warnings"] = processed.warnings;
    }

    return result;
}

json buildHealthJson() {
    json envVars{
        {"CUSTOM_INTER_OP_THREADS_COUNT", std::getenv("CUSTOM_INTER_OP_THREADS_COUNT")},
        {"CUSTOM_INTRA_OP_THREADS_COUNT", std::getenv("CUSTOM_INTRA_OP_THREADS_COUNT")},
        {"DXRT_DYNAMIC_CPU_THREAD", std::getenv("DXRT_DYNAMIC_CPU_THREAD")},
        {"DXRT_TASK_MAX_LOAD", std::getenv("DXRT_TASK_MAX_LOAD")},
        {"NFH_INPUT_WORKER_THREADS", std::getenv("NFH_INPUT_WORKER_THREADS")},
        {"NFH_OUTPUT_WORKER_THREADS", std::getenv("NFH_OUTPUT_WORKER_THREADS")},
    };

    return json{
        {"status", "healthy"},
        {"version", "0.1.0-cpp"},
        {"api", "RapidDoc Offline API (C++/Crow)"},
        {"mode", "closed_environment"},
        {"default_engines", makeEngineJson(true, true)},
        {"environment_variables", envVars},
    };
}

FileParseOptions parseFileParseOptions(const crow::multipart::message& msg) {
    FileParseOptions options;
    options.outputDir = getMultipartField(msg, "output_dir", options.outputDir);
    options.clearOutputFile = parseBool(getMultipartField(msg, "clear_output_file"), false);
    const auto langList = getMultipartFieldValues(msg, "lang_list");
    if (!langList.empty()) {
        options.langList = langList;
    }
    options.backend = getMultipartField(msg, "backend", options.backend);
    options.parseMethod = canonicalParseMethod(
        getMultipartField(msg, "parse_method", options.parseMethod));
    options.formulaEnable = parseBool(
        getMultipartField(msg, "formula_enable"), options.formulaEnable);
    options.tableEnable = parseBool(
        getMultipartField(msg, "table_enable"), options.tableEnable);
    options.deepxRequested = parseBool(getMultipartField(msg, "deepx"), true);
    options.layoutEngine = getMultipartField(msg, "layout_engine", options.layoutEngine);
    options.ocrEngine = getMultipartField(msg, "ocr_engine", options.ocrEngine);
    options.formulaEngine = getMultipartField(msg, "formula_engine", options.formulaEngine);
    options.tableEngine = getMultipartField(msg, "table_engine", options.tableEngine);
    options.returnMd = parseBool(getMultipartField(msg, "return_md"), true);
    options.returnMiddleJson = parseBool(getMultipartField(msg, "return_middle_json"), false);
    options.returnModelOutput = parseBool(getMultipartField(msg, "return_model_output"), false);
    options.returnContentList = parseBool(getMultipartField(msg, "return_content_list"), false);
    options.returnImages = parseBool(getMultipartField(msg, "return_images"), false);
    options.startPageId = parseInt(getMultipartField(msg, "start_page_id"), 0);
    options.endPageId = parseInt(getMultipartField(msg, "end_page_id"), 99999);
    return options;
}

} // namespace

DocServer::DocServer(const ServerConfig& config)
    : config_(config)
{
    fs::create_directories(config_.uploadDir);

    pipeline_ = std::make_unique<DocPipeline>(config_.pipelineConfig);
    if (!pipeline_->initialize()) {
        throw std::runtime_error("Failed to initialize document pipeline");
    }
}

DocServer::~DocServer() {
    stop();
}

void DocServer::run() {
    LOG_INFO("Starting RapidDoc HTTP server on {}:{}", config_.host, config_.port);

    crow::SimpleApp app;

    CROW_ROUTE(app, "/health")
    ([this]() {
        crow::response resp(200, buildHealthJson().dump());
        resp.set_header("Content-Type", "application/json");
        return resp;
    });

    CROW_ROUTE(app, "/status")
    ([this]() {
        crow::response resp(200, buildStatusJson());
        resp.set_header("Content-Type", "application/json");
        return resp;
    });

    CROW_ROUTE(app, "/process").methods("POST"_method)
    ([this](const crow::request& req) {
        requestCount_++;

        try {
            const auto contentType = req.get_header_value("Content-Type");
            if (contentType.find("multipart/form-data") == std::string::npos) {
                errorCount_++;
                return crow::response(400, R"({"error":"Expected multipart/form-data"})");
            }

            crow::multipart::message msg(req);
            auto parts = getMultipartParts(msg, "file");
            if (parts.empty()) {
                parts = getMultipartParts(msg, "files");
            }
            if (parts.empty()) {
                errorCount_++;
                return crow::response(400, R"({"error":"No file field found"})");
            }

            const auto& filePart = parts.front();
            std::string filename = "upload.pdf";
            const auto disposition = filePart.get_header_object("Content-Disposition");
            const auto filenameIt = disposition.params.find("filename");
            if (filenameIt != disposition.params.end()) {
                filename = filenameIt->second;
            }

            FileParseOptions options;
            options.outputDir = fs::path(config_.uploadDir) / "legacy";
            options.returnMd = true;
            options.returnContentList = true;
            options.clearOutputFile = true;

            json legacyResponse;
            {
                std::scoped_lock<std::mutex> lock(pipelineMutex_);
                const ProcessedDocument processed = processDocumentBytes(
                    *pipeline_, filePart.body, filename, options);
                legacyResponse = json{
                    {"pages", processed.result.processedPages},
                    {"total_pages", processed.result.totalPages},
                    {"skipped", processed.result.skippedElements},
                    {"time_ms", processed.result.totalTimeMs},
                    {"stats", makeStatsJson(processed.result)},
                    {"markdown", processed.result.markdown},
                    {"content_list", processed.contentList},
                    {"output_dir", processed.parseDir.string()},
                };
                fs::remove_all(processed.parseDir.parent_path());
            }

            successCount_++;
            crow::response resp(200, legacyResponse.dump());
            resp.set_header("Content-Type", "application/json");
            return resp;
        }
        catch (const std::exception& e) {
            errorCount_++;
            LOG_ERROR("Processing error: {}", e.what());
            return crow::response(500, json{{"error", e.what()}}.dump());
        }
    });

    CROW_ROUTE(app, "/process/base64").methods("POST"_method)
    ([this](const crow::request& req) {
        requestCount_++;

        try {
            json requestBody = json::parse(req.body);
            if (!requestBody.contains("data") || !requestBody["data"].is_string()) {
                errorCount_++;
                return crow::response(400, R"({"error":"Missing 'data' field"})");
            }

            const auto decoded = base64Decode(requestBody["data"].get<std::string>());
            if (decoded.empty()) {
                errorCount_++;
                return crow::response(400, R"({"error":"Invalid base64 data"})");
            }

            const std::string filename = requestBody.value("filename", "upload.pdf");
            FileParseOptions options;
            options.outputDir = fs::path(config_.uploadDir) / "legacy";
            options.returnMd = true;
            options.returnContentList = true;
            options.clearOutputFile = true;

            json legacyResponse;
            {
                std::scoped_lock<std::mutex> lock(pipelineMutex_);
                const ProcessedDocument processed = processDocumentBytes(
                    *pipeline_,
                    std::string(reinterpret_cast<const char*>(decoded.data()), decoded.size()),
                    filename,
                    options);
                legacyResponse = json{
                    {"pages", processed.result.processedPages},
                    {"total_pages", processed.result.totalPages},
                    {"skipped", processed.result.skippedElements},
                    {"time_ms", processed.result.totalTimeMs},
                    {"stats", makeStatsJson(processed.result)},
                    {"markdown", processed.result.markdown},
                    {"content_list", processed.contentList},
                    {"output_dir", processed.parseDir.string()},
                };
                fs::remove_all(processed.parseDir.parent_path());
            }

            successCount_++;
            crow::response resp(200, legacyResponse.dump());
            resp.set_header("Content-Type", "application/json");
            return resp;
        }
        catch (const std::exception& e) {
            errorCount_++;
            LOG_ERROR("Base64 processing error: {}", e.what());
            return crow::response(500, json{{"error", e.what()}}.dump());
        }
    });

    CROW_ROUTE(app, "/file_parse").methods("POST"_method)
    ([this](const crow::request& req) {
        requestCount_++;

        try {
            const auto contentType = req.get_header_value("Content-Type");
            if (contentType.find("multipart/form-data") == std::string::npos) {
                errorCount_++;
                return crow::response(400, R"({"error":"Expected multipart/form-data"})");
            }

            crow::multipart::message msg(req);
            auto fileParts = getMultipartParts(msg, "files");
            if (fileParts.empty()) {
                fileParts = getMultipartParts(msg, "file");
            }
            if (fileParts.empty()) {
                errorCount_++;
                return crow::response(400, R"({"error":"No files provided"})");
            }

            FileParseOptions options = parseFileParseOptions(msg);
            json results = json::array();
            int successFiles = 0;
            const auto requestWarnings = collectRequestWarnings(options);

            std::scoped_lock<std::mutex> lock(pipelineMutex_);
            for (const auto& part : fileParts) {
                std::string filename = "upload.bin";
                const auto disposition = part.get_header_object("Content-Disposition");
                const auto filenameIt = disposition.params.find("filename");
                if (filenameIt != disposition.params.end()) {
                    filename = filenameIt->second;
                }

                try {
                    ProcessedDocument processed = processDocumentBytes(
                        *pipeline_, part.body, filename, options);
                    json fileResult = buildFileResult(processed, options);
                    if (!requestWarnings.empty()) {
                        fileResult["request_warnings"] = requestWarnings;
                    }
                    results.push_back(std::move(fileResult));
                    successFiles++;

                    if (options.clearOutputFile) {
                        fs::remove_all(processed.parseDir.parent_path());
                    }
                }
                catch (const std::exception& e) {
                    results.push_back(json{
                        {"filename", safeFilename(filename)},
                        {"error", e.what()},
                    });
                }
            }

            json responseData{
                {"results", std::move(results)},
                {"total_files", static_cast<int>(fileParts.size())},
                {"successful_files", successFiles},
                {"mode", "closed_environment"},
                {"deepx", true},
                {"engines_used", makeEngineJson(options.tableEnable, options.formulaEnable)},
            };
            if (!requestWarnings.empty()) {
                responseData["warnings"] = requestWarnings;
            }

            successCount_++;
            crow::response resp(200, responseData.dump());
            resp.set_header("Content-Type", "application/json");
            return resp;
        }
        catch (const std::exception& e) {
            errorCount_++;
            LOG_ERROR("file_parse error: {}", e.what());
            return crow::response(500, json{{"error", e.what()}}.dump());
        }
    });

    CROW_ROUTE(app, "/v1/images:annotate").methods("POST"_method)
    ([this](const crow::request& req) {
        requestCount_++;

        try {
            json requestBody = json::parse(req.body);
            if (!requestBody.contains("requests") || !requestBody["requests"].is_array()) {
                errorCount_++;
                return crow::response(400, R"({"error":{"code":400,"message":"Missing requests array","status":"INVALID_ARGUMENT"}})");
            }

            const bool globalDeepx = requestBody.value("deepx", true);
            json responses = json::array();

            std::scoped_lock<std::mutex> lock(pipelineMutex_);
            size_t index = 0;
            for (const auto& requestItem : requestBody["requests"]) {
                try {
                    std::string imageBytes;
                    std::string imageName = "image_" + std::to_string(index++) + ".png";

                    if (requestItem.contains("image") &&
                        requestItem["image"].contains("content") &&
                        requestItem["image"]["content"].is_string()) {
                        const auto decoded = base64Decode(requestItem["image"]["content"].get<std::string>());
                        imageBytes.assign(reinterpret_cast<const char*>(decoded.data()), decoded.size());
                    } else if (requestItem.contains("image") &&
                               requestItem["image"].contains("source") &&
                               requestItem["image"]["source"].contains("imageUri") &&
                               requestItem["image"]["source"]["imageUri"].is_string()) {
                        const std::string uri = requestItem["image"]["source"]["imageUri"].get<std::string>();
                        if (uri.rfind("http://", 0) == 0 || uri.rfind("https://", 0) == 0) {
                            responses.push_back(json{
                                {"error", {
                                    {"code", 403},
                                    {"message", "Network access blocked in closed environment. Please use base64 content or a local path."},
                                    {"status", "PERMISSION_DENIED"},
                                }},
                            });
                            continue;
                        }
                        if (!fs::exists(uri)) {
                            responses.push_back(json{
                                {"error", {
                                    {"code", 400},
                                    {"message", "Invalid imageUri: " + uri},
                                    {"status", "INVALID_ARGUMENT"},
                                }},
                            });
                            continue;
                        }
                        imageBytes = readBinaryFile(uri);
                        imageName = fs::path(uri).filename().string();
                    } else {
                        responses.push_back(json{
                            {"error", {
                                {"code", 400},
                                {"message", "Either image.content or image.source.imageUri must be provided"},
                                {"status", "INVALID_ARGUMENT"},
                            }},
                        });
                        continue;
                    }

                    bool documentTextDetection = false;
                    bool textDetection = false;
                    if (requestItem.contains("features") && requestItem["features"].is_array()) {
                        for (const auto& feature : requestItem["features"]) {
                            const std::string type = feature.value("type", "");
                            documentTextDetection = documentTextDetection || (type == "DOCUMENT_TEXT_DETECTION");
                            textDetection = textDetection || (type == "TEXT_DETECTION");
                        }
                    }

                    if (!documentTextDetection && !textDetection) {
                        responses.push_back(json{
                            {"error", {
                                {"code", 400},
                                {"message", "Unsupported feature types. Supported: TEXT_DETECTION, DOCUMENT_TEXT_DETECTION"},
                                {"status", "INVALID_ARGUMENT"},
                            }},
                        });
                        continue;
                    }

                    FileParseOptions options;
                    options.outputDir = fs::path(config_.uploadDir) / "annotate";
                    options.parseMethod = "auto";
                    options.formulaEnable = documentTextDetection;
                    options.tableEnable = documentTextDetection;
                    options.returnMd = true;
                    options.returnMiddleJson = documentTextDetection;
                    options.returnContentList = documentTextDetection;
                    options.returnModelOutput = false;
                    options.returnImages = false;
                    options.clearOutputFile = true;
                    options.deepxRequested = requestItem.value("deepx", globalDeepx);

                    const ProcessedDocument processed = processDocumentBytes(
                        *pipeline_, imageBytes, imageName, options);

                    json responseItem;
                    if (textDetection) {
                        responseItem["textAnnotations"] = json::array({
                            {
                                {"description", processed.result.markdown},
                                {"locale", "auto"},
                            }
                        });
                    }
                    responseItem["fullTextAnnotation"] = json{
                        {"text", processed.result.markdown},
                    };

                    if (documentTextDetection) {
                        responseItem["middleJson"] = processed.middleJson;
                        responseItem["contentList"] = processed.contentList;
                    }

                    responses.push_back(std::move(responseItem));
                    fs::remove_all(processed.parseDir.parent_path());
                }
                catch (const std::exception& e) {
                    responses.push_back(json{
                        {"error", {
                            {"code", 500},
                            {"message", e.what()},
                            {"status", "INTERNAL"},
                        }},
                    });
                }
            }

            successCount_++;
            crow::response resp(200, json{{"responses", std::move(responses)}}.dump());
            resp.set_header("Content-Type", "application/json");
            return resp;
        }
        catch (const std::exception& e) {
            errorCount_++;
            LOG_ERROR("images:annotate error: {}", e.what());
            return crow::response(
                500,
                json{{"error", {{"code", 500}, {"message", e.what()}, {"status", "INTERNAL"}}}}.dump());
        }
    });

    running_ = true;
    app.bindaddr(config_.host)
       .port(config_.port)
       .concurrency(config_.numWorkers)
       .run();
    running_ = false;
}

void DocServer::stop() {
    running_ = false;
    LOG_INFO("RapidDoc HTTP server stopped");
}

std::string DocServer::handleProcess(const std::string& pdfData, const std::string& filename) {
    FileParseOptions options;
    options.outputDir = fs::path(config_.uploadDir) / "legacy";
    options.returnMd = true;
    options.returnContentList = true;
    options.clearOutputFile = true;

    std::scoped_lock<std::mutex> lock(pipelineMutex_);
    const ProcessedDocument processed = processDocumentBytes(*pipeline_, pdfData, filename, options);

    json response{
        {"pages", processed.result.processedPages},
        {"total_pages", processed.result.totalPages},
        {"skipped", processed.result.skippedElements},
        {"time_ms", processed.result.totalTimeMs},
        {"stats", makeStatsJson(processed.result)},
        {"markdown", processed.result.markdown},
        {"content_list", processed.contentList},
        {"output_dir", processed.parseDir.string()},
    };

    fs::remove_all(processed.parseDir.parent_path());
    return response.dump();
}

std::string DocServer::buildStatusJson() {
    json status{
        {"status", running_.load() ? "running" : "stopped"},
        {"requests", requestCount_.load()},
        {"success", successCount_.load()},
        {"errors", errorCount_.load()},
        {"engines", makeEngineJson(true, true)},
        {"capabilities", {
            {"layout", true},
            {"ocr", true},
            {"wired_table", true},
            {"wireless_table", false},
            {"formula_latex", false},
            {"formula_image_fallback", true},
            {"gradio_ui", true},
            {"file_parse_api", true},
            {"vision_annotate_api", true},
        }},
    };
    return status.dump();
}

} // namespace rapid_doc
