#include "output/detail_report.h"

#include "common/perf_utils.h"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace rapid_doc {

namespace {

const char* typeToString(ContentElement::Type type) {
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

std::string onOff(bool value) {
    return value ? "ON" : "OFF";
}

std::string formatMs(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << value << " ms";
    return out.str();
}

std::string formatBBox(const LayoutBox& box) {
    std::ostringstream out;
    out << "("
        << static_cast<int>(box.x0) << ", "
        << static_cast<int>(box.y0) << ", "
        << static_cast<int>(box.x1) << ", "
        << static_cast<int>(box.y1) << ")";
    return out.str();
}

std::string previewText(const std::string& text) {
    if (text.empty()) {
        return {};
    }

    std::string preview = text;
    std::replace(preview.begin(), preview.end(), '\n', ' ');
    const size_t maxLen = 96;
    if (preview.size() > maxLen) {
        preview.resize(maxLen - 3);
        preview += "...";
    }
    return preview;
}

bool isUnsupportedElement(const ContentElement& elem) {
    return elem.type == ContentElement::Type::UNKNOWN ||
           elem.text.find("[Unsupported layout category:") != std::string::npos;
}

bool isFallbackElement(const ContentElement& elem) {
    return elem.skipped &&
           ((elem.type == ContentElement::Type::TABLE && !elem.text.empty()) ||
            (elem.type == ContentElement::Type::EQUATION) ||
            (elem.type == ContentElement::Type::IMAGE));
}

std::string describeArtifact(const std::string& label, const std::string& value) {
    std::ostringstream out;
    out << "  " << label << ": " << (value.empty() ? "(not generated)" : value) << "\n";
    return out.str();
}

void appendStageLine(
    std::ostringstream& out,
    const std::string& label,
    double value)
{
    out << "  " << label << ": " << formatMs(value) << "\n";
}

} // namespace

std::string buildDetailReport(
    const DocumentResult& result,
    const DetailReportOptions& options)
{
    std::ostringstream out;

    out << "RapidDoc Detail Report\n";
    out << "=====================\n";
    out << "Input: " << (options.inputPath.empty() ? "(unknown)" : options.inputPath) << "\n";
    out << "Stages: "
        << "pdf_render=" << onOff(options.stageConfig.enablePdfRender)
        << ", layout=" << onOff(options.stageConfig.enableLayout)
        << ", ocr=" << onOff(options.stageConfig.enableOcr)
        << ", wired_table=" << onOff(options.stageConfig.enableWiredTable)
        << ", formula_fallback=" << onOff(options.stageConfig.enableFormula)
        << ", reading_order=" << onOff(options.stageConfig.enableReadingOrder)
        << ", markdown=" << onOff(options.stageConfig.enableMarkdownOutput)
        << "\n";
    out << "Artifacts\n";
    out << describeArtifact("output_dir", options.artifacts.outputDir);
    out << describeArtifact("markdown_path", options.artifacts.markdownPath);
    out << describeArtifact("content_list_path", options.artifacts.contentListPath);
    out << describeArtifact("layout_dir", options.saveVisualization ? options.artifacts.layoutDir : "");
    out << describeArtifact("images_dir", options.saveImages ? options.artifacts.imagesDir : "");

    out << "Document Summary\n";
    out << "  pages: " << result.processedPages << "/" << result.totalPages << "\n";
    out << "  skipped_elements: " << result.skippedElements << "\n";
    out << "  total_time: " << formatMs(result.totalTimeMs) << "\n";
    out << "  pipeline_mode: " << result.stats.pipelineMode << "\n";
    out << "  overlap_factor: " << std::fixed << std::setprecision(2)
        << result.stats.pipelineOverlapFactor << "\n";
    out << "  tracked_stage_time: " << formatMs(totalTrackedStageTimeMs(result.stats)) << "\n";
    appendStageLine(out, "pdf_render", result.stats.pdfRenderTimeMs);
    appendStageLine(out, "layout", result.stats.layoutTimeMs);
    appendStageLine(out, "ocr", result.stats.ocrTimeMs);
    appendStageLine(out, "table", result.stats.tableTimeMs);
    appendStageLine(out, "figure", result.stats.figureTimeMs);
    appendStageLine(out, "formula", result.stats.formulaTimeMs);
    appendStageLine(out, "unsupported", result.stats.unsupportedTimeMs);
    appendStageLine(out, "reading_order", result.stats.readingOrderTimeMs);
    appendStageLine(out, "output_gen", result.stats.outputGenTimeMs);
    appendStageLine(out, "npu_serial", result.stats.npuSerialTimeMs);
    appendStageLine(out, "cpu_only", result.stats.cpuOnlyTimeMs);
    appendStageLine(out, "npu_lock_wait", result.stats.npuLockWaitTimeMs);
    appendStageLine(out, "npu_lock_hold", result.stats.npuLockHoldTimeMs);
    appendStageLine(out, "npu_service", result.stats.npuServiceTimeMs);
    appendStageLine(out, "npu_slot_wait", result.stats.npuSlotWaitTimeMs);
    appendStageLine(out, "layout_npu_service", result.stats.layoutNpuServiceTimeMs);
    appendStageLine(out, "layout_npu_slot_wait", result.stats.layoutNpuSlotWaitTimeMs);
    appendStageLine(out, "ocr_outer_slot_hold", result.stats.ocrOuterSlotHoldTimeMs);
    appendStageLine(out, "ocr_submodule_window", result.stats.ocrSubmoduleWindowTimeMs);
    appendStageLine(out, "ocr_slot_wait", result.stats.ocrSlotWaitTimeMs);
    appendStageLine(out, "ocr_collect_wait", result.stats.ocrCollectWaitTimeMs);
    appendStageLine(out, "ocr_inflight_peak", result.stats.ocrInflightPeak);
    appendStageLine(out, "ocr_buffered_out_of_order", result.stats.ocrBufferedOutOfOrderCount);
    appendStageLine(out, "table_npu_service", result.stats.tableNpuServiceTimeMs);
    appendStageLine(out, "table_npu_slot_wait", result.stats.tableNpuSlotWaitTimeMs);
    appendStageLine(out, "table_ocr_service", result.stats.tableOcrServiceTimeMs);
    appendStageLine(out, "table_ocr_slot_wait", result.stats.tableOcrSlotWaitTimeMs);
    appendStageLine(out, "cpu_pre", result.stats.cpuPreTimeMs);
    appendStageLine(out, "cpu_post", result.stats.cpuPostTimeMs);
    appendStageLine(out, "finalize_cpu", result.stats.finalizeCpuTimeMs);
    appendStageLine(out, "table_finalize", result.stats.tableFinalizeTimeMs);
    appendStageLine(out, "ocr_collect_or_merge", result.stats.ocrCollectOrMergeTimeMs);
    appendStageLine(out, "layout_queue_wait", result.stats.layoutQueueWaitTimeMs);
    appendStageLine(out, "plan_queue_wait", result.stats.planQueueWaitTimeMs);
    appendStageLine(out, "ocr_table_queue_wait", result.stats.ocrTableQueueWaitTimeMs);
    appendStageLine(out, "finalize_queue_wait", result.stats.finalizeQueueWaitTimeMs);
    appendStageLine(out, "render_queue_push_block", result.stats.renderQueuePushBlockTimeMs);
    appendStageLine(out, "layout_queue_push_block", result.stats.layoutQueuePushBlockTimeMs);
    appendStageLine(out, "plan_queue_push_block", result.stats.planQueuePushBlockTimeMs);
    appendStageLine(out, "ocr_table_queue_push_block", result.stats.ocrTableQueuePushBlockTimeMs);
    appendStageLine(out, "queue_backpressure", result.stats.queueBackpressureTimeMs);

    out << "Per-page\n";
    for (const auto& page : result.pages) {
        size_t skippedCount = 0;
        size_t fallbackCount = 0;
        size_t unsupportedCount = 0;
        for (const auto& elem : page.elements) {
            if (elem.skipped) {
                ++skippedCount;
            }
            if (isFallbackElement(elem)) {
                ++fallbackCount;
            }
            if (isUnsupportedElement(elem)) {
                ++unsupportedCount;
            }
        }

        out << "  Page " << page.pageIndex
            << " | size=" << page.pageWidth << "x" << page.pageHeight
            << " | total=" << formatMs(page.totalTimeMs)
            << " | tracked=" << formatMs(totalTrackedStageTimeMs(page.stats))
            << " | overlap=" << std::fixed << std::setprecision(2) << page.stats.pipelineOverlapFactor
            << " | elements=" << page.elements.size()
            << " | skipped=" << skippedCount
            << " | fallback=" << fallbackCount
            << " | unsupported=" << unsupportedCount
            << "\n";
        appendStageLine(out, "    layout", page.stats.layoutTimeMs);
        appendStageLine(out, "    ocr", page.stats.ocrTimeMs);
        appendStageLine(out, "    table", page.stats.tableTimeMs);
        appendStageLine(out, "    figure", page.stats.figureTimeMs);
        appendStageLine(out, "    formula", page.stats.formulaTimeMs);
        appendStageLine(out, "    unsupported", page.stats.unsupportedTimeMs);
        appendStageLine(out, "    reading_order", page.stats.readingOrderTimeMs);
        appendStageLine(out, "    npu_serial", page.stats.npuSerialTimeMs);
        appendStageLine(out, "    cpu_only", page.stats.cpuOnlyTimeMs);
        appendStageLine(out, "    npu_lock_wait", page.stats.npuLockWaitTimeMs);
        appendStageLine(out, "    npu_lock_hold", page.stats.npuLockHoldTimeMs);
        appendStageLine(out, "    npu_service", page.stats.npuServiceTimeMs);
        appendStageLine(out, "    npu_slot_wait", page.stats.npuSlotWaitTimeMs);
        appendStageLine(out, "    layout_npu_service", page.stats.layoutNpuServiceTimeMs);
        appendStageLine(out, "    layout_npu_slot_wait", page.stats.layoutNpuSlotWaitTimeMs);
        appendStageLine(out, "    ocr_outer_slot_hold", page.stats.ocrOuterSlotHoldTimeMs);
        appendStageLine(out, "    ocr_submodule_window", page.stats.ocrSubmoduleWindowTimeMs);
        appendStageLine(out, "    ocr_slot_wait", page.stats.ocrSlotWaitTimeMs);
        appendStageLine(out, "    ocr_collect_wait", page.stats.ocrCollectWaitTimeMs);
        appendStageLine(out, "    ocr_inflight_peak", page.stats.ocrInflightPeak);
        appendStageLine(out, "    ocr_buffered_out_of_order", page.stats.ocrBufferedOutOfOrderCount);
        appendStageLine(out, "    table_npu_service", page.stats.tableNpuServiceTimeMs);
        appendStageLine(out, "    table_npu_slot_wait", page.stats.tableNpuSlotWaitTimeMs);
        appendStageLine(out, "    table_ocr_service", page.stats.tableOcrServiceTimeMs);
        appendStageLine(out, "    table_ocr_slot_wait", page.stats.tableOcrSlotWaitTimeMs);
        appendStageLine(out, "    cpu_pre", page.stats.cpuPreTimeMs);
        appendStageLine(out, "    cpu_post", page.stats.cpuPostTimeMs);
        appendStageLine(out, "    finalize_cpu", page.stats.finalizeCpuTimeMs);
        appendStageLine(out, "    table_finalize", page.stats.tableFinalizeTimeMs);
        appendStageLine(out, "    ocr_collect_or_merge", page.stats.ocrCollectOrMergeTimeMs);
        appendStageLine(out, "    layout_queue_wait", page.stats.layoutQueueWaitTimeMs);
        appendStageLine(out, "    plan_queue_wait", page.stats.planQueueWaitTimeMs);
        appendStageLine(out, "    ocr_table_queue_wait", page.stats.ocrTableQueueWaitTimeMs);
        appendStageLine(out, "    finalize_queue_wait", page.stats.finalizeQueueWaitTimeMs);
        appendStageLine(out, "    render_queue_push_block", page.stats.renderQueuePushBlockTimeMs);
        appendStageLine(out, "    layout_queue_push_block", page.stats.layoutQueuePushBlockTimeMs);
        appendStageLine(out, "    plan_queue_push_block", page.stats.planQueuePushBlockTimeMs);
        appendStageLine(out, "    ocr_table_queue_push_block", page.stats.ocrTableQueuePushBlockTimeMs);
        appendStageLine(out, "    queue_backpressure", page.stats.queueBackpressureTimeMs);

        bool wroteFlaggedElement = false;
        for (const auto& elem : page.elements) {
            const bool flagged =
                elem.skipped || isFallbackElement(elem) || isUnsupportedElement(elem) ||
                !elem.imagePath.empty();
            if (!flagged) {
                continue;
            }

            wroteFlaggedElement = true;
            out << "    - " << typeToString(elem.type);
            if (elem.skipped) {
                out << " [skipped]";
            }
            if (isFallbackElement(elem)) {
                out << " [fallback]";
            }
            if (isUnsupportedElement(elem)) {
                out << " [unsupported]";
            }
            out << " bbox=" << formatBBox(elem.layoutBox);
            if (!elem.imagePath.empty()) {
                out << " image=" << elem.imagePath;
            }
            const std::string text = previewText(elem.text);
            if (!text.empty()) {
                out << " text=\"" << text << "\"";
            }
            out << "\n";
        }

        if (!wroteFlaggedElement) {
            out << "    - no skipped/fallback/unsupported elements\n";
        }
    }

    return out.str();
}

} // namespace rapid_doc
