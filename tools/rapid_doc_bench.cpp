/**
 * @file rapid_doc_bench.cpp
 * @brief Phase 2 performance baseline runner for RapidDocCpp.
 */

#include "common/config.h"
#include "common/perf_utils.h"
#include "pipeline/doc_pipeline.h"

#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>

#include <chrono>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace rapid_doc;

namespace {

struct BenchmarkCaseDefinition {
    std::string name;
    std::string description;
    std::string inputPath;
    std::function<void(PipelineConfig&)> configure;
};

struct BenchmarkIterationResult {
    double totalTimeMs = 0.0;
    double pipelineCallMs = 0.0;
    int processedPages = 0;
    DocumentStageStats stageStats;
    std::vector<double> pageTimesMs;
    std::string markdownSha256;
    std::string contentListSha256;
    std::string pipelineMode = "serial";
    std::string ocrOuterMode = "immediate_per_task";
};

struct BenchmarkCaseResult {
    BenchmarkCaseDefinition definition;
    int warmupIterations = 0;
    PipelineMode pipelineMode = PipelineMode::Serial;
    OcrOuterMode ocrOuterMode = OcrOuterMode::ImmediatePerTask;
    std::string status = "ok";
    std::string error;
    std::vector<BenchmarkIterationResult> iterations;
};

std::string formatMs(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << value << " ms";
    return out.str();
}

std::string formatCount(double value) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(2) << value;
    return out.str();
}

double pagesPerSecond(int processedPages, double totalTimeMs) {
    if (processedPages <= 0 || totalTimeMs <= 0.0) {
        return 0.0;
    }
    return static_cast<double>(processedPages) * 1000.0 / totalTimeMs;
}

namespace sha256 {

constexpr std::array<uint32_t, 64> kRoundConstants = {
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u,
};

constexpr std::array<uint32_t, 8> kInitialState = {
    0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
    0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u,
};

constexpr uint32_t rotr(uint32_t value, uint32_t count) {
    return (value >> count) | (value << (32 - count));
}

std::string digest(std::string_view input) {
    std::array<uint32_t, 8> state = kInitialState;
    std::vector<uint8_t> bytes(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        bytes[i] = static_cast<uint8_t>(input[i]);
    }

    const uint64_t bitLength = static_cast<uint64_t>(bytes.size()) * 8u;
    bytes.push_back(0x80u);
    while ((bytes.size() % 64u) != 56u) {
        bytes.push_back(0u);
    }
    for (int shift = 56; shift >= 0; shift -= 8) {
        bytes.push_back(static_cast<uint8_t>((bitLength >> shift) & 0xffu));
    }

    for (size_t offset = 0; offset < bytes.size(); offset += 64u) {
        std::array<uint32_t, 64> schedule{};
        for (size_t i = 0; i < 16u; ++i) {
            const size_t base = offset + (i * 4u);
            schedule[i] =
                (static_cast<uint32_t>(bytes[base]) << 24) |
                (static_cast<uint32_t>(bytes[base + 1]) << 16) |
                (static_cast<uint32_t>(bytes[base + 2]) << 8) |
                static_cast<uint32_t>(bytes[base + 3]);
        }
        for (size_t i = 16u; i < 64u; ++i) {
            const uint32_t s0 = rotr(schedule[i - 15u], 7u) ^
                                rotr(schedule[i - 15u], 18u) ^
                                (schedule[i - 15u] >> 3u);
            const uint32_t s1 = rotr(schedule[i - 2u], 17u) ^
                                rotr(schedule[i - 2u], 19u) ^
                                (schedule[i - 2u] >> 10u);
            schedule[i] = schedule[i - 16u] + s0 + schedule[i - 7u] + s1;
        }

        uint32_t a = state[0];
        uint32_t b = state[1];
        uint32_t c = state[2];
        uint32_t d = state[3];
        uint32_t e = state[4];
        uint32_t f = state[5];
        uint32_t g = state[6];
        uint32_t h = state[7];

        for (size_t i = 0; i < 64u; ++i) {
            const uint32_t s1 = rotr(e, 6u) ^ rotr(e, 11u) ^ rotr(e, 25u);
            const uint32_t ch = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = h + s1 + ch + kRoundConstants[i] + schedule[i];
            const uint32_t s0 = rotr(a, 2u) ^ rotr(a, 13u) ^ rotr(a, 22u);
            const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = s0 + maj;

            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }

        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }

    std::ostringstream out;
    out << std::hex << std::setfill('0');
    for (uint32_t word : state) {
        out << std::setw(8) << word;
    }
    return out.str();
}

} // namespace sha256

json summaryToJson(const PercentileSummary& summary) {
    return json{
        {"samples", summary.sampleCount},
        {"min_ms", summary.minMs},
        {"mean_ms", summary.meanMs},
        {"max_ms", summary.maxMs},
        {"p50_ms", summary.p50Ms},
        {"p95_ms", summary.p95Ms},
    };
}

json stageStatsToJson(const DocumentStageStats& stats) {
    return json{
        {"pdf_render_ms", stats.pdfRenderTimeMs},
        {"layout_ms", stats.layoutTimeMs},
        {"ocr_ms", stats.ocrTimeMs},
        {"table_ms", stats.tableTimeMs},
        {"figure_ms", stats.figureTimeMs},
        {"formula_ms", stats.formulaTimeMs},
        {"unsupported_ms", stats.unsupportedTimeMs},
        {"reading_order_ms", stats.readingOrderTimeMs},
        {"output_gen_ms", stats.outputGenTimeMs},
        {"npu_serial_ms", stats.npuSerialTimeMs},
        {"cpu_only_ms", stats.cpuOnlyTimeMs},
        {"npu_lock_wait_ms", stats.npuLockWaitTimeMs},
        {"npu_lock_hold_ms", stats.npuLockHoldTimeMs},
        {"npu_service_ms", stats.npuServiceTimeMs},
        {"npu_slot_wait_ms", stats.npuSlotWaitTimeMs},
        {"layout_npu_service_ms", stats.layoutNpuServiceTimeMs},
        {"layout_npu_slot_wait_ms", stats.layoutNpuSlotWaitTimeMs},
        {"ocr_outer_slot_hold_ms", stats.ocrOuterSlotHoldTimeMs},
        {"ocr_submodule_window_ms", stats.ocrSubmoduleWindowTimeMs},
        {"ocr_slot_wait_ms", stats.ocrSlotWaitTimeMs},
        {"ocr_collect_wait_ms", stats.ocrCollectWaitTimeMs},
        {"ocr_inflight_peak", stats.ocrInflightPeak},
        {"ocr_buffered_out_of_order_count", stats.ocrBufferedOutOfOrderCount},
        {"table_npu_service_ms", stats.tableNpuServiceTimeMs},
        {"table_npu_slot_wait_ms", stats.tableNpuSlotWaitTimeMs},
        {"table_ocr_service_ms", stats.tableOcrServiceTimeMs},
        {"table_ocr_slot_wait_ms", stats.tableOcrSlotWaitTimeMs},
        {"cpu_pre_ms", stats.cpuPreTimeMs},
        {"cpu_post_ms", stats.cpuPostTimeMs},
        {"finalize_cpu_ms", stats.finalizeCpuTimeMs},
        {"table_finalize_ms", stats.tableFinalizeTimeMs},
        {"ocr_collect_or_merge_ms", stats.ocrCollectOrMergeTimeMs},
        {"layout_queue_wait_ms", stats.layoutQueueWaitTimeMs},
        {"plan_queue_wait_ms", stats.planQueueWaitTimeMs},
        {"ocr_table_queue_wait_ms", stats.ocrTableQueueWaitTimeMs},
        {"finalize_queue_wait_ms", stats.finalizeQueueWaitTimeMs},
        {"render_queue_push_block_ms", stats.renderQueuePushBlockTimeMs},
        {"layout_queue_push_block_ms", stats.layoutQueuePushBlockTimeMs},
        {"plan_queue_push_block_ms", stats.planQueuePushBlockTimeMs},
        {"ocr_table_queue_push_block_ms", stats.ocrTableQueuePushBlockTimeMs},
        {"queue_backpressure_ms", stats.queueBackpressureTimeMs},
        {"pipeline_overlap_factor", stats.pipelineOverlapFactor},
        {"pipeline_mode", stats.pipelineMode},
        {"text_boxes_raw_count", stats.textBoxesRawCount},
        {"text_boxes_after_dedup_count", stats.textBoxesAfterDedupCount},
        {"table_boxes_raw_count", stats.tableBoxesRawCount},
        {"table_boxes_after_dedup_count", stats.tableBoxesAfterDedupCount},
        {"ocr_submit_count", stats.ocrSubmitCount},
        {"ocr_submit_area_sum", stats.ocrSubmitAreaSum},
        {"ocr_submit_area_mean", stats.ocrSubmitAreaMean},
        {"ocr_submit_area_p50", stats.ocrSubmitAreaP50},
        {"ocr_submit_area_p95", stats.ocrSubmitAreaP95},
        {"ocr_submit_small_count", stats.ocrSubmitSmallCount},
        {"ocr_submit_medium_count", stats.ocrSubmitMediumCount},
        {"ocr_submit_large_count", stats.ocrSubmitLargeCount},
        {"ocr_submit_text_count", stats.ocrSubmitTextCount},
        {"ocr_submit_title_count", stats.ocrSubmitTitleCount},
        {"ocr_submit_code_count", stats.ocrSubmitCodeCount},
        {"ocr_submit_list_count", stats.ocrSubmitListCount},
        {"ocr_dedup_skipped_count", stats.ocrDedupSkippedCount},
        {"table_npu_submit_count", stats.tableNpuSubmitCount},
        {"table_dedup_skipped_count", stats.tableDedupSkippedCount},
        {"ocr_timeout_count", stats.ocrTimeoutCount},
        {"ocr_buffered_result_hit_count", stats.ocrBufferedResultHitCount},
        {"tracked_total_ms", totalTrackedStageTimeMs(stats)},
    };
}

json benchmarkStatsToJson(const BenchmarkIterationResult& iteration) {
    return json{
        {"time_ms", iteration.totalTimeMs},
        {"pages_per_sec", pagesPerSecond(iteration.processedPages, iteration.totalTimeMs)},
        {"pdf_render_ms", iteration.stageStats.pdfRenderTimeMs},
        {"layout_ms", iteration.stageStats.layoutTimeMs},
        {"ocr_ms", iteration.stageStats.ocrTimeMs},
        {"table_ms", iteration.stageStats.tableTimeMs},
        {"reading_order_ms", iteration.stageStats.readingOrderTimeMs},
        {"output_gen_ms", iteration.stageStats.outputGenTimeMs},
        {"npu_serial_ms", iteration.stageStats.npuSerialTimeMs},
        {"cpu_only_ms", iteration.stageStats.cpuOnlyTimeMs},
        {"npu_lock_wait_ms", iteration.stageStats.npuLockWaitTimeMs},
        {"npu_lock_hold_ms", iteration.stageStats.npuLockHoldTimeMs},
        {"npu_service_ms", iteration.stageStats.npuServiceTimeMs},
        {"npu_slot_wait_ms", iteration.stageStats.npuSlotWaitTimeMs},
        {"layout_npu_service_ms", iteration.stageStats.layoutNpuServiceTimeMs},
        {"layout_npu_slot_wait_ms", iteration.stageStats.layoutNpuSlotWaitTimeMs},
        {"ocr_outer_slot_hold_ms", iteration.stageStats.ocrOuterSlotHoldTimeMs},
        {"ocr_submodule_window_ms", iteration.stageStats.ocrSubmoduleWindowTimeMs},
        {"ocr_slot_wait_ms", iteration.stageStats.ocrSlotWaitTimeMs},
        {"ocr_collect_wait_ms", iteration.stageStats.ocrCollectWaitTimeMs},
        {"ocr_inflight_peak", iteration.stageStats.ocrInflightPeak},
        {"ocr_buffered_out_of_order_count", iteration.stageStats.ocrBufferedOutOfOrderCount},
        {"table_npu_service_ms", iteration.stageStats.tableNpuServiceTimeMs},
        {"table_npu_slot_wait_ms", iteration.stageStats.tableNpuSlotWaitTimeMs},
        {"table_ocr_service_ms", iteration.stageStats.tableOcrServiceTimeMs},
        {"table_ocr_slot_wait_ms", iteration.stageStats.tableOcrSlotWaitTimeMs},
        {"cpu_pre_ms", iteration.stageStats.cpuPreTimeMs},
        {"cpu_post_ms", iteration.stageStats.cpuPostTimeMs},
        {"finalize_cpu_ms", iteration.stageStats.finalizeCpuTimeMs},
        {"table_finalize_ms", iteration.stageStats.tableFinalizeTimeMs},
        {"ocr_collect_or_merge_ms", iteration.stageStats.ocrCollectOrMergeTimeMs},
        {"layout_queue_wait_ms", iteration.stageStats.layoutQueueWaitTimeMs},
        {"plan_queue_wait_ms", iteration.stageStats.planQueueWaitTimeMs},
        {"ocr_table_queue_wait_ms", iteration.stageStats.ocrTableQueueWaitTimeMs},
        {"finalize_queue_wait_ms", iteration.stageStats.finalizeQueueWaitTimeMs},
        {"render_queue_push_block_ms", iteration.stageStats.renderQueuePushBlockTimeMs},
        {"layout_queue_push_block_ms", iteration.stageStats.layoutQueuePushBlockTimeMs},
        {"plan_queue_push_block_ms", iteration.stageStats.planQueuePushBlockTimeMs},
        {"ocr_table_queue_push_block_ms", iteration.stageStats.ocrTableQueuePushBlockTimeMs},
        {"queue_backpressure_ms", iteration.stageStats.queueBackpressureTimeMs},
        {"pipeline_overlap_factor", iteration.stageStats.pipelineOverlapFactor},
        {"pipeline_mode", iteration.pipelineMode},
        {"text_boxes_raw_count", iteration.stageStats.textBoxesRawCount},
        {"text_boxes_after_dedup_count", iteration.stageStats.textBoxesAfterDedupCount},
        {"table_boxes_raw_count", iteration.stageStats.tableBoxesRawCount},
        {"table_boxes_after_dedup_count", iteration.stageStats.tableBoxesAfterDedupCount},
        {"ocr_submit_count", iteration.stageStats.ocrSubmitCount},
        {"ocr_submit_area_sum", iteration.stageStats.ocrSubmitAreaSum},
        {"ocr_submit_area_mean", iteration.stageStats.ocrSubmitAreaMean},
        {"ocr_submit_area_p50", iteration.stageStats.ocrSubmitAreaP50},
        {"ocr_submit_area_p95", iteration.stageStats.ocrSubmitAreaP95},
        {"ocr_submit_small_count", iteration.stageStats.ocrSubmitSmallCount},
        {"ocr_submit_medium_count", iteration.stageStats.ocrSubmitMediumCount},
        {"ocr_submit_large_count", iteration.stageStats.ocrSubmitLargeCount},
        {"ocr_submit_text_count", iteration.stageStats.ocrSubmitTextCount},
        {"ocr_submit_title_count", iteration.stageStats.ocrSubmitTitleCount},
        {"ocr_submit_code_count", iteration.stageStats.ocrSubmitCodeCount},
        {"ocr_submit_list_count", iteration.stageStats.ocrSubmitListCount},
        {"ocr_dedup_skipped_count", iteration.stageStats.ocrDedupSkippedCount},
        {"table_npu_submit_count", iteration.stageStats.tableNpuSubmitCount},
        {"table_dedup_skipped_count", iteration.stageStats.tableDedupSkippedCount},
        {"ocr_timeout_count", iteration.stageStats.ocrTimeoutCount},
        {"ocr_buffered_result_hit_count", iteration.stageStats.ocrBufferedResultHitCount},
        {"pipeline_call_ms", iteration.pipelineCallMs},
    };
}

DocumentStageStats meanStageStats(const std::vector<BenchmarkIterationResult>& iterations) {
    DocumentStageStats mean;
    if (iterations.empty()) {
        return mean;
    }

    double ocrSubmitAreaP50Weighted = 0.0;
    double ocrSubmitAreaP95Weighted = 0.0;
    double ocrSubmitCountTotal = 0.0;

    for (const auto& iteration : iterations) {
        mean.pdfRenderTimeMs += iteration.stageStats.pdfRenderTimeMs;
        mean.layoutTimeMs += iteration.stageStats.layoutTimeMs;
        mean.ocrTimeMs += iteration.stageStats.ocrTimeMs;
        mean.tableTimeMs += iteration.stageStats.tableTimeMs;
        mean.figureTimeMs += iteration.stageStats.figureTimeMs;
        mean.formulaTimeMs += iteration.stageStats.formulaTimeMs;
        mean.unsupportedTimeMs += iteration.stageStats.unsupportedTimeMs;
        mean.readingOrderTimeMs += iteration.stageStats.readingOrderTimeMs;
        mean.outputGenTimeMs += iteration.stageStats.outputGenTimeMs;
        mean.npuSerialTimeMs += iteration.stageStats.npuSerialTimeMs;
        mean.cpuOnlyTimeMs += iteration.stageStats.cpuOnlyTimeMs;
        mean.npuLockWaitTimeMs += iteration.stageStats.npuLockWaitTimeMs;
        mean.npuLockHoldTimeMs += iteration.stageStats.npuLockHoldTimeMs;
        mean.npuServiceTimeMs += iteration.stageStats.npuServiceTimeMs;
        mean.npuSlotWaitTimeMs += iteration.stageStats.npuSlotWaitTimeMs;
        mean.layoutNpuServiceTimeMs += iteration.stageStats.layoutNpuServiceTimeMs;
        mean.layoutNpuSlotWaitTimeMs += iteration.stageStats.layoutNpuSlotWaitTimeMs;
        mean.ocrOuterSlotHoldTimeMs += iteration.stageStats.ocrOuterSlotHoldTimeMs;
        mean.ocrSubmoduleWindowTimeMs += iteration.stageStats.ocrSubmoduleWindowTimeMs;
        mean.ocrSlotWaitTimeMs += iteration.stageStats.ocrSlotWaitTimeMs;
        mean.ocrCollectWaitTimeMs += iteration.stageStats.ocrCollectWaitTimeMs;
        mean.ocrInflightPeak += iteration.stageStats.ocrInflightPeak;
        mean.ocrBufferedOutOfOrderCount += iteration.stageStats.ocrBufferedOutOfOrderCount;
        mean.tableNpuServiceTimeMs += iteration.stageStats.tableNpuServiceTimeMs;
        mean.tableNpuSlotWaitTimeMs += iteration.stageStats.tableNpuSlotWaitTimeMs;
        mean.tableOcrServiceTimeMs += iteration.stageStats.tableOcrServiceTimeMs;
        mean.tableOcrSlotWaitTimeMs += iteration.stageStats.tableOcrSlotWaitTimeMs;
        mean.cpuPreTimeMs += iteration.stageStats.cpuPreTimeMs;
        mean.cpuPostTimeMs += iteration.stageStats.cpuPostTimeMs;
        mean.finalizeCpuTimeMs += iteration.stageStats.finalizeCpuTimeMs;
        mean.tableFinalizeTimeMs += iteration.stageStats.tableFinalizeTimeMs;
        mean.ocrCollectOrMergeTimeMs += iteration.stageStats.ocrCollectOrMergeTimeMs;
        mean.layoutQueueWaitTimeMs += iteration.stageStats.layoutQueueWaitTimeMs;
        mean.planQueueWaitTimeMs += iteration.stageStats.planQueueWaitTimeMs;
        mean.ocrTableQueueWaitTimeMs += iteration.stageStats.ocrTableQueueWaitTimeMs;
        mean.finalizeQueueWaitTimeMs += iteration.stageStats.finalizeQueueWaitTimeMs;
        mean.renderQueuePushBlockTimeMs += iteration.stageStats.renderQueuePushBlockTimeMs;
        mean.layoutQueuePushBlockTimeMs += iteration.stageStats.layoutQueuePushBlockTimeMs;
        mean.planQueuePushBlockTimeMs += iteration.stageStats.planQueuePushBlockTimeMs;
        mean.ocrTableQueuePushBlockTimeMs += iteration.stageStats.ocrTableQueuePushBlockTimeMs;
        mean.queueBackpressureTimeMs += iteration.stageStats.queueBackpressureTimeMs;
        mean.pipelineOverlapFactor += iteration.stageStats.pipelineOverlapFactor;
        mean.textBoxesRawCount += iteration.stageStats.textBoxesRawCount;
        mean.textBoxesAfterDedupCount += iteration.stageStats.textBoxesAfterDedupCount;
        mean.tableBoxesRawCount += iteration.stageStats.tableBoxesRawCount;
        mean.tableBoxesAfterDedupCount += iteration.stageStats.tableBoxesAfterDedupCount;
        mean.ocrSubmitCount += iteration.stageStats.ocrSubmitCount;
        mean.ocrSubmitAreaSum += iteration.stageStats.ocrSubmitAreaSum;
        mean.ocrSubmitSmallCount += iteration.stageStats.ocrSubmitSmallCount;
        mean.ocrSubmitMediumCount += iteration.stageStats.ocrSubmitMediumCount;
        mean.ocrSubmitLargeCount += iteration.stageStats.ocrSubmitLargeCount;
        mean.ocrSubmitTextCount += iteration.stageStats.ocrSubmitTextCount;
        mean.ocrSubmitTitleCount += iteration.stageStats.ocrSubmitTitleCount;
        mean.ocrSubmitCodeCount += iteration.stageStats.ocrSubmitCodeCount;
        mean.ocrSubmitListCount += iteration.stageStats.ocrSubmitListCount;
        mean.ocrDedupSkippedCount += iteration.stageStats.ocrDedupSkippedCount;
        mean.tableNpuSubmitCount += iteration.stageStats.tableNpuSubmitCount;
        mean.tableDedupSkippedCount += iteration.stageStats.tableDedupSkippedCount;
        mean.ocrTimeoutCount += iteration.stageStats.ocrTimeoutCount;
        mean.ocrBufferedResultHitCount += iteration.stageStats.ocrBufferedResultHitCount;

        const double submitCount = std::max(0.0, iteration.stageStats.ocrSubmitCount);
        if (submitCount > 0.0) {
            ocrSubmitCountTotal += submitCount;
            ocrSubmitAreaP50Weighted += iteration.stageStats.ocrSubmitAreaP50 * submitCount;
            ocrSubmitAreaP95Weighted += iteration.stageStats.ocrSubmitAreaP95 * submitCount;
        }
    }

    const double denom = static_cast<double>(iterations.size());
    mean.pdfRenderTimeMs /= denom;
    mean.layoutTimeMs /= denom;
    mean.ocrTimeMs /= denom;
    mean.tableTimeMs /= denom;
    mean.figureTimeMs /= denom;
    mean.formulaTimeMs /= denom;
    mean.unsupportedTimeMs /= denom;
    mean.readingOrderTimeMs /= denom;
    mean.outputGenTimeMs /= denom;
    mean.npuSerialTimeMs /= denom;
    mean.cpuOnlyTimeMs /= denom;
    mean.npuLockWaitTimeMs /= denom;
    mean.npuLockHoldTimeMs /= denom;
    mean.npuServiceTimeMs /= denom;
    mean.npuSlotWaitTimeMs /= denom;
    mean.layoutNpuServiceTimeMs /= denom;
    mean.layoutNpuSlotWaitTimeMs /= denom;
    mean.ocrOuterSlotHoldTimeMs /= denom;
    mean.ocrSubmoduleWindowTimeMs /= denom;
    mean.ocrSlotWaitTimeMs /= denom;
    mean.ocrCollectWaitTimeMs /= denom;
    mean.ocrInflightPeak /= denom;
    mean.ocrBufferedOutOfOrderCount /= denom;
    mean.tableNpuServiceTimeMs /= denom;
    mean.tableNpuSlotWaitTimeMs /= denom;
    mean.tableOcrServiceTimeMs /= denom;
    mean.tableOcrSlotWaitTimeMs /= denom;
    mean.cpuPreTimeMs /= denom;
    mean.cpuPostTimeMs /= denom;
    mean.finalizeCpuTimeMs /= denom;
    mean.tableFinalizeTimeMs /= denom;
    mean.ocrCollectOrMergeTimeMs /= denom;
    mean.layoutQueueWaitTimeMs /= denom;
    mean.planQueueWaitTimeMs /= denom;
    mean.ocrTableQueueWaitTimeMs /= denom;
    mean.finalizeQueueWaitTimeMs /= denom;
    mean.renderQueuePushBlockTimeMs /= denom;
    mean.layoutQueuePushBlockTimeMs /= denom;
    mean.planQueuePushBlockTimeMs /= denom;
    mean.ocrTableQueuePushBlockTimeMs /= denom;
    mean.queueBackpressureTimeMs /= denom;
    mean.pipelineOverlapFactor /= denom;
    mean.textBoxesRawCount /= denom;
    mean.textBoxesAfterDedupCount /= denom;
    mean.tableBoxesRawCount /= denom;
    mean.tableBoxesAfterDedupCount /= denom;
    mean.ocrSubmitCount /= denom;
    mean.ocrSubmitAreaSum /= denom;
    mean.ocrSubmitSmallCount /= denom;
    mean.ocrSubmitMediumCount /= denom;
    mean.ocrSubmitLargeCount /= denom;
    mean.ocrSubmitTextCount /= denom;
    mean.ocrSubmitTitleCount /= denom;
    mean.ocrSubmitCodeCount /= denom;
    mean.ocrSubmitListCount /= denom;
    mean.ocrDedupSkippedCount /= denom;
    mean.tableNpuSubmitCount /= denom;
    mean.tableDedupSkippedCount /= denom;
    mean.ocrTimeoutCount /= denom;
    mean.ocrBufferedResultHitCount /= denom;
    if (mean.ocrSubmitCount > 0.0) {
        mean.ocrSubmitAreaMean = mean.ocrSubmitAreaSum / mean.ocrSubmitCount;
    }
    if (ocrSubmitCountTotal > 0.0) {
        mean.ocrSubmitAreaP50 = ocrSubmitAreaP50Weighted / ocrSubmitCountTotal;
        mean.ocrSubmitAreaP95 = ocrSubmitAreaP95Weighted / ocrSubmitCountTotal;
    }
    mean.pipelineMode = iterations.front().pipelineMode;
    return mean;
}

BenchmarkIterationResult runOnce(DocPipeline& pipeline, const std::string& inputPath) {
    const auto pipelineStart = std::chrono::steady_clock::now();
    const auto result = pipeline.processPdf(inputPath);
    const auto pipelineEnd = std::chrono::steady_clock::now();

    BenchmarkIterationResult iteration;
    iteration.totalTimeMs = result.totalTimeMs;
    iteration.pipelineCallMs =
        std::chrono::duration<double, std::milli>(pipelineEnd - pipelineStart).count();
    iteration.processedPages = result.processedPages;
    iteration.stageStats = result.stats;
    iteration.pageTimesMs.reserve(result.pages.size());
    for (const auto& page : result.pages) {
        iteration.pageTimesMs.push_back(page.totalTimeMs);
    }
    iteration.markdownSha256 = sha256::digest(result.markdown);
    iteration.contentListSha256 = sha256::digest(result.contentListJson);
    iteration.pipelineMode = result.stats.pipelineMode;
    iteration.ocrOuterMode = ocrOuterModeToString(pipeline.config().runtime.ocrOuterMode);
    return iteration;
}

BenchmarkCaseResult runBenchmarkCase(
    const BenchmarkCaseDefinition& definition,
    int warmupIterations,
    int measureIterations,
    const std::string& projectRoot,
    const std::string& outputRoot,
    PipelineMode pipelineMode,
    OcrOuterMode ocrOuterMode,
    size_t ocrShadowWindow)
{
    PipelineConfig config = PipelineConfig::Default(projectRoot);
    config.runtime.pipelineMode = pipelineMode;
    config.runtime.ocrOuterMode = ocrOuterMode;
    config.runtime.ocrShadowWindow = std::max<size_t>(1, ocrShadowWindow);
    config.runtime.outputDir =
        (fs::path(outputRoot) / definition.name / pipelineModeToString(pipelineMode) /
         ocrOuterModeToString(ocrOuterMode)).string();
    config.runtime.saveImages = false;
    config.runtime.saveVisualization = false;
    definition.configure(config);

    DocPipeline pipeline(config);
    if (!pipeline.initialize()) {
        throw std::runtime_error("Failed to initialize pipeline for case: " + definition.name);
    }

    BenchmarkCaseResult result;
    result.definition = definition;
    result.warmupIterations = warmupIterations;
    result.pipelineMode = pipelineMode;
    result.ocrOuterMode = ocrOuterMode;

    for (int i = 0; i < warmupIterations; ++i) {
        (void)runOnce(pipeline, definition.inputPath);
    }

    for (int i = 0; i < measureIterations; ++i) {
        result.iterations.push_back(runOnce(pipeline, definition.inputPath));
    }

    return result;
}

std::vector<BenchmarkCaseDefinition> makeCases(const std::string& projectRoot) {
    const fs::path root(projectRoot);
    return {
        {
            "single_page_pdf",
            "Single-page baseline (layout only, OCR/table disabled)",
            (root / "test_files" / "rmrb2026010601_origin.pdf").string(),
            [](PipelineConfig& cfg) {
                cfg.runtime.maxPages = 1;
                cfg.runtime.startPageId = 0;
                cfg.runtime.endPageId = 0;
                cfg.stages.enableOcr = false;
                cfg.stages.enableWiredTable = false;
                cfg.stages.enableFormula = false;
            },
        },
        {
            "multi_page_pdf",
            "Multi-page baseline (layout only, OCR/table disabled)",
            (root / "test_files" / "BVRC_Meeting_Minutes_2024-04_origin.pdf").string(),
            [](PipelineConfig& cfg) {
                cfg.stages.enableOcr = false;
                cfg.stages.enableWiredTable = false;
                cfg.stages.enableFormula = false;
            },
        },
        {
            "ocr_only",
            "Layout + OCR only on OCR-heavy document",
            (root / "test_files" / "small_ocr_origin.pdf").string(),
            [](PipelineConfig& cfg) {
                cfg.stages.enableWiredTable = false;
                cfg.stages.enableFormula = false;
                cfg.runtime.saveImages = false;
            },
        },
        {
            "table_heavy",
            "Table-heavy document (layout + table, OCR disabled)",
            (root / "test_files" / "表格0.pdf").string(),
            [](PipelineConfig& cfg) {
                cfg.stages.enableOcr = false;
                cfg.stages.enableFormula = false;
            },
        },
        {
            "layout_ocr_table_full_chain",
            "Full chain on mixed finance document (first two pages)",
            (root / "test_files" / "比亚迪财报_origin.pdf").string(),
            [](PipelineConfig& cfg) {
                cfg.runtime.maxPages = 2;
                cfg.runtime.startPageId = 0;
                cfg.runtime.endPageId = 1;
            },
        },
    };
}

json buildJsonSummary(const BenchmarkCaseResult& result) {
    if (result.status != "ok") {
        return json{
            {"name", result.definition.name},
            {"pipeline_mode", pipelineModeToString(result.pipelineMode)},
            {"ocr_outer_mode", ocrOuterModeToString(result.ocrOuterMode)},
            {"description", result.definition.description},
            {"input_path", result.definition.inputPath},
            {"warmup_iterations", result.warmupIterations},
            {"measured_iterations", result.iterations.size()},
            {"status", result.status},
            {"error", result.error},
        };
    }

    std::vector<double> documentTotals;
    std::vector<double> pipelineCallTotals;
    std::vector<double> allPageTotals;
    std::set<std::string> markdownHashes;
    std::set<std::string> contentListHashes;
    for (const auto& iteration : result.iterations) {
        documentTotals.push_back(iteration.totalTimeMs);
        pipelineCallTotals.push_back(iteration.pipelineCallMs);
        allPageTotals.insert(allPageTotals.end(), iteration.pageTimesMs.begin(), iteration.pageTimesMs.end());
        markdownHashes.insert(iteration.markdownSha256);
        contentListHashes.insert(iteration.contentListSha256);
    }

    json rawIterations = json::array();
    for (const auto& iteration : result.iterations) {
        rawIterations.push_back(json{
            {"total_time_ms", iteration.totalTimeMs},
            {"pipeline_call_ms", iteration.pipelineCallMs},
            {"processed_pages", iteration.processedPages},
            {"page_times_ms", iteration.pageTimesMs},
            {"markdown_sha256", iteration.markdownSha256},
            {"content_list_sha256", iteration.contentListSha256},
            {"stats", benchmarkStatsToJson(iteration)},
            {"stage_stats", stageStatsToJson(iteration.stageStats)},
        });
    }

    const auto documentSummary = summarizeSamples(documentTotals);
    const auto pipelineCallSummary = summarizeSamples(pipelineCallTotals);
    const auto pageSummary = summarizeSamples(allPageTotals);
    const DocumentStageStats meanStages = meanStageStats(result.iterations);
    BenchmarkIterationResult meanIteration;
    meanIteration.totalTimeMs = documentSummary.meanMs;
    meanIteration.pipelineCallMs = pipelineCallSummary.meanMs;
    meanIteration.processedPages =
        result.iterations.empty() ? 0 : result.iterations.front().processedPages;
    meanIteration.stageStats = meanStages;
    meanIteration.pipelineMode = pipelineModeToString(result.pipelineMode);
    meanIteration.ocrOuterMode = ocrOuterModeToString(result.ocrOuterMode);

    return json{
        {"name", result.definition.name},
        {"pipeline_mode", pipelineModeToString(result.pipelineMode)},
        {"ocr_outer_mode", ocrOuterModeToString(result.ocrOuterMode)},
        {"description", result.definition.description},
        {"input_path", result.definition.inputPath},
        {"warmup_iterations", result.warmupIterations},
        {"measured_iterations", result.iterations.size()},
        {"status", "ok"},
        {"document_total", summaryToJson(documentSummary)},
        {"pipeline_call", summaryToJson(pipelineCallSummary)},
        {"page_total", summaryToJson(pageSummary)},
        {"mean_stats", benchmarkStatsToJson(meanIteration)},
        {"mean_stage_breakdown", stageStatsToJson(meanStages)},
        {"markdown_sha256", markdownHashes.empty() ? std::string() : *markdownHashes.begin()},
        {"markdown_hash_consistent", markdownHashes.size() <= 1},
        {"content_list_sha256", contentListHashes.empty() ? std::string() : *contentListHashes.begin()},
        {"content_list_hash_consistent", contentListHashes.size() <= 1},
        {"hash_mismatch", markdownHashes.size() > 1 || contentListHashes.size() > 1},
        {"iterations", std::move(rawIterations)},
    };
}

std::string buildHumanSummary(const json& summary) {
    std::ostringstream out;
    out << summary.value("name", "(unknown)") << "\n";
    out << "  pipeline_mode: " << summary.value("pipeline_mode", "serial") << "\n";
    out << "  ocr_outer_mode: " << summary.value("ocr_outer_mode", "immediate_per_task") << "\n";
    out << "  description: " << summary.value("description", "") << "\n";
    out << "  input: " << summary.value("input_path", "") << "\n";
    if (summary.value("status", "ok") != "ok") {
        out << "  status: " << summary.value("status", "blocked") << "\n";
        out << "  error: " << summary.value("error", "(unknown)") << "\n";
        return out.str();
    }

    const auto& documentTotal = summary.at("document_total");
    const auto& pageTotal = summary.at("page_total");
    const auto& stageBreakdown = summary.at("mean_stage_breakdown");
    out << "  warmup: " << summary.value("warmup_iterations", 0)
        << ", measured_iterations: " << summary.value("measured_iterations", 0) << "\n";
    out << "  document_total: mean=" << formatMs(documentTotal.value("mean_ms", 0.0))
        << ", p50=" << formatMs(documentTotal.value("p50_ms", 0.0))
        << ", p95=" << formatMs(documentTotal.value("p95_ms", 0.0))
        << ", min=" << formatMs(documentTotal.value("min_ms", 0.0))
        << ", max=" << formatMs(documentTotal.value("max_ms", 0.0)) << "\n";
    const auto& pipelineCall = summary.at("pipeline_call");
    out << "  pipeline_call: mean=" << formatMs(pipelineCall.value("mean_ms", 0.0))
        << ", p50=" << formatMs(pipelineCall.value("p50_ms", 0.0))
        << ", p95=" << formatMs(pipelineCall.value("p95_ms", 0.0)) << "\n";
    out << "  page_total: samples=" << pageTotal.value("samples", 0)
        << ", mean=" << formatMs(pageTotal.value("mean_ms", 0.0))
        << ", p50=" << formatMs(pageTotal.value("p50_ms", 0.0))
        << ", p95=" << formatMs(pageTotal.value("p95_ms", 0.0)) << "\n";
    const auto& iterations = summary.at("iterations");
    const int pagesPerIteration = iterations.empty()
        ? 0
        : iterations.front().value("processed_pages", 0);
    out << "  pages_per_iteration: " << pagesPerIteration << "\n";
    out << "  pages_per_sec: "
        << formatCount(summary.at("mean_stats").value("pages_per_sec", 0.0)) << "\n";
    out << "  overlap_factor: "
        << formatCount(stageBreakdown.value("pipeline_overlap_factor", 0.0)) << "\n";
    out << "  mean_stage_breakdown:\n";
    out << "    pdf_render=" << formatMs(stageBreakdown.value("pdf_render_ms", 0.0))
        << ", layout=" << formatMs(stageBreakdown.value("layout_ms", 0.0))
        << ", ocr=" << formatMs(stageBreakdown.value("ocr_ms", 0.0))
        << ", table=" << formatMs(stageBreakdown.value("table_ms", 0.0))
        << ", reading_order=" << formatMs(stageBreakdown.value("reading_order_ms", 0.0))
        << ", output_gen=" << formatMs(stageBreakdown.value("output_gen_ms", 0.0))
        << ", npu_serial=" << formatMs(stageBreakdown.value("npu_serial_ms", 0.0))
        << ", cpu_only=" << formatMs(stageBreakdown.value("cpu_only_ms", 0.0))
        << ", npu_lock_wait=" << formatMs(stageBreakdown.value("npu_lock_wait_ms", 0.0))
        << ", npu_lock_hold=" << formatMs(stageBreakdown.value("npu_lock_hold_ms", 0.0))
        << ", npu_service=" << formatMs(stageBreakdown.value("npu_service_ms", 0.0))
        << ", npu_slot_wait=" << formatMs(stageBreakdown.value("npu_slot_wait_ms", 0.0))
        << ", layout_npu_service=" << formatMs(stageBreakdown.value("layout_npu_service_ms", 0.0))
        << ", layout_npu_slot_wait=" << formatMs(stageBreakdown.value("layout_npu_slot_wait_ms", 0.0))
        << ", ocr_outer_slot_hold=" << formatMs(stageBreakdown.value("ocr_outer_slot_hold_ms", 0.0))
        << ", ocr_submodule_window=" << formatMs(stageBreakdown.value("ocr_submodule_window_ms", 0.0))
        << ", ocr_slot_wait=" << formatMs(stageBreakdown.value("ocr_slot_wait_ms", 0.0))
        << ", ocr_collect_wait=" << formatMs(stageBreakdown.value("ocr_collect_wait_ms", 0.0))
        << ", table_npu_service=" << formatMs(stageBreakdown.value("table_npu_service_ms", 0.0))
        << ", table_npu_slot_wait=" << formatMs(stageBreakdown.value("table_npu_slot_wait_ms", 0.0))
        << ", table_ocr_service=" << formatMs(stageBreakdown.value("table_ocr_service_ms", 0.0))
        << ", table_ocr_slot_wait=" << formatMs(stageBreakdown.value("table_ocr_slot_wait_ms", 0.0))
        << ", cpu_pre=" << formatMs(stageBreakdown.value("cpu_pre_ms", 0.0))
        << ", cpu_post=" << formatMs(stageBreakdown.value("cpu_post_ms", 0.0))
        << ", finalize_cpu=" << formatMs(stageBreakdown.value("finalize_cpu_ms", 0.0))
        << ", table_finalize=" << formatMs(stageBreakdown.value("table_finalize_ms", 0.0))
        << ", ocr_collect_or_merge=" << formatMs(stageBreakdown.value("ocr_collect_or_merge_ms", 0.0))
        << ", layout_queue_wait=" << formatMs(stageBreakdown.value("layout_queue_wait_ms", 0.0))
        << ", plan_queue_wait=" << formatMs(stageBreakdown.value("plan_queue_wait_ms", 0.0))
        << ", ocr_table_queue_wait=" << formatMs(stageBreakdown.value("ocr_table_queue_wait_ms", 0.0))
        << ", finalize_queue_wait=" << formatMs(stageBreakdown.value("finalize_queue_wait_ms", 0.0))
        << ", render_queue_push_block=" << formatMs(stageBreakdown.value("render_queue_push_block_ms", 0.0))
        << ", layout_queue_push_block=" << formatMs(stageBreakdown.value("layout_queue_push_block_ms", 0.0))
        << ", plan_queue_push_block=" << formatMs(stageBreakdown.value("plan_queue_push_block_ms", 0.0))
        << ", ocr_table_queue_push_block=" << formatMs(stageBreakdown.value("ocr_table_queue_push_block_ms", 0.0))
        << ", queue_backpressure=" << formatMs(stageBreakdown.value("queue_backpressure_ms", 0.0))
        << ", figure=" << formatMs(stageBreakdown.value("figure_ms", 0.0))
        << ", formula=" << formatMs(stageBreakdown.value("formula_ms", 0.0))
        << ", unsupported=" << formatMs(stageBreakdown.value("unsupported_ms", 0.0))
        << "\n";
    out << "  mean_attribution_counts:\n";
    out << "    text_boxes_raw=" << formatCount(stageBreakdown.value("text_boxes_raw_count", 0.0))
        << ", text_boxes_after_dedup=" << formatCount(stageBreakdown.value("text_boxes_after_dedup_count", 0.0))
        << ", table_boxes_raw=" << formatCount(stageBreakdown.value("table_boxes_raw_count", 0.0))
        << ", table_boxes_after_dedup=" << formatCount(stageBreakdown.value("table_boxes_after_dedup_count", 0.0))
        << ", ocr_submit=" << formatCount(stageBreakdown.value("ocr_submit_count", 0.0))
        << ", ocr_submit_area_sum=" << formatCount(stageBreakdown.value("ocr_submit_area_sum", 0.0))
        << ", ocr_submit_area_mean=" << formatCount(stageBreakdown.value("ocr_submit_area_mean", 0.0))
        << ", ocr_submit_area_p50=" << formatCount(stageBreakdown.value("ocr_submit_area_p50", 0.0))
        << ", ocr_submit_area_p95=" << formatCount(stageBreakdown.value("ocr_submit_area_p95", 0.0))
        << ", ocr_submit_small=" << formatCount(stageBreakdown.value("ocr_submit_small_count", 0.0))
        << ", ocr_submit_medium=" << formatCount(stageBreakdown.value("ocr_submit_medium_count", 0.0))
        << ", ocr_submit_large=" << formatCount(stageBreakdown.value("ocr_submit_large_count", 0.0))
        << ", ocr_submit_text=" << formatCount(stageBreakdown.value("ocr_submit_text_count", 0.0))
        << ", ocr_submit_title=" << formatCount(stageBreakdown.value("ocr_submit_title_count", 0.0))
        << ", ocr_submit_code=" << formatCount(stageBreakdown.value("ocr_submit_code_count", 0.0))
        << ", ocr_submit_list=" << formatCount(stageBreakdown.value("ocr_submit_list_count", 0.0))
        << ", ocr_dedup_skipped=" << formatCount(stageBreakdown.value("ocr_dedup_skipped_count", 0.0))
        << ", table_npu_submit=" << formatCount(stageBreakdown.value("table_npu_submit_count", 0.0))
        << ", table_dedup_skipped=" << formatCount(stageBreakdown.value("table_dedup_skipped_count", 0.0))
        << ", ocr_timeout=" << formatCount(stageBreakdown.value("ocr_timeout_count", 0.0))
        << ", ocr_buffered_result_hit=" << formatCount(stageBreakdown.value("ocr_buffered_result_hit_count", 0.0))
        << ", ocr_inflight_peak=" << formatCount(stageBreakdown.value("ocr_inflight_peak", 0.0))
        << ", ocr_buffered_out_of_order=" << formatCount(stageBreakdown.value("ocr_buffered_out_of_order_count", 0.0))
        << "\n";
    out << "  markdown_sha256: " << summary.value("markdown_sha256", "") << "\n";
    out << "  content_list_sha256: " << summary.value("content_list_sha256", "") << "\n";
    out << "  hash_mismatch: " << (summary.value("hash_mismatch", false) ? "true" : "false") << "\n";
    return out.str();
}

const BenchmarkCaseDefinition* findCaseDefinition(
    const std::vector<BenchmarkCaseDefinition>& cases,
    const std::string& name)
{
    for (const auto& definition : cases) {
        if (definition.name == name) {
            return &definition;
        }
    }
    return nullptr;
}

int runWorkerMode(
    const std::string& caseName,
    int warmup,
    int iterations,
    const std::string& projectRoot,
    const std::string& outputDir,
    const std::string& jsonOutPath,
    PipelineMode pipelineMode,
    OcrOuterMode ocrOuterMode,
    size_t ocrShadowWindow)
{
    const auto definitions = makeCases(projectRoot);
    const auto* definition = findCaseDefinition(definitions, caseName);
    if (definition == nullptr) {
        std::cerr << "Unknown worker case: " << caseName << "\n";
        return 1;
    }

    BenchmarkCaseResult result;
    try {
        result = runBenchmarkCase(
            *definition,
            warmup,
            iterations,
            projectRoot,
            outputDir,
            pipelineMode,
            ocrOuterMode,
            ocrShadowWindow);
    } catch (const std::exception& e) {
        result.definition = *definition;
        result.warmupIterations = warmup;
        result.pipelineMode = pipelineMode;
        result.ocrOuterMode = ocrOuterMode;
        result.status = "blocked";
        result.error = e.what();
    }

    const fs::path outputPath(jsonOutPath);
    if (!outputPath.parent_path().empty()) {
        fs::create_directories(outputPath.parent_path());
    }
    std::ofstream out(outputPath);
    out << buildJsonSummary(result).dump(2);
    return 0;
}

json runCaseInSubprocess(
    const std::string& binaryPath,
    const BenchmarkCaseDefinition& definition,
    int warmup,
    int iterations,
    const std::string& projectRoot,
    const std::string& outputDir,
    PipelineMode pipelineMode,
    OcrOuterMode ocrOuterMode,
    size_t ocrShadowWindow)
{
    const fs::path workerJsonPath =
        fs::path(outputDir) /
        (definition.name + "_" + pipelineModeToString(pipelineMode) + "_" +
         ocrOuterModeToString(ocrOuterMode) + "_worker_summary.json");
    if (!workerJsonPath.parent_path().empty()) {
        fs::create_directories(workerJsonPath.parent_path());
    }
    fs::remove(workerJsonPath);

    std::vector<std::string> args = {
        binaryPath,
        "--worker-case", definition.name,
        "--worker-json", workerJsonPath.string(),
        "--iterations", std::to_string(iterations),
        "--warmup", std::to_string(warmup),
        "--output-dir", outputDir,
        "--project-root", projectRoot,
        "--pipeline-mode", pipelineModeToString(pipelineMode),
        "--ocr-outer-mode", ocrOuterModeToString(ocrOuterMode),
        "--ocr-shadow-window", std::to_string(std::max<size_t>(1, ocrShadowWindow)),
    };

    pid_t pid = fork();
    if (pid == 0) {
        std::vector<char*> argvExec;
        argvExec.reserve(args.size() + 1);
        for (auto& arg : args) {
            argvExec.push_back(arg.data());
        }
        argvExec.push_back(nullptr);
        execv(binaryPath.c_str(), argvExec.data());
        _exit(127);
    }
    if (pid < 0) {
        return json{
            {"name", definition.name},
            {"pipeline_mode", pipelineModeToString(pipelineMode)},
            {"ocr_outer_mode", ocrOuterModeToString(ocrOuterMode)},
            {"description", definition.description},
            {"input_path", definition.inputPath},
            {"warmup_iterations", warmup},
            {"measured_iterations", 0},
            {"status", "blocked"},
            {"error", "fork_failed"},
        };
    }

    int status = 0;
    waitpid(pid, &status, 0);

    if (WIFEXITED(status) && WEXITSTATUS(status) == 0 && fs::exists(workerJsonPath)) {
        std::ifstream in(workerJsonPath);
        return json::parse(in, nullptr, false);
    }

    std::ostringstream error;
    if (WIFSIGNALED(status)) {
        error << "worker_killed_by_signal_" << WTERMSIG(status);
    } else if (WIFEXITED(status)) {
        error << "worker_exit_code_" << WEXITSTATUS(status);
    } else {
        error << "worker_failed_without_status";
    }

        return json{
            {"name", definition.name},
            {"pipeline_mode", pipelineModeToString(pipelineMode)},
            {"ocr_outer_mode", ocrOuterModeToString(ocrOuterMode)},
            {"description", definition.description},
        {"input_path", definition.inputPath},
        {"warmup_iterations", warmup},
        {"measured_iterations", 0},
        {"status", "blocked"},
        {"error", error.str()},
    };
}

void printUsage(const char* program) {
    std::cout << "RapidDoc Phase 2 Benchmark\n\n";
    std::cout << "Usage: " << program << " [options]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --iterations <n>   Measured iterations per case (default: 5)\n";
    std::cout << "  --warmup <n>       Warmup iterations per case (default: 1)\n";
    std::cout << "  --case <name>      Run only the named case (repeatable)\n";
    std::cout << "  --json-out <path>  Write JSON summary to file\n";
    std::cout << "  --output-dir <p>   Benchmark scratch output dir (default: ./output-benchmark)\n";
    std::cout << "  --pipeline-mode <m> serial|page_pipeline_mvp|both (default: both)\n";
    std::cout << "  --ocr-outer-mode <m> immediate_per_task|shadow_windowed_collect (default: immediate_per_task)\n";
    std::cout << "  --ocr-shadow-window <n> Max OCR inflight window for shadow mode (default: 8)\n";
    std::cout << "  --project-root <p> Override project root (internal/debug)\n";
    std::cout << "  --help             Show this help\n";
}

json buildModeComparison(
    const BenchmarkCaseDefinition& definition,
    const json& serialSummary,
    const json& pipelineSummary)
{
    if (serialSummary.value("status", "ok") != "ok" ||
        pipelineSummary.value("status", "ok") != "ok") {
        return json{
            {"name", definition.name},
            {"status", "blocked"},
            {"serial_status", serialSummary.value("status", "blocked")},
            {"page_pipeline_mvp_status", pipelineSummary.value("status", "blocked")},
        };
    }

    const double serialWallMs =
        serialSummary.at("document_total").value("mean_ms", 0.0);
    const double pipelineWallMs =
        pipelineSummary.at("document_total").value("mean_ms", 0.0);
    const double serialPagesPerSec =
        serialSummary.at("mean_stats").value("pages_per_sec", 0.0);
    const double pipelinePagesPerSec =
        pipelineSummary.at("mean_stats").value("pages_per_sec", 0.0);
    const double serialOverlap =
        serialSummary.at("mean_stage_breakdown").value("pipeline_overlap_factor", 0.0);
    const double pipelineOverlap =
        pipelineSummary.at("mean_stage_breakdown").value("pipeline_overlap_factor", 0.0);
    const bool markdownHashMatch =
        serialSummary.value("markdown_sha256", "") ==
        pipelineSummary.value("markdown_sha256", "");
    const bool contentListHashMatch =
        serialSummary.value("content_list_sha256", "") ==
        pipelineSummary.value("content_list_sha256", "");

    return json{
        {"name", definition.name},
        {"ocr_outer_mode", serialSummary.value("ocr_outer_mode", "immediate_per_task")},
        {"status", "ok"},
        {"serial_wall_time_ms", serialWallMs},
        {"page_pipeline_mvp_wall_time_ms", pipelineWallMs},
        {"wall_time_delta_ms", pipelineWallMs - serialWallMs},
        {"wall_time_speedup_ratio", pipelineWallMs > 0.0 ? serialWallMs / pipelineWallMs : 0.0},
        {"serial_pages_per_sec", serialPagesPerSec},
        {"page_pipeline_mvp_pages_per_sec", pipelinePagesPerSec},
        {"pages_per_sec_delta", pipelinePagesPerSec - serialPagesPerSec},
        {"serial_overlap_factor", serialOverlap},
        {"page_pipeline_mvp_overlap_factor", pipelineOverlap},
        {"overlap_factor_delta", pipelineOverlap - serialOverlap},
        {"markdown_hash_match", markdownHashMatch},
        {"content_list_hash_match", contentListHashMatch},
        {"hash_mismatch", !(markdownHashMatch && contentListHashMatch)},
    };
}

} // namespace

int main(int argc, char** argv) {
    std::string projectRoot = PROJECT_ROOT_DIR;
    int iterations = 5;
    int warmup = 1;
    std::string jsonOutPath;
    std::string outputDir = (fs::path(projectRoot) / "output-benchmark").string();
    std::string workerCaseName;
    std::string workerJsonPath;
    std::string pipelineModeArg = "both";
    std::string ocrOuterModeArg = "immediate_per_task";
    size_t ocrShadowWindow = 8;
    std::set<std::string> requestedCases;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::max(1, std::atoi(argv[++i]));
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::max(0, std::atoi(argv[++i]));
        } else if (arg == "--case" && i + 1 < argc) {
            requestedCases.insert(argv[++i]);
        } else if (arg == "--json-out" && i + 1 < argc) {
            jsonOutPath = argv[++i];
        } else if (arg == "--output-dir" && i + 1 < argc) {
            outputDir = argv[++i];
        } else if (arg == "--pipeline-mode" && i + 1 < argc) {
            pipelineModeArg = argv[++i];
        } else if (arg == "--ocr-outer-mode" && i + 1 < argc) {
            ocrOuterModeArg = argv[++i];
        } else if (arg == "--ocr-shadow-window" && i + 1 < argc) {
            const long long parsed = std::strtoll(argv[++i], nullptr, 10);
            ocrShadowWindow = parsed > 0 ? static_cast<size_t>(parsed) : 1;
        } else if (arg == "--project-root" && i + 1 < argc) {
            projectRoot = argv[++i];
        } else if (arg == "--worker-case" && i + 1 < argc) {
            workerCaseName = argv[++i];
        } else if (arg == "--worker-json" && i + 1 < argc) {
            workerJsonPath = argv[++i];
        } else if (arg == "--help") {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            printUsage(argv[0]);
            return 1;
        }
    }

    spdlog::set_level(spdlog::level::err);

    std::vector<PipelineMode> requestedModes;
    OcrOuterMode ocrOuterMode = OcrOuterMode::ImmediatePerTask;
    if (!parseOcrOuterMode(ocrOuterModeArg, ocrOuterMode)) {
        std::cerr << "Invalid --ocr-outer-mode: " << ocrOuterModeArg << "\n";
        return 1;
    }
    if (pipelineModeArg == "both") {
        requestedModes = {
            PipelineMode::Serial,
            PipelineMode::PagePipelineMvp,
        };
    } else {
        PipelineMode parsedMode = PipelineMode::Serial;
        if (!parsePipelineMode(pipelineModeArg, parsedMode)) {
            std::cerr << "Invalid --pipeline-mode: " << pipelineModeArg << "\n";
            return 1;
        }
        requestedModes = {parsedMode};
    }

    if (!workerCaseName.empty()) {
        if (workerJsonPath.empty()) {
            std::cerr << "--worker-json is required with --worker-case\n";
            return 1;
        }
        return runWorkerMode(
            workerCaseName,
            warmup,
            iterations,
            projectRoot,
            outputDir,
            workerJsonPath,
            requestedModes.front(),
            ocrOuterMode,
            ocrShadowWindow);
    }

    std::vector<BenchmarkCaseDefinition> selectedCases;
    for (const auto& definition : makeCases(projectRoot)) {
        if (requestedCases.empty() || requestedCases.count(definition.name) > 0) {
            selectedCases.push_back(definition);
        }
    }

    if (!requestedCases.empty() && selectedCases.size() != requestedCases.size()) {
        std::cerr << "Unknown benchmark case requested.\n";
        return 1;
    }

    json root = json::object();
    root["generated_at_utc_ms"] =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    root["iterations"] = iterations;
    root["warmup"] = warmup;
    root["pipeline_modes"] = json::array();
    for (PipelineMode mode : requestedModes) {
        root["pipeline_modes"].push_back(pipelineModeToString(mode));
    }
    root["ocr_outer_mode"] = ocrOuterModeToString(ocrOuterMode);
    root["ocr_shadow_window"] = ocrShadowWindow;
    root["cases"] = json::array();
    root["comparisons"] = json::array();
    const std::string binaryPath = fs::absolute(argv[0]).string();

    for (const auto& definition : selectedCases) {
        if (!fs::exists(definition.inputPath)) {
            std::cerr << "Missing input fixture for case " << definition.name
                      << ": " << definition.inputPath << "\n";
            return 1;
        }

        json caseEntry{
            {"name", definition.name},
            {"description", definition.description},
            {"input_path", definition.inputPath},
            {"runs", json::array()},
        };

        json serialSummary;
        json pipelineSummary;
        bool haveSerial = false;
        bool havePagePipeline = false;
        for (PipelineMode mode : requestedModes) {
            const json summary = runCaseInSubprocess(
                binaryPath,
                definition,
                warmup,
                iterations,
                projectRoot,
                outputDir,
                mode,
                ocrOuterMode,
                ocrShadowWindow);
            std::cout << buildHumanSummary(summary) << "\n";
            caseEntry["runs"].push_back(summary);
            if (mode == PipelineMode::Serial) {
                serialSummary = summary;
                haveSerial = true;
            } else if (mode == PipelineMode::PagePipelineMvp) {
                pipelineSummary = summary;
                havePagePipeline = true;
            }
        }

        if (haveSerial && havePagePipeline) {
            const json comparison = buildModeComparison(definition, serialSummary, pipelineSummary);
            caseEntry["ab_compare"] = comparison;
            root["comparisons"].push_back(comparison);
        }
        root["cases"].push_back(caseEntry);
    }

    if (!jsonOutPath.empty()) {
        const fs::path parent = fs::path(jsonOutPath).parent_path();
        if (!parent.empty()) {
            fs::create_directories(parent);
        }
        std::ofstream out(jsonOutPath);
        out << root.dump(2);
        std::cout << "Saved JSON benchmark summary: " << jsonOutPath << "\n";
    }

    return 0;
}
