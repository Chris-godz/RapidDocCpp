#include "common/perf_utils.h"

#include <algorithm>
#include <numeric>

namespace rapid_doc {

namespace {

double percentileFromSorted(const std::vector<double>& values, double percentile) {
    if (values.empty()) {
        return 0.0;
    }

    const double clamped = std::max(0.0, std::min(100.0, percentile));
    const double rank = (clamped / 100.0) * static_cast<double>(values.size() - 1);
    const size_t lower = static_cast<size_t>(rank);
    const size_t upper = std::min(values.size() - 1, lower + 1);
    const double weight = rank - static_cast<double>(lower);
    return values[lower] + (values[upper] - values[lower]) * weight;
}

void accumulatePageStats(PageStageStats& target, const PageStageStats& source) {
    target.layoutTimeMs += source.layoutTimeMs;
    target.ocrTimeMs += source.ocrTimeMs;
    target.tableTimeMs += source.tableTimeMs;
    target.figureTimeMs += source.figureTimeMs;
    target.formulaTimeMs += source.formulaTimeMs;
    target.unsupportedTimeMs += source.unsupportedTimeMs;
    target.readingOrderTimeMs += source.readingOrderTimeMs;
    target.npuSerialTimeMs += source.npuSerialTimeMs;
    target.cpuOnlyTimeMs += source.cpuOnlyTimeMs;
    target.npuLockWaitTimeMs += source.npuLockWaitTimeMs;
    target.npuLockHoldTimeMs += source.npuLockHoldTimeMs;
    target.textBoxesRawCount += source.textBoxesRawCount;
    target.textBoxesAfterDedupCount += source.textBoxesAfterDedupCount;
    target.tableBoxesRawCount += source.tableBoxesRawCount;
    target.tableBoxesAfterDedupCount += source.tableBoxesAfterDedupCount;
    target.ocrSubmitCount += source.ocrSubmitCount;
    target.ocrSubmitAreaSum += source.ocrSubmitAreaSum;
    target.ocrSubmitSmallCount += source.ocrSubmitSmallCount;
    target.ocrSubmitMediumCount += source.ocrSubmitMediumCount;
    target.ocrSubmitLargeCount += source.ocrSubmitLargeCount;
    target.ocrSubmitTextCount += source.ocrSubmitTextCount;
    target.ocrSubmitTitleCount += source.ocrSubmitTitleCount;
    target.ocrSubmitCodeCount += source.ocrSubmitCodeCount;
    target.ocrSubmitListCount += source.ocrSubmitListCount;
    target.ocrDedupSkippedCount += source.ocrDedupSkippedCount;
    target.tableNpuSubmitCount += source.tableNpuSubmitCount;
    target.tableDedupSkippedCount += source.tableDedupSkippedCount;
    target.ocrTimeoutCount += source.ocrTimeoutCount;
    target.ocrBufferedResultHitCount += source.ocrBufferedResultHitCount;
}

} // namespace

PercentileSummary summarizeSamples(std::vector<double> samples) {
    PercentileSummary summary;
    summary.sampleCount = samples.size();
    if (samples.empty()) {
        return summary;
    }

    std::sort(samples.begin(), samples.end());
    summary.minMs = samples.front();
    summary.maxMs = samples.back();
    summary.meanMs = std::accumulate(samples.begin(), samples.end(), 0.0) /
                     static_cast<double>(samples.size());
    summary.p50Ms = percentileFromSorted(samples, 50.0);
    summary.p95Ms = percentileFromSorted(samples, 95.0);
    return summary;
}

DocumentStageStats accumulateDocumentStageStats(const std::vector<PageResult>& pages) {
    DocumentStageStats stats;
    double ocrSubmitAreaP50Weighted = 0.0;
    double ocrSubmitAreaP95Weighted = 0.0;
    for (const auto& page : pages) {
        accumulatePageStats(stats, page.stats);
        const double submitCount = std::max(0.0, page.stats.ocrSubmitCount);
        if (submitCount > 0.0) {
            ocrSubmitAreaP50Weighted += page.stats.ocrSubmitAreaP50 * submitCount;
            ocrSubmitAreaP95Weighted += page.stats.ocrSubmitAreaP95 * submitCount;
        }
    }

    if (stats.ocrSubmitCount > 0.0) {
        stats.ocrSubmitAreaMean = stats.ocrSubmitAreaSum / stats.ocrSubmitCount;
        stats.ocrSubmitAreaP50 = ocrSubmitAreaP50Weighted / stats.ocrSubmitCount;
        stats.ocrSubmitAreaP95 = ocrSubmitAreaP95Weighted / stats.ocrSubmitCount;
    }
    return stats;
}

double totalTrackedStageTimeMs(const PageStageStats& stats) {
    return stats.layoutTimeMs +
           stats.ocrTimeMs +
           stats.tableTimeMs +
           stats.figureTimeMs +
           stats.formulaTimeMs +
           stats.unsupportedTimeMs +
           stats.readingOrderTimeMs;
}

double totalTrackedStageTimeMs(const DocumentStageStats& stats) {
    return stats.pdfRenderTimeMs +
           totalTrackedStageTimeMs(static_cast<const PageStageStats&>(stats)) +
           stats.outputGenTimeMs;
}

} // namespace rapid_doc
