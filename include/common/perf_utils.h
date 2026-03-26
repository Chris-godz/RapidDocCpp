#pragma once

#include "common/types.h"

#include <cstddef>
#include <vector>

namespace rapid_doc {

struct PercentileSummary {
    size_t sampleCount = 0;
    double minMs = 0.0;
    double meanMs = 0.0;
    double maxMs = 0.0;
    double p50Ms = 0.0;
    double p95Ms = 0.0;
};

PercentileSummary summarizeSamples(std::vector<double> samples);
DocumentStageStats accumulateDocumentStageStats(const std::vector<PageResult>& pages);
double totalTrackedStageTimeMs(const PageStageStats& stats);
double totalTrackedStageTimeMs(const DocumentStageStats& stats);

} // namespace rapid_doc
