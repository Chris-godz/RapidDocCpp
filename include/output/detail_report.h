#pragma once

#include "common/config.h"
#include "common/types.h"

#include <string>

namespace rapid_doc {

struct DetailArtifactPaths {
    std::string outputDir;
    std::string markdownPath;
    std::string contentListPath;
    std::string layoutDir;
    std::string imagesDir;
};

struct DetailReportOptions {
    std::string inputPath;
    PipelineStages stageConfig;
    bool saveImages = false;
    bool saveVisualization = false;
    DetailArtifactPaths artifacts;
};

std::string buildDetailReport(
    const DocumentResult& result,
    const DetailReportOptions& options);

} // namespace rapid_doc
