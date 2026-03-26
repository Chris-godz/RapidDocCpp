#include <gtest/gtest.h>

#include "common/perf_utils.h"

using namespace rapid_doc;

TEST(PerfUtilsTest, summarizeSamplesComputesMeanAndPercentiles) {
    const PercentileSummary summary = summarizeSamples({10.0, 20.0, 30.0, 40.0, 50.0});

    EXPECT_EQ(summary.sampleCount, 5u);
    EXPECT_DOUBLE_EQ(summary.minMs, 10.0);
    EXPECT_DOUBLE_EQ(summary.meanMs, 30.0);
    EXPECT_DOUBLE_EQ(summary.maxMs, 50.0);
    EXPECT_DOUBLE_EQ(summary.p50Ms, 30.0);
    EXPECT_DOUBLE_EQ(summary.p95Ms, 48.0);
}

TEST(PerfUtilsTest, accumulateDocumentStageStatsSumsPageStageStats) {
    PageResult page0;
    page0.stats.layoutTimeMs = 10.0;
    page0.stats.ocrTimeMs = 20.0;
    page0.stats.figureTimeMs = 2.5;
    page0.stats.npuSerialTimeMs = 30.0;
    page0.stats.cpuOnlyTimeMs = 4.0;
    page0.stats.npuLockWaitTimeMs = 1.0;
    page0.stats.npuLockHoldTimeMs = 31.0;

    PageResult page1;
    page1.stats.layoutTimeMs = 1.0;
    page1.stats.tableTimeMs = 8.0;
    page1.stats.formulaTimeMs = 4.0;
    page1.stats.unsupportedTimeMs = 3.0;
    page1.stats.readingOrderTimeMs = 5.0;
    page1.stats.npuSerialTimeMs = 9.0;
    page1.stats.cpuOnlyTimeMs = 7.0;
    page1.stats.npuLockWaitTimeMs = 2.0;
    page1.stats.npuLockHoldTimeMs = 11.0;

    const DocumentStageStats stats = accumulateDocumentStageStats({page0, page1});

    EXPECT_DOUBLE_EQ(stats.layoutTimeMs, 11.0);
    EXPECT_DOUBLE_EQ(stats.ocrTimeMs, 20.0);
    EXPECT_DOUBLE_EQ(stats.tableTimeMs, 8.0);
    EXPECT_DOUBLE_EQ(stats.figureTimeMs, 2.5);
    EXPECT_DOUBLE_EQ(stats.formulaTimeMs, 4.0);
    EXPECT_DOUBLE_EQ(stats.unsupportedTimeMs, 3.0);
    EXPECT_DOUBLE_EQ(stats.readingOrderTimeMs, 5.0);
    EXPECT_DOUBLE_EQ(stats.npuSerialTimeMs, 39.0);
    EXPECT_DOUBLE_EQ(stats.cpuOnlyTimeMs, 11.0);
    EXPECT_DOUBLE_EQ(stats.npuLockWaitTimeMs, 3.0);
    EXPECT_DOUBLE_EQ(stats.npuLockHoldTimeMs, 42.0);
}

TEST(PerfUtilsTest, totalTrackedStageTimeIncludesDocumentOnlyStages) {
    DocumentStageStats stats;
    stats.pdfRenderTimeMs = 9.0;
    stats.layoutTimeMs = 10.0;
    stats.ocrTimeMs = 20.0;
    stats.tableTimeMs = 30.0;
    stats.figureTimeMs = 4.0;
    stats.formulaTimeMs = 5.0;
    stats.unsupportedTimeMs = 6.0;
    stats.readingOrderTimeMs = 7.0;
    stats.outputGenTimeMs = 8.0;

    EXPECT_DOUBLE_EQ(totalTrackedStageTimeMs(stats), 99.0);
}
