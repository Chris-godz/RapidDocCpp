#include <gtest/gtest.h>
#include "metrics.h"
#include <vector>

using namespace rapid_doc;
using namespace rapid_doc::test_metrics;

TEST(MetricsTest, IoU_ExactMatch) {
    LayoutBox box1 = {10.0f, 10.0f, 100.0f, 100.0f, LayoutCategory::TEXT, 1.0f, 0};
    LayoutBox box2 = {10.0f, 10.0f, 100.0f, 100.0f, LayoutCategory::TEXT, 1.0f, 0};

    float iou = calculateIoU(box1, box2);
    EXPECT_FLOAT_EQ(iou, 1.0f);
}

TEST(MetricsTest, IoU_NoOverlap) {
    LayoutBox box1 = {10.0f, 10.0f, 50.0f, 50.0f, LayoutCategory::TEXT, 1.0f, 0};
    LayoutBox box2 = {60.0f, 60.0f, 100.0f, 100.0f, LayoutCategory::TEXT, 1.0f, 0};

    float iou = calculateIoU(box1, box2);
    EXPECT_FLOAT_EQ(iou, 0.0f);
}

TEST(MetricsTest, IoU_PartialOverlap) {
    // 10x10 to 30x30, area = 400
    LayoutBox box1 = {10.0f, 10.0f, 30.0f, 30.0f, LayoutCategory::TEXT, 1.0f, 0};
    // 20x20 to 40x40, area = 400
    // Intersection: 20x20 to 30x30, area = 100
    // Union: 400 + 400 - 100 = 700
    // IoU: 100 / 700 = 1/7 = 0.142857
    LayoutBox box2 = {20.0f, 20.0f, 40.0f, 40.0f, LayoutCategory::TEXT, 1.0f, 0};

    float iou = calculateIoU(box1, box2);
    EXPECT_NEAR(iou, 0.142857f, 1e-4);
}

TEST(MetricsTest, CosineSimilarity_Exact) {
    std::vector<float> v1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> v2 = {1.0f, 2.0f, 3.0f};
    
    float cos_sim = calculateCosineSimilarity(v1, v2);
    EXPECT_FLOAT_EQ(cos_sim, 1.0f);
}

TEST(MetricsTest, CosineSimilarity_Orthogonal) {
    std::vector<float> v1 = {1.0f, 0.0f};
    std::vector<float> v2 = {0.0f, 1.0f};
    
    float cos_sim = calculateCosineSimilarity(v1, v2);
    EXPECT_FLOAT_EQ(cos_sim, 0.0f);
}

TEST(MetricsTest, CosineSimilarity_Opposite) {
    std::vector<float> v1 = {1.0f, 2.0f, 3.0f};
    std::vector<float> v2 = {-1.0f, -2.0f, -3.0f};
    
    float cos_sim = calculateCosineSimilarity(v1, v2);
    EXPECT_FLOAT_EQ(cos_sim, -1.0f);
}

TEST(MetricsTest, EditDistance_Identical) {
    EXPECT_EQ(calculateEditDistance("hello", "hello"), 0);
    EXPECT_FLOAT_EQ(calculateCER("hello", "hello"), 0.0f);
}

TEST(MetricsTest, EditDistance_Difference) {
    // hello -> jello : 1 replace
    EXPECT_EQ(calculateEditDistance("hello", "jello"), 1);
    EXPECT_FLOAT_EQ(calculateCER("hello", "jello"), 1.0f / 5.0f);
    
    // kitten -> sitting : k->s, e->i, add g : 3 edits
    EXPECT_EQ(calculateEditDistance("kitten", "sitting"), 3);
}
