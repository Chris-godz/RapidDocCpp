/**
 * @file test_tensor_compare.cpp
 * @brief Unit tests for tensor comparison utilities
 */

#include <gtest/gtest.h>
#include "test_utils/tensor_compare.h"

using namespace rapid_doc::test_utils;

// ========================================
// Cosine Similarity Tests
// ========================================

TEST(TensorCompare, IdenticalTensors) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto r = compareTensors(a, a);
    EXPECT_TRUE(r.pass);
    EXPECT_NEAR(r.cosineSimilarity, 1.0, 1e-9);
    EXPECT_NEAR(r.rmse, 0.0, 1e-9);
    EXPECT_NEAR(r.maxAbsError, 0.0, 1e-9);
    EXPECT_NEAR(r.pearsonCorr, 1.0, 1e-9);
    EXPECT_EQ(r.mismatchCount, 0u);
}

TEST(TensorCompare, SmallQuantizationNoise) {
    // Simulate INT8 quantization noise: ~0.001 uniform additive
    std::vector<float> a = {0.5f, 1.2f, -0.3f, 2.1f, 0.0f, -1.5f, 0.8f, 3.0f};
    std::vector<float> b = {0.501f, 1.199f, -0.302f, 2.101f, 0.001f, -1.499f, 0.799f, 3.002f};

    CompareThresholds thresh;
    thresh.minCosineSimilarity = 0.999;
    thresh.maxRmse = 0.005;
    thresh.maxAbsError = 0.01;

    auto r = compareTensors(a, b, thresh);
    EXPECT_TRUE(r.pass) << r.summary();
    EXPECT_GT(r.cosineSimilarity, 0.9999);
    EXPECT_LT(r.rmse, 0.003);
}

TEST(TensorCompare, LargeDeviation_Fails) {
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {1.0f, 2.0f, 10.0f};  // Large diff on element 2

    auto r = compareTensors(a, b);
    EXPECT_FALSE(r.pass);
    EXPECT_NEAR(r.maxAbsError, 7.0, 1e-6);
}

TEST(TensorCompare, EmptyTensorsPasses) {
    std::vector<float> a, b;
    auto r = compareTensors(a, b);
    EXPECT_TRUE(r.pass);
}

TEST(TensorCompare, OrthogonalVectors) {
    std::vector<float> a = {1.0f, 0.0f, 0.0f};
    std::vector<float> b = {0.0f, 1.0f, 0.0f};

    auto r = compareTensors(a, b);
    EXPECT_NEAR(r.cosineSimilarity, 0.0, 1e-9);
    EXPECT_FALSE(r.pass);
}

TEST(TensorCompare, ScaledVectors_CosineStays) {
    // Cosine similarity should be invariant to scaling
    std::vector<float> a = {1.0f, 2.0f, 3.0f};
    std::vector<float> b = {10.0f, 20.0f, 30.0f};

    auto r = compareTensors(a, b);
    EXPECT_NEAR(r.cosineSimilarity, 1.0, 1e-9);
    // But RMSE / maxAbsError will be large → test should fail
    EXPECT_FALSE(r.pass);
}

// ========================================
// IoU Tests
// ========================================

TEST(IoU, PerfectOverlap) {
    float box[4] = {10, 20, 100, 200};
    EXPECT_NEAR(computeIoU(box, box), 1.0, 1e-6);
}

TEST(IoU, NoOverlap) {
    float a[4] = {0, 0, 10, 10};
    float b[4] = {20, 20, 30, 30};
    EXPECT_NEAR(computeIoU(a, b), 0.0, 1e-6);
}

TEST(IoU, PartialOverlap) {
    float a[4] = {0, 0, 10, 10};
    float b[4] = {5, 5, 15, 15};
    // intersection = 5*5 = 25, union = 100 + 100 - 25 = 175
    EXPECT_NEAR(computeIoU(a, b), 25.0 / 175.0, 1e-6);
}

// ========================================
// Edit Distance / Character Accuracy Tests
// ========================================

TEST(EditDistance, Identical) {
    EXPECT_EQ(editDistance("hello", "hello"), 0);
}

TEST(EditDistance, SingleInsert) {
    EXPECT_EQ(editDistance("hell", "hello"), 1);
}

TEST(EditDistance, MultipleEdits) {
    EXPECT_EQ(editDistance("kitten", "sitting"), 3);
}

TEST(CharAccuracy, Perfect) {
    EXPECT_NEAR(characterAccuracy("test", "test"), 1.0, 1e-9);
}

TEST(CharAccuracy, OneError) {
    // "test" vs "tast" → edit distance 1, maxLen 4 → 0.75
    EXPECT_NEAR(characterAccuracy("test", "tast"), 0.75, 1e-9);
}

TEST(CharAccuracy, BothEmpty) {
    EXPECT_NEAR(characterAccuracy("", ""), 1.0, 1e-9);
}
