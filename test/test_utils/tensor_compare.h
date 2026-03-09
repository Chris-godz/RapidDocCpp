/**
 * @file tensor_compare.h
 * @brief Tensor comparison utilities for DXNN vs ONNX accuracy validation
 *
 * Provides multiple metrics suitable for comparing quantized model outputs:
 *  - Cosine similarity    (direction-invariant, best for quantization noise)
 *  - RMSE                 (root mean square error)
 *  - Max absolute error   (L∞ norm)
 *  - Pearson correlation  (linear fit)
 *  - Element-wise close   (numpy-style allclose)
 *
 * Design rationale (from DL perspective):
 *  Quantized models (INT8/FP16) introduce small additive noise that is
 *  roughly uniform.  Cosine similarity is the single best metric because
 *  it measures angular distance in high-dimensional space and is
 *  invariant to uniform scaling — exactly the behavior we want when the
 *  quantization merely shifts magnitudes slightly.
 *  We complement it with L∞ and RMSE to detect outlier activations
 *  and average drift respectively.
 */

#pragma once

#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <string>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cassert>

namespace rapid_doc {
namespace test_utils {

/**
 * @brief Result of a tensor comparison
 */
struct CompareResult {
    bool pass = false;

    // Core metrics
    double cosineSimilarity = 0.0;
    double rmse             = 0.0;
    double maxAbsError      = 0.0;
    double pearsonCorr      = 0.0;
    double meanAbsError     = 0.0;

    // Counts
    size_t totalElements    = 0;
    size_t mismatchCount    = 0;    // Elements outside atol/rtol

    std::string summary() const {
        std::ostringstream ss;
        ss << "CompareResult { "
           << "pass=" << (pass ? "TRUE" : "FALSE")
           << ", cosine=" << cosineSimilarity
           << ", rmse=" << rmse
           << ", maxAbsErr=" << maxAbsError
           << ", meanAbsErr=" << meanAbsError
           << ", pearson=" << pearsonCorr
           << ", elements=" << totalElements
           << ", mismatches=" << mismatchCount
           << " }";
        return ss.str();
    }
};

/**
 * @brief Comparison thresholds
 */
struct CompareThresholds {
    double minCosineSimilarity = 0.995;   // >= 0.995
    double maxRmse             = 0.01;    // <= 0.01  (on normalized tensors)
    double maxAbsError         = 0.05;    // <= 0.05  (on normalized tensors)
    double minPearsonCorr      = 0.999;   // >= 0.999
    double atol                = 1e-3;    // absolute tolerance for allclose
    double rtol                = 1e-3;    // relative tolerance for allclose
};

/**
 * @brief Compare two float arrays using multiple DL-relevant metrics
 *
 * @param a        First tensor (e.g. DXNN output)
 * @param b        Second tensor (e.g. ONNX baseline output)
 * @param n        Number of elements
 * @param thresh   Acceptance thresholds
 * @return CompareResult with all metrics populated
 */
inline CompareResult compareTensors(
    const float* a,
    const float* b,
    size_t n,
    const CompareThresholds& thresh = {})
{
    CompareResult r;
    r.totalElements = n;

    if (n == 0) {
        r.pass = true;
        r.cosineSimilarity = 1.0;
        r.pearsonCorr = 1.0;
        return r;
    }

    // ---- Single-pass accumulation ----
    double dotAB  = 0.0, normA2 = 0.0, normB2 = 0.0;
    double sumA   = 0.0, sumB   = 0.0;
    double sumA2  = 0.0, sumB2  = 0.0, sumAB = 0.0;
    double sumSqErr = 0.0, sumAbsErr = 0.0;
    double maxAbs = 0.0;
    size_t mismatches = 0;

    for (size_t i = 0; i < n; ++i) {
        double va = static_cast<double>(a[i]);
        double vb = static_cast<double>(b[i]);
        double diff = va - vb;

        dotAB  += va * vb;
        normA2 += va * va;
        normB2 += vb * vb;
        sumA   += va;
        sumB   += vb;
        sumA2  += va * va;
        sumB2  += vb * vb;
        sumAB  += va * vb;

        double absDiff = std::abs(diff);
        sumSqErr += diff * diff;
        sumAbsErr += absDiff;
        if (absDiff > maxAbs) maxAbs = absDiff;

        // allclose check: |a - b| <= atol + rtol * |b|
        if (absDiff > thresh.atol + thresh.rtol * std::abs(vb)) {
            mismatches++;
        }
    }

    double dn = static_cast<double>(n);

    // Cosine similarity
    double denomCos = std::sqrt(normA2) * std::sqrt(normB2);
    r.cosineSimilarity = (denomCos > 1e-12) ? (dotAB / denomCos) : 1.0;

    // RMSE
    r.rmse = std::sqrt(sumSqErr / dn);

    // Max absolute error
    r.maxAbsError = maxAbs;

    // Mean absolute error
    r.meanAbsError = sumAbsErr / dn;

    // Pearson correlation
    double meanA = sumA / dn;
    double meanB = sumB / dn;
    double covAB = sumAB / dn - meanA * meanB;
    double stdA  = std::sqrt(std::max(0.0, sumA2 / dn - meanA * meanA));
    double stdB  = std::sqrt(std::max(0.0, sumB2 / dn - meanB * meanB));
    double denomPearson = stdA * stdB;
    r.pearsonCorr = (denomPearson > 1e-12) ? (covAB / denomPearson) : 1.0;

    // Mismatch count
    r.mismatchCount = mismatches;

    // Pass/fail decision: ALL thresholds must pass
    r.pass = (r.cosineSimilarity >= thresh.minCosineSimilarity)
          && (r.rmse             <= thresh.maxRmse)
          && (r.maxAbsError      <= thresh.maxAbsError)
          && (r.pearsonCorr      >= thresh.minPearsonCorr);

    return r;
}

/**
 * @brief Compare two std::vector<float> tensors
 */
inline CompareResult compareTensors(
    const std::vector<float>& a,
    const std::vector<float>& b,
    const CompareThresholds& thresh = {})
{
    assert(a.size() == b.size() && "Tensor sizes must match");
    return compareTensors(a.data(), b.data(), a.size(), thresh);
}

// ========================================
// Task-level comparison helpers
// ========================================

/**
 * @brief Compute Intersection-over-Union between two axis-aligned boxes
 * @param box format: [x0, y0, x1, y1]
 */
inline double computeIoU(const float* boxA, const float* boxB) {
    float x0 = std::max(boxA[0], boxB[0]);
    float y0 = std::max(boxA[1], boxB[1]);
    float x1 = std::min(boxA[2], boxB[2]);
    float y1 = std::min(boxA[3], boxB[3]);

    float interArea = std::max(0.0f, x1 - x0) * std::max(0.0f, y1 - y0);
    float areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
    float areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);
    float unionArea = areaA + areaB - interArea;

    return (unionArea > 0.0f) ? static_cast<double>(interArea / unionArea) : 0.0;
}

/**
 * @brief Edit distance (Levenshtein) between two strings
 * For OCR text accuracy comparison
 */
inline int editDistance(const std::string& a, const std::string& b) {
    int m = static_cast<int>(a.size());
    int n = static_cast<int>(b.size());
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));

    for (int i = 0; i <= m; ++i) dp[i][0] = i;
    for (int j = 0; j <= n; ++j) dp[0][j] = j;

    for (int i = 1; i <= m; ++i) {
        for (int j = 1; j <= n; ++j) {
            int cost = (a[i-1] == b[j-1]) ? 0 : 1;
            dp[i][j] = std::min({dp[i-1][j] + 1,
                                 dp[i][j-1] + 1,
                                 dp[i-1][j-1] + cost});
        }
    }
    return dp[m][n];
}

/**
 * @brief Character-level accuracy between two strings
 * accuracy = 1 - editDistance / max(len(a), len(b))
 */
inline double characterAccuracy(const std::string& a, const std::string& b) {
    if (a.empty() && b.empty()) return 1.0;
    int maxLen = static_cast<int>(std::max(a.size(), b.size()));
    return 1.0 - static_cast<double>(editDistance(a, b)) / maxLen;
}

} // namespace test_utils
} // namespace rapid_doc
