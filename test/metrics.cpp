#include "metrics.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace rapid_doc {
namespace test_metrics {

float calculateIoU(const LayoutBox& box1, const LayoutBox& box2) {
    float x_left = std::max(box1.x0, box2.x0);
    float y_top = std::max(box1.y0, box2.y0);
    float x_right = std::min(box1.x1, box2.x1);
    float y_bottom = std::min(box1.y1, box2.y1);

    if (x_right < x_left || y_bottom < y_top) {
        return 0.0f;
    }

    float intersection_area = (x_right - x_left) * (y_bottom - y_top);
    float box1_area = (box1.x1 - box1.x0) * (box1.y1 - box1.y0);
    float box2_area = (box2.x1 - box2.x0) * (box2.y1 - box2.y0);

    float iou = intersection_area / float(box1_area + box2_area - intersection_area);
    
    return std::max(0.0f, std::min(1.0f, iou));
}

float calculateCosineSimilarity(const float* vec1, const float* vec2, size_t size) {
    if (size == 0 || vec1 == nullptr || vec2 == nullptr) return 0.0f;

    double dot_product = 0.0;
    double norm_a = 0.0;
    double norm_b = 0.0;

    for (size_t i = 0; i < size; ++i) {
        dot_product += vec1[i] * vec2[i];
        norm_a += vec1[i] * vec1[i];
        norm_b += vec2[i] * vec2[i];
    }

    if (norm_a == 0.0 || norm_b == 0.0) {
        return 0.0f;
    }

    return static_cast<float>(dot_product / (std::sqrt(norm_a) * std::sqrt(norm_b)));
}

float calculateCosineSimilarity(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    if (vec1.size() != vec2.size()) {
        return 0.0f; // Invalid
    }
    return calculateCosineSimilarity(vec1.data(), vec2.data(), vec1.size());
}

int calculateEditDistance(const std::string& str1, const std::string& str2) {
    size_t len1 = str1.size();
    size_t len2 = str2.size();
    
    std::vector<std::vector<int>> d(len1 + 1, std::vector<int>(len2 + 1));
    
    for (size_t i = 0; i <= len1; ++i) d[i][0] = i;
    for (size_t j = 0; j <= len2; ++j) d[0][j] = j;
    
    for (size_t i = 1; i <= len1; ++i) {
        for (size_t j = 1; j <= len2; ++j) {
            int cost = (str1[i - 1] == str2[j - 1]) ? 0 : 1;
            d[i][j] = std::min({ d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost });
        }
    }
    
    return d[len1][len2];
}

float calculateCER(const std::string& truth, const std::string& prediction) {
    if (truth.empty()) {
        if (prediction.empty()) return 0.0f;
        return 1.0f; // Truth is empty, prediction is not -> 100% error relative
    }
    
    int editDist = calculateEditDistance(truth, prediction);
    float cer = static_cast<float>(editDist) / static_cast<float>(truth.size());
    // Cap at 1.0
    return std::min(1.0f, cer);
}

} // namespace test_metrics
} // namespace rapid_doc
