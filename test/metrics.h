#pragma once

#include <vector>
#include <string>
#include "common/types.h"

namespace rapid_doc {
namespace test_metrics {

// Calculate Intersection over Union (IoU) of two layout boxes
float calculateIoU(const LayoutBox& box1, const LayoutBox& box2);

// Calculate cosine similarity between two feature vectors (tensors)
float calculateCosineSimilarity(const std::vector<float>& vec1, const std::vector<float>& vec2);
float calculateCosineSimilarity(const float* vec1, const float* vec2, size_t size);

// Calculate Levenshtein distance (Edit Distance) between two strings
int calculateEditDistance(const std::string& str1, const std::string& str2);

// Calculate Character Error Rate (CER) or Word Error Rate (WER) based on Edit distance
// Returns [0.0, 1.0], 0 means perfect match (0 edit distance), 1 means completely different
float calculateCER(const std::string& truth, const std::string& prediction);

} // namespace test_metrics
} // namespace rapid_doc
