#pragma once

/**
 * @file npy_loader.h
 * @brief Minimal NumPy .npy format reader for C++ test fixtures
 *
 * Supports loading float32 and uint8 arrays saved by numpy.save().
 * Reference: https://numpy.org/devdocs/reference/generated/numpy.lib.format.html
 */

#include <string>
#include <vector>
#include <cstdint>

namespace rapid_doc {
namespace test_utils {

struct NpyArray {
    std::vector<uint8_t> rawData;
    std::vector<size_t> shape;
    std::string dtype;  // "<f4" (float32), "|u1" (uint8), "<f8" (float64), "<i4" (int32)
    bool fortranOrder = false;

    size_t elementCount() const;
    size_t elementSize() const;

    // Typed accessors (data must match dtype)
    const float* asFloat32() const;
    const uint8_t* asUint8() const;
    const double* asFloat64() const;
    const int32_t* asInt32() const;

    std::vector<float> toFloatVector() const;
    std::vector<uint8_t> toUint8Vector() const;
};

/**
 * @brief Load a .npy file
 * @param path File path
 * @return NpyArray with data and metadata
 * @throws std::runtime_error if file cannot be read or format is invalid
 */
NpyArray loadNpy(const std::string& path);

/**
 * @brief Load a JSON file as a string
 */
std::string loadJsonString(const std::string& path);

} // namespace test_utils
} // namespace rapid_doc
