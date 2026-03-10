/**
 * @file npy_loader.cpp
 * @brief Minimal NumPy .npy format reader implementation
 */

#include "npy_loader.h"

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cstring>

namespace rapid_doc {
namespace test_utils {

size_t NpyArray::elementCount() const {
    size_t count = 1;
    for (auto d : shape) count *= d;
    return count;
}

size_t NpyArray::elementSize() const {
    if (dtype == "<f4" || dtype == "<i4") return 4;
    if (dtype == "|u1" || dtype == "<u1") return 1;
    if (dtype == "<f8") return 8;
    if (dtype == "<i8") return 8;
    return 0;
}

const float* NpyArray::asFloat32() const {
    return reinterpret_cast<const float*>(rawData.data());
}

const uint8_t* NpyArray::asUint8() const {
    return rawData.data();
}

const double* NpyArray::asFloat64() const {
    return reinterpret_cast<const double*>(rawData.data());
}

const int32_t* NpyArray::asInt32() const {
    return reinterpret_cast<const int32_t*>(rawData.data());
}

std::vector<float> NpyArray::toFloatVector() const {
    if (dtype == "<f4") {
        const float* p = asFloat32();
        return std::vector<float>(p, p + elementCount());
    }
    if (dtype == "|u1" || dtype == "<u1") {
        std::vector<float> result(elementCount());
        for (size_t i = 0; i < elementCount(); ++i)
            result[i] = static_cast<float>(rawData[i]);
        return result;
    }
    if (dtype == "<f8") {
        const double* p = asFloat64();
        std::vector<float> result(elementCount());
        for (size_t i = 0; i < elementCount(); ++i)
            result[i] = static_cast<float>(p[i]);
        return result;
    }
    throw std::runtime_error("Cannot convert dtype " + dtype + " to float");
}

std::vector<uint8_t> NpyArray::toUint8Vector() const {
    if (dtype == "|u1" || dtype == "<u1") {
        return std::vector<uint8_t>(rawData.begin(), rawData.begin() + elementCount());
    }
    throw std::runtime_error("Cannot convert dtype " + dtype + " to uint8");
}

// Parse the .npy header dict string, e.g.:
//   {'descr': '<f4', 'fortran_order': False, 'shape': (1, 800, 800, 3), }
static void parseHeader(const std::string& header, NpyArray& arr) {
    // Extract descr
    auto descrPos = header.find("'descr'");
    if (descrPos != std::string::npos) {
        auto q1 = header.find("'", descrPos + 7);
        auto q2 = header.find("'", q1 + 1);
        if (q1 != std::string::npos && q2 != std::string::npos) {
            arr.dtype = header.substr(q1 + 1, q2 - q1 - 1);
        }
    }

    // Extract fortran_order
    arr.fortranOrder = (header.find("True") != std::string::npos &&
                        header.find("fortran_order") != std::string::npos &&
                        header.find("fortran_order': True") != std::string::npos);

    // Extract shape
    auto shapePos = header.find("'shape'");
    if (shapePos != std::string::npos) {
        auto parenStart = header.find("(", shapePos);
        auto parenEnd = header.find(")", parenStart);
        if (parenStart != std::string::npos && parenEnd != std::string::npos) {
            std::string shapeStr = header.substr(parenStart + 1, parenEnd - parenStart - 1);
            arr.shape.clear();
            std::istringstream ss(shapeStr);
            std::string token;
            while (std::getline(ss, token, ',')) {
                token.erase(std::remove(token.begin(), token.end(), ' '), token.end());
                if (!token.empty()) {
                    arr.shape.push_back(std::stoull(token));
                }
            }
        }
    }
}

NpyArray loadNpy(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open .npy file: " + path);
    }

    // Read magic: \x93NUMPY
    char magic[6];
    file.read(magic, 6);
    if (magic[0] != '\x93' || std::string(magic + 1, 5) != "NUMPY") {
        throw std::runtime_error("Invalid .npy magic in: " + path);
    }

    // Read version
    uint8_t major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);

    // Read header length
    uint32_t headerLen = 0;
    if (major == 1) {
        uint16_t len16;
        file.read(reinterpret_cast<char*>(&len16), 2);
        headerLen = len16;
    } else {
        file.read(reinterpret_cast<char*>(&headerLen), 4);
    }

    // Read header string
    std::string header(headerLen, '\0');
    file.read(header.data(), headerLen);

    NpyArray arr;
    parseHeader(header, arr);

    // Read data
    size_t dataSize = arr.elementCount() * arr.elementSize();
    arr.rawData.resize(dataSize);
    file.read(reinterpret_cast<char*>(arr.rawData.data()), dataSize);

    if (!file) {
        throw std::runtime_error("Failed to read data from .npy file: " + path);
    }

    return arr;
}

std::string loadJsonString(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open JSON file: " + path);
    }
    std::ostringstream ss;
    ss << file.rdbuf();
    return ss.str();
}

} // namespace test_utils
} // namespace rapid_doc
