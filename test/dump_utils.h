#pragma once
/**
 * @file dump_utils.h
 * @brief Utilities for dumping cv::Mat as .npy files for cross-language diagnostics.
 *
 * Supports float32 and uint8 matrices. The output follows the NumPy v1.0
 * array format so that files can be loaded directly with numpy.load().
 */

#include <opencv2/core.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <cstring>
#include <stdexcept>

namespace dump_utils {

inline void writeNpyHeader(std::ostream& os,
                           const std::string& dtype_descr,
                           const std::vector<size_t>& shape) {
    std::string shape_str = "(";
    for (size_t i = 0; i < shape.size(); ++i) {
        shape_str += std::to_string(shape[i]);
        if (i + 1 < shape.size() || shape.size() == 1)
            shape_str += ",";
    }
    shape_str += ")";

    std::string dict = "{'descr': '" + dtype_descr +
                       "', 'fortran_order': False, 'shape': " + shape_str + ", }";

    // Header must be padded to a multiple of 64 bytes (including the 10-byte preamble).
    size_t preamble = 10;  // magic(6) + version(2) + HEADER_LEN(2)
    size_t padding = 64 - (preamble + dict.size() + 1) % 64;
    if (padding == 64) padding = 0;
    dict.append(padding, ' ');
    dict += '\n';

    uint16_t headerLen = static_cast<uint16_t>(dict.size());

    char magic[6] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    uint8_t version[2] = {1, 0};

    os.write(magic, 6);
    os.write(reinterpret_cast<const char*>(version), 2);
    os.write(reinterpret_cast<const char*>(&headerLen), 2);
    os.write(dict.data(), dict.size());
}

/**
 * @brief Save a cv::Mat as a .npy file.
 *
 * @param path  Output file path.
 * @param mat   Input matrix (CV_32F or CV_8U, 1-4 channels).
 *
 * The array is written in C-contiguous (row-major) order with shape
 *   (rows, cols)          for single-channel
 *   (rows, cols, channels) for multi-channel
 */
inline void saveMatAsNpy(const std::string& path, const cv::Mat& mat) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open())
        throw std::runtime_error("dump_utils: cannot open " + path);

    std::string dtype;
    int depth = mat.depth();
    if (depth == CV_32F) {
        dtype = "<f4";
    } else if (depth == CV_8U) {
        dtype = "|u1";
    } else {
        throw std::runtime_error("dump_utils: unsupported mat depth " + std::to_string(depth));
    }

    std::vector<size_t> shape;
    shape.push_back(static_cast<size_t>(mat.rows));
    shape.push_back(static_cast<size_t>(mat.cols));
    if (mat.channels() > 1)
        shape.push_back(static_cast<size_t>(mat.channels()));

    writeNpyHeader(ofs, dtype, shape);

    if (mat.isContinuous()) {
        ofs.write(reinterpret_cast<const char*>(mat.data),
                  mat.total() * mat.elemSize());
    } else {
        for (int r = 0; r < mat.rows; ++r) {
            ofs.write(reinterpret_cast<const char*>(mat.ptr(r)),
                      static_cast<std::streamsize>(mat.cols * mat.elemSize()));
        }
    }
}

/**
 * @brief Save a raw float vector as a .npy file with given shape.
 */
inline void saveFloatVecAsNpy(const std::string& path,
                              const std::vector<float>& data,
                              const std::vector<size_t>& shape) {
    std::ofstream ofs(path, std::ios::binary);
    if (!ofs.is_open())
        throw std::runtime_error("dump_utils: cannot open " + path);

    writeNpyHeader(ofs, "<f4", shape);
    ofs.write(reinterpret_cast<const char*>(data.data()),
              data.size() * sizeof(float));
}

} // namespace dump_utils
