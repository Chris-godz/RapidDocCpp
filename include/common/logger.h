#pragma once

/**
 * @file logger.h
 * @brief Centralized logging macros for RapidDoc C++
 * 
 * Wraps spdlog with project-standard log macros.
 * Log level can be set at compile time via -DLOG_LEVEL=LOG_LEVEL_WARN
 */

#include <spdlog/spdlog.h>

/* Only define if not already defined (e.g. by DXNN-OCR-cpp logger.hpp) to avoid redefinition warnings. */
#ifndef LOG_TRACE
#define LOG_TRACE(...) SPDLOG_TRACE(__VA_ARGS__)
#endif
#ifndef LOG_DEBUG
#define LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#endif
#ifndef LOG_INFO
#define LOG_INFO(...)  SPDLOG_INFO(__VA_ARGS__)
#endif
#ifndef LOG_WARN
#define LOG_WARN(...)  SPDLOG_WARN(__VA_ARGS__)
#endif
#ifndef LOG_ERROR
#define LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)
#endif
