#pragma once

/**
 * @file logger.h
 * @brief Centralized logging macros for RapidDoc C++
 * 
 * Wraps spdlog with project-standard log macros.
 * Log level can be set at compile time via -DLOG_LEVEL=LOG_LEVEL_WARN
 */

#include <spdlog/spdlog.h>

#define LOG_TRACE(...) SPDLOG_TRACE(__VA_ARGS__)
#define LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#define LOG_INFO(...)  SPDLOG_INFO(__VA_ARGS__)
#define LOG_WARN(...)  SPDLOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)
