#pragma once

#include <string.h>
#include <stdio.h>

#define ERROR_PREFIX "[ERROR]"
#define WARNING_PREFIX "[WARNING]"
#define INFO_PREFIX "[INFO]"
#define DEBUG_PREFIX "[Debug]"

#define SOPHON_LOG(format, ...) \
    printf("[SOPHON-LOG] %s, %d, %s, ", __FILE__, __LINE__, __func__); \
    printf(format, ##__VA_ARGS__);

#define SOPHON_LOGE(fmt, args...) SOPHON_LOG(ERROR_PREFIX fmt, ##args)

#define SOPHON_LOGW(fmt, args...) SOPHON_LOG(WARNING_PREFIX fmt, ##args)

#define SOPHON_LOGI(fmt, args...) SOPHON_LOG(INFO_PREFIX fmt, ##args)

#define SOPHON_LOGD(fmt, args...) SOPHON_LOG(DEBUG_PREFIX fmt, ##args)