#pragma once

#include <string.h>
#include <stdio.h>

#define ERROR ""
#define WARNING ""
#define INFO ""

#define SOPHON_LOG(format, ...) \
    printf("[SOPHON-LOG] %s, %d, %s, ", __FILE__, __LINE__, __func__); \
    printf(format, ##__VA_ARGS__);

#define SOPHON_LOGE(fmt, args...) SOPHON_LOG(ERROR fmt, ##args)

#define SOPHON_LOGW(fmt, args...) SOPHON_LOG(WARNING fmt, ##args)

#define SOPHON_LOGI(fmt, args...) SOPHON_LOG(INFO fmt, ##args)

#define SOPHON_LOGD(fmt, args...) SOPHON_LOG(DEBUG fmt, ##args)