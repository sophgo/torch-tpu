#pragma once

#include <string.h>
#include <stdio.h>

#define ERROR ""
#define WARNING ""
#define INFO ""

#define SOPHON_LOG(...) \
    printf("[SOPHON-LOG] %s, %d, %s, %s \n", __FILE__, __LINE__, __func__, ##__VA_ARGS__);

#define SOPHON_LOGE(fmt, ...) SOPHON_LOG(ERROR, fmt, ##__VA_ARGS__)

#define SOPHON_LOGW(fmt, ...) SOPHON_LOG(WARNING, fmt, ##__VA_ARGS__)

#define SOPHON_LOGI(fmt, ...) SOPHON_LOG(INFO, fmt, ##__VA_ARGS__)

#define SOPHON_LOGD(fmt, ...) SOPHON_LOG(DEBUG, fmt, ##__VA_ARGS__)