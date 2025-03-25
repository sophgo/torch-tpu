#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <optional>
#include <functional>
#include <cstring>

#include <torch/torch.h>
// time
#include <chrono>
#include "net_help.h"

#define TIME_START auto start = std::chrono::high_resolution_clock::now();

#define TIME_END \
    { \
        auto end = std::chrono::high_resolution_clock::now(); \
        std::cout << __func__ << " : " \
                  << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() \
                  << " ms" << std::endl; \
    }


#define TIME_START_NAME(name) auto start_##name = std::chrono::high_resolution_clock::now();

#define TIME_END_NAME(name) \
    { \
        auto end_##name = std::chrono::high_resolution_clock::now(); \
        std::cout << #name << " : " \
        << std::chrono::duration_cast<std::chrono::milliseconds>(end_##name - start_##name).count() \
        << " ms" << std::endl; \
    }
