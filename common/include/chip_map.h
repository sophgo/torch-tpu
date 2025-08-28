#pragma once
#include <unordered_map>
#include <string>

static std::unordered_map<std::string, std::string> CHIP_MAP = {
    {"bm1684x", "tpu_6_0"},
    {"bm1684xe", "tpu_6_0_e"},
    {"bm1688", "tpul_6_0"},
    {"mars3", "tpul_8_1"},
    {"sg2380", "tpul_8_0"},
    {"bm1690", "tpub_7_1"},
    {"sg2260e", "tpub_7_1_e"},
    {"sg2262", "tpub_9_0"},
    {"sg2262rv", "tpub_9_0_rv"},
};
