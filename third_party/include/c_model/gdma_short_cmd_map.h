#ifndef __GDMA_SHORT_CMD_MAP_H__
#define __GDMA_SHORT_CMD_MAP_H__
#include <unordered_map>


static std::unordered_map<u16, std::pair<u16, u16>> DMA_matrix_s2l_map = {
        {1, {1, 1}},
        {2, {2, 1}},
        {3, {3, 1}},
        {4, {4, 1}},
        {5, {5, 4}},
        {9, {9, 20}},
        {32, {32, 4}},
        {36, {36, 3}},
        {39, {39, 1}},
        {40, {40, 3}},
        {64, {64, 20}},
        {192, {96, 32}},
        {256, {128, 32}},
        {288, {160, 32}},
        {416, {192, 16}},
        {432, {208, 16}},
        {448, {224, 16}},
        {464, {240, 16}},
        {496, {256, 16}},
        {512, {288, 32}},
        {544, {320, 8}},
        {576, {352, 32}},
        {608, {384, 8}},
        {704, {448, 32}},
        {736, {480, 32}},
    };

#define MAPPING_DMA_matrix_s2l()  \
do {  \
    for (auto it = DMA_matrix_s2l_map.begin(); it != DMA_matrix_s2l_map.end(); it++)  \
        set_reg_id_val(  \
            lcmd,  \
            {it->first, it->second.second},  \
            get_reg_id_val(cmd, {it->second.first, it->second.second}));  \
} while(0)

static std::unordered_map<u16, std::pair<u16, u16>> DMA_matrix_l2s_map = {
        {1, {1, 1}},
        {2, {2, 1}},
        {3, {3, 1}},
        {4, {4, 1}},
        {5, {5, 4}},
        {9, {9, 20}},
        {32, {32, 4}},
        {36, {36, 3}},
        {39, {39, 1}},
        {40, {40, 3}},
        {64, {64, 20}},
        {320, {96, 32}},
        {128, {128, 32}},
        {160, {160, 32}},
        {480, {192, 16}},
        {496, {208, 16}},
        {384, {224, 16}},
        {400, {240, 16}},
        {432, {256, 16}},
        {512, {288, 32}},
        {544, {320, 8}},
        {576, {352, 32}},
        {608, {384, 8}},
        {704, {448, 32}},
        {736, {480, 32}},
    };

#define MAPPING_DMA_matrix_l2s()  \
do {  \
    for (auto it = DMA_matrix_l2s_map.begin(); it != DMA_matrix_l2s_map.end(); it++)  \
        set_reg_id_val(  \
            lcmd,  \
            {it->first, it->second.second},  \
            get_reg_id_val(cmd, {it->second.first, it->second.second}));  \
} while(0)

static std::unordered_map<u16, std::pair<u16, u16>> DMA_masked_select_map = {
        {1, {1, 1}},
        {2, {2, 1}},
        {3, {3, 1}},
        {4, {4, 1}},
        {5, {5, 4}},
        {9, {9, 20}},
        {32, {32, 4}},
        {36, {36, 3}},
        {39, {39, 1}},
        {40, {40, 3}},
        {43, {43, 3}},
        {64, {64, 20}},
        {384, {96, 16}},
        {400, {112, 16}},
        {416, {128, 16}},
        {432, {144, 16}},
        {448, {160, 16}},
        {464, {176, 16}},
        {480, {192, 16}},
        {496, {208, 16}},
        {512, {224, 32}},
        {544, {256, 8}},
        {576, {288, 32}},
        {608, {320, 8}},
        {640, {352, 32}},
        {672, {384, 32}},
    };

#define MAPPING_DMA_masked_select()  \
do {  \
    for (auto it = DMA_masked_select_map.begin(); it != DMA_masked_select_map.end(); it++)  \
        set_reg_id_val(  \
            lcmd,  \
            {it->first, it->second.second},  \
            get_reg_id_val(cmd, {it->second.first, it->second.second}));  \
} while(0)

static std::unordered_map<u16, std::pair<u16, u16>> DMA_general_map = {
        {1, {1, 1}},
        {2, {2, 1}},
        {3, {3, 1}},
        {4, {4, 1}},
        {5, {5, 4}},
        {9, {9, 20}},
        {32, {32, 4}},
        {36, {36, 3}},
        {39, {39, 1}},
        {40, {40, 3}},
        {64, {64, 20}},
        {96, {96, 32}},
        {160, {128, 32}},
        {464, {160, 16}},
        {512, {192, 32}},
        {544, {224, 8}},
        {576, {256, 32}},
        {608, {288, 8}},
        {704, {320, 32}},
        {736, {352, 32}},
    };

#define MAPPING_DMA_general()  \
do {  \
    for (auto it = DMA_general_map.begin(); it != DMA_general_map.end(); it++)  \
        set_reg_id_val(  \
            lcmd,  \
            {it->first, it->second.second},  \
            get_reg_id_val(cmd, {it->second.first, it->second.second}));  \
} while(0)

static std::unordered_map<u16, std::pair<u16, u16>> DMA_nonzero_map = {
        {1, {1, 1}},
        {2, {2, 1}},
        {3, {3, 1}},
        {4, {4, 1}},
        {5, {5, 4}},
        {9, {9, 20}},
        {32, {32, 4}},
        {36, {36, 3}},
        {39, {39, 1}},
        {40, {40, 3}},
        {43, {43, 3}},
        {64, {64, 20}},
        {256, {96, 32}},
        {384, {128, 16}},
        {400, {144, 16}},
        {416, {160, 16}},
        {432, {176, 16}},
        {512, {192, 32}},
        {544, {224, 8}},
        {576, {256, 32}},
        {608, {288, 8}},
    };

#define MAPPING_DMA_nonzero()  \
do {  \
    for (auto it = DMA_nonzero_map.begin(); it != DMA_nonzero_map.end(); it++)  \
        set_reg_id_val(  \
            lcmd,  \
            {it->first, it->second.second},  \
            get_reg_id_val(cmd, {it->second.first, it->second.second}));  \
} while(0)

#define MAPPING_DMA() \
    do { \
        if (need_map) { \
            u32 cmd_type = get_reg_id_val(cmd, GDMA_ID_CMD_TYPE); \
            if (cmd_type == GDMA_VALUE_TYPE_MATRIX) { \
                MAPPING_DMA_matrix_s2l(); \
                u32 src_start_addr_l32 = (u32)get_reg_id_val(lcmd, (reg_id_t)GDMA_ID_SRC_START_ADDR_L32); \
                u32 src_start_addr_h8 = (u32)get_reg_id_val(lcmd, (reg_id_t)GDMA_ID_SRC_START_ADDR_H8); \
                u64 real_src_addr = (u64)src_start_addr_l32 | ((u64)src_start_addr_h8 << 32); \
                CONTINUOUS_MEM* src_mem = get_continuous_mem(node_idx, real_src_addr); \
                if (src_mem->type == MEM_TYPE_LOCAL) { \
                    MAPPING_DMA_matrix_l2s(); \
                } \
            } else if (cmd_type == GDMA_VALUE_TYPE_MASKED_SEL) { \
                MAPPING_DMA_masked_select(); \
            } else if (cmd_type == GDMA_VALUE_TYPE_GENERAL) { \
                MAPPING_DMA_general(); \
            } else if (cmd_type == GDMA_VALUE_TYPE_NONZERO) { \
                MAPPING_DMA_nonzero(); \
            } else if (cmd_type == GDMA_VALUE_TYPE_SYS) { \
                memcpy(lcmd, cmd, cmd_sz); \
            } else { \
                ASSERT_INFO(0, "no need map for dma type\n"); \
            } \
        } \
    } while (0)

#endif  // __GDMA_SHORT_CMD_MAP_H__