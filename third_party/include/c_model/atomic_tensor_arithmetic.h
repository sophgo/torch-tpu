#ifndef ATOMIC_TENSOR_ARITHMETIC_H_
#define ATOMIC_TENSOR_ARITHMETIC_H_

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tensor_arithmetic_param {
    u32 A_addr, B_addr, C_addr, R_addr;
    int N, C, H, W;
    int A_is_const, B_is_const, C_is_const;
    int A_sign, B_sign, C_sign;
    PREC A_prec, B_prec, C_prec, R_prec;
    int A_short_str, B_short_str, C_short_str, R_short_str;
    int b_n_is1, b_h_is1, b_w_is1;
    int opd_num;
    int iter;
    AR_OP op;
    ROUND_MODE round_mode;
    int start_local_mem_idx;
    u64 lane_mask;
} TENSOR_ARITHMETIC_PARAM;

void atomic_tensor_arithmetic(int node_idx, P_COMMAND cmd);

void data_convert(char * A, char * R, TENSOR_ARITHMETIC_PARAM *p_param);
void tensor_calculate_core(char* A, char* B, char* R, TENSOR_ARITHMETIC_PARAM *p_param);
void tensor_calculate_core_fixed_point(char *A, char *B, char *C, char *R, TENSOR_ARITHMETIC_PARAM *p_param);

#ifdef __cplusplus
}
#endif

#endif
