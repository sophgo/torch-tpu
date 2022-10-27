#ifndef ATOMIC_MM_H_
#define ATOMIC_MM_H_

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct
{
  int L_row_num;
  int L_col_num;
  int R_col_num;
  int L_tensor_C;
  int L_tensor_W;
  int R_tensor_C;
  int R_tensor_W;
  int is_L_trans;
  int is_L_const;
  int add_result;
  int AddResLShiftBits;
  int rshiftbits;
  PREC LR_prec;
  PREC Y_prec;
  PREC opd2_prec;
  int L_sign;
  int R_sign;
  int opd2_sign;
  int opd2_enable;
  int opd2_is_const;
  DataUnion opd2_val;
  ROUND_MODE round_mode;
  MM_OP mm_op;
  u32 L_addr;
  u32 R_addr;
  u32 opd2_addr;
  u32 Y_addr;
} MM_PARAM;

typedef struct {
  int L_row_num;
  int L_col_num;
  int R_col_num;
  int is_L_trans;
  int is_R_trans;
  int is_L_const;
  int is_R_const;
  int is_zp_const;
  PREC LR_prec;
  PREC Y_prec;
  int L_sign;
  int R_sign;
  int add_result;
  MM_OP mm_op;
  u32 L_addr;
  u32 R_addr;
  u32 Y_addr;
  u32 zp_addr;
} MM2_PARAM;

void atomic_mm(int node_idx, P_COMMAND cmd);
void atomic_mm2(int node_idx, P_COMMAND cmd);

#ifdef __cplusplus
}
#endif

#endif