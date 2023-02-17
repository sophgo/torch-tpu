#ifndef NODECHIP_FC_H_
#define NODECHIP_FC_H_

#include "sg_api_struct.h"
#include "tpu_utils.h"
#include "tpu_kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

// fp32/fp16/bf16
void nodechip_fc(
  system_addr_t L_global_addr,
  system_addr_t R_global_addr,
  system_addr_t bias_global_addr,
  system_addr_t Y_global_addr,
  int L_row_num,
  int L_col_num,
  int R_col_num,
  int R_transpose,
  int have_bias,
  data_type_t L_dtype,
  data_type_t R_dtype,
  data_type_t Y_dtype,
  int if_relu,
  float relu_upper_limit);

void general_matmul(
  system_addr_t L_global_addr,
  system_addr_t R_global_addr,
  system_addr_t bias_global_addr,
  system_addr_t Y_global_addr,
  int input_batch,
  int L_row_num,
  int L_col_num,
  int R_col_num,
  int R_transpose,
  int have_bias,
  data_type_t L_dtype,
  data_type_t R_dtype,
  data_type_t Y_dtype,
  int if_relu,
  float relu_upper_limit);

#ifdef __cplusplus
}
#endif

#endif
