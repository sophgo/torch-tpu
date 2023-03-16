#ifndef NODECHIP_BATCH_MATMUL_H_
#define NODECHIP_BATCH_MATMUL_H_

#include "nodechip_fc.h"

#ifdef __cplusplus
extern "C" {
#endif

// interface for fp32/fp16/bfp16
void nodechip_batch_matmul_float(
  global_addr_t L_global_addr,
  global_addr_t R_global_addr,
  global_addr_t bias_global_addr,
  global_addr_t Y_global_addr,
  data_type_t in_dtype,
  data_type_t out_dtype,
  const int* L_shape,
  const int* R_shape,
  int L_dim,
  int R_dim,
  int* Y_shape,
  int* Y_dim,
  int L_trans,
  int R_trans,
  int hdim_is_batch,
  int has_bias,
  bool do_relu,
  float relu_upper_limit);

void nodechip_batch_matmul_fix8b(
  global_addr_t L_global_addr,
  global_addr_t R_global_addr,
  global_addr_t bias_global_addr,
  global_addr_t zp_global_addr,
  global_addr_t Y_global_addr,
  data_type_t L_dtype,
  data_type_t R_dtype,
  data_type_t bias_dtype,
  data_type_t Y_dtype,
  const int* L_shape,
  const int* R_shape,
  int L_dim,
  int R_dim,
  int* Y_shape,
  int* Y_dim,
  int L_trans,
  int R_trans,
  int hdim_is_batch,
  int has_bias,
  int zp_is_const, // if false, perchannel R_zp
  int zp_const_val,
  int izp_const_val,
  /* requantize params */
  int requant_mode,  // if mode < 0, no requantize
  int scale_val,
  int shift_val,
  int offset_val,
  rounding_mode_t round_mode);

typedef struct {
  global_addr_t L_addr;
  global_addr_t R_addr;
  global_addr_t bias_addr;
  global_addr_t rzp_addr;
  global_addr_t Y_addr;
  int* L_shape;
  int* R_shape;
  data_type_t L_dtype;
  data_type_t R_dtype;
  data_type_t bias_dtype;
  data_type_t Y_dtype;
  bool L_trans;
  bool R_trans;
  bool hdim_is_batch;
  bool has_bias;
  bool do_relu;
  float relu_upper_limit;
  /* fix8b param */
  bool has_zp;
  bool rzp_is_const;
  int rzp_const_val;
  int izp_const_val;
  /* requantize param */
  int requant_mode;
  int scale_val;
  int shift_val;
  int offset_val;
  rounding_mode_t round_mode;
} mmII_param_t;

void general_matmul_II(const mmII_param_t* param);

#ifdef __cplusplus
}

#endif

#endif
