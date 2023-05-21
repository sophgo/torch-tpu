#ifndef NODECHIP_BATCH_MATMUL_LOCAL_H_
#define NODECHIP_BATCH_MATMUL_LOCAL_H_

#include "tpu_kernel.h"
#include "nodechip_common.h"

#ifdef __cplusplus
extern "C" {
#endif

void nodechip_batch_matmul_float_local(
    local_addr_t L_addr,
    local_addr_t R_addr,
    local_addr_t bias_addr,
    local_addr_t Y_addr,
    local_addr_t ori_buffer_addr, // {1, M, N}, used only for local layer
    const int* L_input_shape,
    const int* R_input_shape,
    data_type_t  LR_dtype,
    data_type_t  Y_dtype,
    int L_trans,
    int R_trans,
    int hdim_is_batch,
    int has_bias,
    bool do_relu,
    float relu_upper_limit,
    int if_global_layer,
    int add_result,
    int left_reuse);

// support bias_dtype
void nodechip_batch_matmul_float_local_v2(
    local_addr_t L_addr,
    local_addr_t R_addr,
    local_addr_t bias_addr,
    local_addr_t Y_addr,
    local_addr_t ori_buffer_addr, // {1, M, N}, used only for local layer
    const int* L_input_shape,
    const int* R_input_shape,
    data_type_t  LR_dtype,
    data_type_t  bias_dtype,
    data_type_t  Y_dtype,
    int L_trans,
    int R_trans,
    int hdim_is_batch,
    int has_bias,
    bool do_relu,
    float relu_upper_limit,
    int if_global_layer,
    int add_result,
    int left_reuse);

void nodechip_batch_matmul_fix8b_local(
    local_addr_t L_addr,
    local_addr_t R_addr,
    local_addr_t bias_addr,
    local_addr_t rzp_addr,
    local_addr_t Y_addr,
    local_addr_t ori_buffer_addr,
    const int* L_input_shape,
    const int* R_input_shape,
    data_type_t  L_dtype,
    data_type_t  R_dtype,
    data_type_t  bias_dtype,
    data_type_t  Y_dtype,
    int L_trans,
    int R_trans,
    int hdim_is_batch,
    int has_bias,
    int rzp_is_const,
    int rzp_const_val,
    int izp_const_val,
    /* requantize params */
    int requant_mode,  // if mode < 0, no requantize
    int scale_val,
    int shift_val,
    int offset_val,
    int if_global_layer,
    int add_result,
    rounding_mode_t round_mode,
    int left_reuse);

void nodechip_batch_matmul_fix8b_local_v2(
    local_addr_t L_addr,
    local_addr_t R_addr,
    local_addr_t bias_addr,
    local_addr_t rzp_addr,
    local_addr_t Y_addr,
    local_addr_t ori_buffer_addr,
    const int* L_input_shape,
    const int* R_input_shape,
    data_type_t  L_dtype,
    data_type_t  R_dtype,
    data_type_t  rzp_dtype,
    data_type_t  bias_dtype,
    data_type_t  Y_dtype,
    int L_trans,
    int R_trans,
    int hdim_is_batch,
    int has_bias,

    int rzp_is_const,
    int rzp_const_val,
    int izp_const_val,

    int add_result,
    int do_relu,

    /* requantize params */
    int requant_mode,  // if mode < 0, no requantize
    int is_perchannel,
    local_addr_t requant_addr,
    int scale_val,
    int shift_val,
    int offset_val,
    rounding_mode_t round_mode,
    int do_sym_saturate,

    int if_global_layer,

    data_layout_t L_layout,
    data_layout_t R_layout,
    data_layout_t Y_layout,
    int left_reuse
    );

static int is_right_n_dim_need_bcast(
    dim4* L_shape,
    dim4* R_shape,
    dim4* Y_shape,
    int L_trans,
    int R_trans
){
  if(!L_trans){
    if (L_shape->n != R_shape->n && 1 == R_shape->n && (Y_shape->n * Y_shape->c <= NPU_NUM))
    {
      L_shape->c *= L_shape->n;
      L_shape->n = 1;
      Y_shape->c *= Y_shape->n;
      Y_shape->n = 1;
      return 1;
    }
  }
  return 0;
}

static int left_ndim_to_hdim(
    local_addr_t L_addr,
    dim4 L_origin_shape,
    dim4 L_shape,
    data_type_t  L_dtype
){
  if(0 == (L_origin_shape.c % tpu_npu_num())){
    // if NPU_NUM is an aliquot part of origin channel, we dont need do anything
    // to change [n,c,h,w] to [1, n*c, h, w] in local layer
    return 0;
  } else {
    // THEN
    int batch_size = L_origin_shape.n;
    dim4 single_shape = {1, L_origin_shape.c, L_origin_shape.h, L_origin_shape.w};
    dim4 L_origin_stride = {0};
    tpu_aligned_stride(&L_origin_stride, 0, &L_origin_shape, L_dtype);
    for (int i_batch = 1; i_batch < batch_size; i_batch++)
    {
      int current_batch_offset = i_batch * L_origin_stride.n *
        tpu_data_type_size(L_dtype);
      int current_accumulated_channels = i_batch * L_origin_shape.c;
      int current_dst_offset = tpu_unified_c_offset(
        current_accumulated_channels, L_origin_stride.c, L_dtype);
      tpu_bdc_cpy_cross_npu(L_addr + current_dst_offset,
        L_addr + current_batch_offset, &single_shape, L_dtype);
    }
  }
  return 0;
}

static int left_hdim_to_ndim(
    local_addr_t L_addr,
    dim4 L_origin_shape,
    dim4 L_shape,
    data_type_t L_dtype
){
  if(0 == (L_origin_shape.c % tpu_npu_num())) {
    return 0;
  }else{
    int batch_size = L_origin_shape.n;
    dim4 single_shape = {1, L_origin_shape.c, L_origin_shape.h, L_origin_shape.w};
    dim4 L_origin_stride = {0};
    tpu_aligned_stride(&L_origin_stride, 0, &L_origin_shape, L_dtype);
    for (int i_batch = batch_size - 1; i_batch > 0; i_batch--)
    {
      int current_batch_offset = i_batch * L_origin_stride.n *
        tpu_data_type_size(L_dtype);
      int current_accumulated_channels = i_batch * L_origin_shape.c;
      int current_dst_offset = tpu_unified_c_offset(
        current_accumulated_channels, L_origin_stride.c, L_dtype);
      tpu_bdc_cpy_cross_npu(L_addr + current_batch_offset,
        L_addr + current_dst_offset, &single_shape, L_dtype);
    }
  }
  return 0;
}

static int y_hdim_to_ndim(
    local_addr_t Y_addr,
    dim4 Y_origin_shape,
    dim4 Y_shape,
    data_type_t Y_dtype
){
  if(0 == (Y_origin_shape.c % tpu_npu_num())) {
    return 0;
  }else{
    int batch_size = Y_origin_shape.n;
    dim4 single_shape = {1, Y_origin_shape.c, Y_origin_shape.h, Y_origin_shape.w};
    dim4 Y_origin_stride = {0};
    tpu_aligned_stride(&Y_origin_stride, 0, &Y_origin_shape, Y_dtype);
    for (int i_batch = batch_size - 1; i_batch > 0; i_batch--)
    {
      int current_batch_offset = i_batch * Y_origin_stride.n *
        tpu_data_type_size(Y_dtype);
      int current_accumulated_channels = i_batch * Y_origin_shape.c;
      int current_dst_offset = tpu_unified_c_offset(
        current_accumulated_channels, Y_origin_stride.c, Y_dtype);
      tpu_bdc_cpy_cross_npu(Y_addr + current_batch_offset,
        Y_addr + current_dst_offset, &single_shape, Y_dtype);
    }
  }
  return 0;
}

#ifdef __cplusplus
}
#endif

#endif
