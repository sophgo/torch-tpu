#ifndef  GET_ATOMIC_PROFILE_H
#define  GET_ATOMIC_PROFILE_H

#include "common.h"

#ifdef __cplusplus
extern "C" {
#endif

INST_PROFILE atomic_fused_linear_get_profile(
  int res_n,
  int res_c,
  int res_h,
  int res_w,
  PREC prec,
  u32 res_addr);

INST_PROFILE atomic_fused_cmp_get_profile(
  int res_n,
  int res_c,
  int res_h,
  int res_w,
  PREC prec,
  u32 res_addr);

INST_PROFILE atomic_sfu_get_profile(
  int res_n,
  int res_c,
  int res_h,
  int res_w,
  SFU_OP op,
  PREC prec,
  int tailor_n,
  u32 res_addr);

INST_PROFILE atomic_mm_get_profile(
  int res_n,
  int res_c,
  int res_w,
  int opd1_w,
  int opd0_c,
  int opd0_w,
  int opd0_n,
  PREC prec,
  u32 res_addr,
  int have_bias,
  int have_left_tran,
  int add_result);

INST_PROFILE atomic_mm2_get_profile(
  int res_c,
  int res_w,
  int opd1_c,
  int opd1_w,
  MM_OP mm_op,
  PREC prec,
  int add_result);

INST_PROFILE atomic_vec_corr_get_profile(
  int res_n,
  int res_c,
  int res_h,
  int res_w,
  u32 res_addr,
  PREC A_prec,
  PREC B_prec,
  PREC R_prec,
  AR_OP vec_corr_op);

INST_PROFILE atomic_ar_get_profile(
  int res_n,
  int res_c,
  int res_h,
  int res_w,
  int res_h_stride,
  int res_w_stride,
  int res_short_str,
  int opd0_h_stride,
  int opd0_w_stride,
  int opd0_short_str,
  int opd1_h_stride,
  int opd1_w_stride,
  int opd1_short_str,
  u32 opd0_addr,
  u32 opd1_addr,
  u32 res_addr,
  PREC opd0_prec,
  PREC opd1_prec,
  PREC res_prec,
  int div_iter,
  AR_OP ar_op,
  int opd0_is_const,
  int opd1_is_const);

INST_PROFILE atomic_pool_depthwise_get_profile(
  int res_n,
  int res_c,
  int res_h,
  int res_w,
  int opd1_h,
  int opd1_w,
  int stride_h,
  int stride_w,
  PD_OP op,
  u32 res_addr,
  PREC prec);

INST_PROFILE atomic_rqdq_get_profile(
  u32 A_addr,
  u32 B_addr,
  u32 R_addr,
  int N,
  int C,
  int H,
  int W,
  bool B_is_const);

INST_PROFILE atomic_cw_trans_get_profile(
  int res_n,
  int res_c,
  int res_h,
  int res_w,
  PREC prec,
  TRAN_OP tran_op);

INST_PROFILE atomic_conv_get_profile(
  int res_n,
  int res_c,
  int res_h,
  int res_w,
  int opd0_c,
  int opd1_h,
  int opd1_w,
  PREC prec);

INST_PROFILE atomic_sg_get_profile(
  int res_n,
  int res_c,
  int opd1_w,
  u32 opd3_addr,
  SG_OP op,
  PREC prec,
  u32 res_addr);

INST_PROFILE atomic_sgl_get_profile(
  int res_n,
  int res_c,
  int res_w,
  int opd1_h,
  SG_OP op,
  PREC prec,
  u32 res_addr);

INST_PROFILE atomic_lane_bc_get_profile(
  int n,
  int src_c,
  int dst_c,
  int h,
  int w,
  PREC prec,
  u32 src_addr,
  u32 dst_addr);

INST_PROFILE atomic_static_bc_get_profile(
  int dst_c,
  int w,
  PREC prec);

INST_PROFILE atomic_dis_bc_get_profile(int c, PREC prec);

INST_PROFILE atomic_gdma_get_profile(
  int src_n,
  int src_c,
  int src_h,
  int src_w,
  int dst_n,
  int dst_c,
  int dst_h,
  int dst_w,
  u64 g_addr,
  int src_data_format,
  int dst_data_format,
  int result_add,
  int direction,
  int transpose,
  bool is_cwtrans,
  bool is_const_fill,
  int src_wstride,
  int dst_wstride,
  int src_hstride,
  int dst_hstride,
  int src_cstride,
  int dst_cstride,
  int src_nstride,
  int dst_nstride,
  int stride_enable,
  int dma_tpye);

INST_PROFILE atomic_gdma_tensor_get_profile(
  int src_n,
  int src_c,
  int src_h,
  int src_w,
  u64 src_addr,
  u64 dst_addr,
  int data_format,
  int direction,
  int special_func,
  int store_type,
  int src_wstride,
  int dst_wstride,
  int src_hstride,
  int dst_hstride,
  int src_cstride,
  int dst_cstride,
  int src_nstride,
  int dst_nstride,
  int stride_enable);


INST_PROFILE atomic_gdma_matrix_get_profile(
  int row_num,
  int col_num,
  int sec_size,
  u64 src_addr,
  u64 dst_addr,
  int data_format,
  int direction,
  int transpose,
  int global_row_stride,
  int local_row_stride,
  int local_sec_stride,
  int stride_enable);

INST_PROFILE atomic_gdma_constant_get_profile(
  int dst_n,
  int dst_c,
  int dst_h,
  int dst_w,
  u64 src_addr,
  u64 dst_addr,
  int data_format,
  int special_func,
  int is_local_dst,
  int dst_wstride,
  int dst_hstride,
  int dst_cstride,
  int dst_nstride,
  int stride_enable);

INST_PROFILE atomic_gdma_general_get_profile(
  int src_count,
  u64 src_addr,
  u64 dst_addr,
  int data_format,
  int src_is_const);

INST_PROFILE atomic_gdma_cw_trans_get_profile(
  int src_n,
  int src_c,
  int src_h,
  int src_w,
  u64 src_addr,
  u64 dst_addr,
  int data_format,
  int direction,
  int src_wstride,
  int dst_wstride,
  int src_hstride,
  int dst_hstride,
  int src_cstride,
  int dst_cstride,
  int src_nstride,
  int dst_nstride,
  int stride_enable);

INST_PROFILE atomic_gdma_masked_sel_get_profile(
  int src_n,
  int src_c,
  int src_h,
  int src_w,
  u64 src_addr,
  u64 dst_addr,
  int data_format,
  int direction);

INST_PROFILE atomic_gdma_nonzero_get_profile(
  int src_n,
  int src_c,
  int src_h,
  int src_w,
  u64 src_addr,
  u64 dst_addr,
  int src_data_format,
  int dst_data_format,
  int direction);

INST_PROFILE atomic_gdma_broadcast_get_profile(
  int src_n,
  int src_h,
  int src_w,
  int dst_c,
  u64 src_addr,
  u64 dst_addr,
  int data_format,
  int direction,
  int src_wstride,
  int dst_wstride,
  int src_hstride,
  int dst_hstride,
  int src_cstride,
  int dst_cstride,
  int src_nstride,
  int dst_nstride,
  int stride_enable);

INST_PROFILE atomic_gdma_gather_get_profile(
  int src_c,
  int src_h,
  int src_w,
  int index_H,
  u64 src_addr,
  u64 dst_addr,
  int data_format,
  int direction,
  int src_wstride,
  int dst_wstride,
  int src_hstride,
  int dst_hstride,
  int src_cstride,
  int dst_cstride,
  int src_nstride,
  int dst_nstride,
  int stride_enable);

INST_PROFILE atomic_gdma_scatter_get_profile(
  int src_c,
  int src_h,
  int src_w,
  int dst_h,
  u64 src_addr,
  u64 dst_addr,
  int data_format,
  int direction,
  int src_C_is1,
  int index_C_is1,
  int src_wstride,
  int dst_wstride,
  int src_hstride,
  int dst_hstride,
  int src_cstride,
  int dst_cstride,
  int src_nstride,
  int dst_nstride,
  int stride_enable);

#ifdef __cplusplus
}
#endif

#endif /* GET_ATOMIC_PROFILE_H */
















