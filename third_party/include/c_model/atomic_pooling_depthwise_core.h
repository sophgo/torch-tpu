#ifndef ATOMIC_POOLING_DEPTHWISE_CORE_H
#define ATOMIC_POOLING_DEPTHWISE_CORE_H

#include "common.h"
#include "sg_fp16.h"
#include "cmodel_common.h"

template <typename T>
void roi_img2col(
    T *img_col, T *img,
    unsigned short *roi, T imm_val,
    int input_h, int input_w, int output_w,
    int kh, int kw, int *coordinate) {
  for (int nr_idx = 0; nr_idx < output_w; ++nr_idx) {
    int offset = nr_idx * 4;
    int img_h_up = roi[offset + 1];
    int img_w_lf = roi[offset];
    int img_h_dn = roi[offset + 3];
    int img_w_rt = roi[offset + 2];
    // ASSERT_INFO((img_h_up <= img_h_dn && img_w_lf <= img_w_rt &&
    //              img_h_up >= 0 && img_w_lf >= 0 &&
    //              img_h_dn < input_h && img_w_rt < input_w),
    //             "img_h_up=%d, img_h_dn=%d, img_w_lf=%d, img_w_rt=%d, input_h=%d, input_w=%d\n",
    //             img_h_up, img_h_dn, img_w_lf, img_w_rt, input_h, input_w);
    // ASSERT_INFO(((img_h_up + kh > img_h_dn) && (img_w_lf + kw > img_w_rt)),
    //             "img_h_up=%d, img_h_dn=%d, img_w_lf=%d, img_w_rt=%d, kh=%d, kw=%d\n",
    //             img_h_up, img_h_dn, img_w_lf, img_w_rt, kh, kw);

    for (int kh_idx = 0; kh_idx < kh; ++kh_idx) {
      for (int kw_idx = 0; kw_idx < kw; ++kw_idx) {
        int img_h_idx = img_h_up + kh_idx;
        int img_w_idx = img_w_lf + kw_idx;
        int dst_idx = nr_idx * kh * kw +
                      kh_idx * kw +
                      kw_idx;
        if (img_h_idx > img_h_dn || img_w_idx > img_w_rt) {
          img_col[dst_idx] = imm_val;
          if (coordinate) {
            coordinate[3 * dst_idx + 0] = -1;
            coordinate[3 * dst_idx + 1] = -1;
            coordinate[3 * dst_idx + 2] = -1;
          }
        }
        else {
          img_col[dst_idx] = img[img_h_idx * input_w + img_w_idx];
          if (coordinate) {
            coordinate[3 * dst_idx + 0] = 0;
            coordinate[3 * dst_idx + 1] = img_h_idx;
            coordinate[3 * dst_idx + 2] = img_w_idx;
          }
        }
      }
    }
  }
}

void pooling_core_float(
    void *input_col, void *output,
    DataUnion kernel_data,
    int output_h, int output_w,
    int kh, int kw,
    PREC prec, bool is_avg_pooling);

void pooling_core_quant(
    signed char *input_col,
    void *output,
    DataUnion kernel_data,
    PREC output_prec,
    int output_h, int output_w,
    int kh, int kw,
    int rshift_num,
    bool input_sign,
    bool kernel_sign,
    ROUND_MODE round_mode,
    bool is_avg_pooling,
    bool is_roi_op);

void depthwise_core_quant(
    signed char *input_col, int *weight,
    void* output, bool has_bias, int bias,
    int output_h, int output_w,
    int kh, int kw,
    int rshift_num,
    ROUND_MODE round_mode,
    PREC output_prec,
    bool input_sign, bool kernel_sign,
    bool bias_sign, bool if_relu,
    bool is_roi_op);

void depthwise_core_float(
    void *input_col, void *weight,
    void *output, bool has_bias, void *bias,
    int output_h, int output_w,
    int kh, int kw,
    PREC input_prec);

#ifdef SG_TV_GEN
void pooling_core_float_with_log(
    void *input_col, void *output,
    DataUnion kernel_data,
    int output_h, int output_w,
    int kh, int kw,
    PREC prec, bool is_avg_pooling,
    bool is_roi_op,
    int on_idx, int oc_idx, int *coordinate,
    u32 input_addr, u32 output_addr, unsigned short *roi,
    int input_c, int input_h, int input_w);

void pooling_core_quant_with_log(
    signed char *input_col,
    void *output,
    DataUnion kernel_data,
    PREC output_prec,
    int output_h, int output_w,
    int kh, int kw,
    int rshift_num,
    bool input_sign,
    bool kernel_sign,
    ROUND_MODE round_mode,
    bool is_avg_pooling,
    bool is_roi_op,
    int on_idx, int oc_idx, int *coordinate,
    u32 input_addr, u32 output_addr, unsigned short *roi,
    int input_c, int input_h, int input_w);

void depthwise_core_quant_with_log(
    signed char *input_col, int *weight,
    void* output, bool has_bias, int bias,
    int output_h, int output_w,
    int kh, int kw,
    int rshift_num,
    ROUND_MODE round_mode,
    PREC output_prec,
    bool input_sign, bool kernel_sign,
    bool bias_sign, bool if_relu,
    bool is_roi_op,
    int on_idx, int oc_idx, int *coordinate,
    u32 input_addr, u32 weight_addr, u32 bias_addr,
    u32 output_addr, unsigned short *roi,
    int input_c, int input_h, int input_w,
    bool kernel_rotate, bool kernel_is_const, bool bias_is_const);

void depthwise_core_float_with_log(
    void *input_col, void *weight,
    void *output, bool has_bias, void *bias,
    int output_h, int output_w,
    int kh, int kw,
    PREC input_prec, bool is_roi_op,
    int on_idx, int oc_idx, int *coordinate,
    u32 input_addr, u32 weight_addr, u32 bias_addr,
    u32 output_addr, unsigned short *roi,
    int input_c, int input_h, int input_w,
    bool kernel_rotate, bool kernel_is_const, bool bias_is_const);
#endif

#endif /* ATOMIC_POOLING_DEPTHWISE_CORE_H */
