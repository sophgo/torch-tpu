#ifndef ATOMIC_UTILS_H_
#define ATOMIC_UTILS_H_

#include "common.h"
#include "sg_fp16.h"


// #define DEBUG_IMG2COL

template<typename T>
T RightShiftRound(T src, int shift_num, ROUND_MODE round_mode)
{
  if (shift_num == 0) return src;
  if (shift_num > 63) shift_num = 63;
  T val, res;
  val = src >> shift_num;
  res = val;
  T lo_mask = (1ull << shift_num) - 1;
  T mant = src & lo_mask;
  T mant_0d5 = 1ull << (shift_num - 1);
  if (round_mode == ROUND_HALF_TO_EVEN) {
    if (mant == mant_0d5) {
      res = val + (val & 1);
    } else if (mant > mant_0d5) {
      res = val + 1;
    }
  } else if (round_mode == ROUND_HALF_AWAY_FROM_ZERO) {
    if (src >= 0 && mant >= mant_0d5) {
      res = val + 1;
    } else if (src < 0 && mant > mant_0d5) {
      res = val + 1;
    }
  } else if (round_mode == ROUND_TOWARDS_ZERO) {
    if (src < 0) res = val + (mant != 0);
  } else if (round_mode == ROUND_DOWN) {
    res = val;
  } else if (round_mode == ROUND_UP) {
    res = val + (mant != 0);
  } else if (round_mode == ROUND_HALF_UP) {
    if (mant >= mant_0d5) res = val + 1;
  } else if (round_mode == ROUND_HALF_DOWN) {
    if (mant > mant_0d5) res = val + 1;
  }
  return res;
}

// saturate INT32 to INT8/INT16
inline static void saturate_to(int* src_ptr, void* dst_ptr, int len, PREC prec, int sign_unsign) {
  int satu_max = (prec == INT8) ? (sign_unsign ? 127 : 255) : (sign_unsign ? 32767 : 65535);
  int satu_min = (prec == INT8) ? (sign_unsign ? -128 : 0) : (sign_unsign ? -32768 : 0);
  for (int i = 0; i < len; i++) {
    int temp = src_ptr[i] > satu_max ? satu_max : src_ptr[i];
    temp = temp < satu_min ? satu_min : temp;
    memcpy((u8*)dst_ptr + i * get_bytesize(prec), &temp, get_bytesize(prec));
  }
}

#define ADDTREE_NUM (32)
float float_addtree(float *data, int num);

inline static float from_binary16(fp16 data) {
    return fp16_to_fp32(data).fval;
}

inline static float from_binary16(bf16 data) {
    return bf16_to_fp32(data).fval;
}

template<typename T>
void kernel_rotate(T *weight, int kh, int kw) {
    int swap_count = 0;
    for (int hidx = 0; hidx < kh; hidx++) {
        for (int widx = 0; widx < kw; widx++) {
            int idx = hidx * kw + widx;
            if (swap_count < (kh * kw / 2)) {
                int idx_swap = (kh - hidx - 1) * kw + kw - widx - 1;
                T temp = weight[idx];
                weight[idx] = weight[idx_swap];
                weight[idx_swap] = temp;
                swap_count++;
            } else {
              return;
            }
        }
    }
}

template<typename T>
void img2col(
        T* img, int input_c, int input_h, int input_w,
        int pad_h_t, int pad_h_b, int pad_w_l, int pad_w_r,
        int ins_h, int ins_w, T* pad_ins,
        int stride_h, int stride_w, int dh, int dw,
        int kh, int kw, PAD_MODE pad_mode,
        T *img_col, int *coordinate) {
#ifdef DEBUG_IMG2COL
    static int call_count = 0;
    call_count++;
    FILE* img2col_file = NULL;
    if (call_count == 1) {
        img2col_file = fopen("img2col.data", "w");
        fprintf(img2col_file, "img(%d * %d * %d)\n", input_c, input_h, input_w);
        for (int ic = 0; ic < input_c; ic++) {
            fprintf(img2col_file, "%d\n", ic);
            for (int ih = 0; ih < input_h; ih++) {
                for (int iw = 0; iw < input_w; iw++) {
                    fprintf(img2col_file, "%d  ", img[(ic * input_h + ih) * input_w + iw]);
                }
                fprintf(img2col_file, "\n");
            }
        }
        fprintf(img2col_file, "ph(%d * %d) pw(%d * %d) ins(%d * %d) pad_mode(%d)\n",
            pad_h_t, pad_h_b, pad_w_l, pad_w_r, ins_h, ins_w, pad_mode);
        fprintf(img2col_file, "kernel(%d * %d) stride(%d * %d) dilation(%d * %d)\n",
            kh, kw, stride_h, stride_w, dh, dw);
    }
#endif
    int kh_ext = (kh - 1) * dh + 1;
    int kw_ext = (kw - 1) * dw + 1;
    int ih_ext = pad_h_t + pad_h_b + (input_h - 1) * (ins_h + 1) + 1;
    int iw_ext = pad_w_l + pad_w_r + (input_w - 1) * (ins_w + 1) + 1;
    int oh = (ih_ext - kh_ext) / stride_h + 1;
    int ow = (iw_ext - kw_ext) / stride_w + 1;
    int conv_mac_size = input_c * kh * kw;
    for (int cidx = 0; cidx < input_c; cidx++) {
        for (int ohidx = 0; ohidx < oh; ohidx++) {
            for (int owidx = 0; owidx < ow; owidx++) {
                for (int khidx = 0; khidx < kh; khidx++) {
                    for (int kwidx = 0; kwidx < kw; kwidx++) {
                        int ih_pos = ohidx * stride_h + khidx * dh - pad_h_t;
                        int iw_pos = owidx * stride_w + kwidx * dw - pad_w_l;
                        bool pos_ins = ih_pos % (ins_h + 1) > 0 ||
                                       iw_pos % (ins_w + 1) > 0;
                        bool pos_pad = ih_pos < 0 || ih_pos >= (ih_ext - pad_h_b - pad_h_t) ||
                                       iw_pos < 0 || iw_pos >= (iw_ext - pad_w_l - pad_w_r);
                        u64 dst_idx = (u64)ohidx * (ow * conv_mac_size) +
                                      owidx * conv_mac_size +
                                      cidx * kh * kw +
                                      khidx * kw +
                                      kwidx;
                        s64 src_idx = ((s64)cidx * input_h + ih_pos / (ins_h + 1)) * input_w +
                                      iw_pos / (ins_w + 1);
                        if (ih_pos >= 0 && ih_pos / (ins_h + 1) < input_h &&
                                iw_pos >= 0 && iw_pos / (ins_w + 1) < input_w) {
                            img_col[dst_idx] = img[src_idx];
                            if (coordinate) {
                                coordinate[dst_idx * 3 + 0] = cidx;
                                coordinate[dst_idx * 3 + 1] = ih_pos / (ins_h + 1);
                                coordinate[dst_idx * 3 + 2] = iw_pos / (ins_w + 1);
                            }
                        }
                        if (pos_ins) {
                            img_col[dst_idx] = pad_ins[cidx * 2 + 1];
                            if (coordinate) {
                                coordinate[dst_idx * 3 + 0] = -2;
                                coordinate[dst_idx * 3 + 1] = -2;
                                coordinate[dst_idx * 3 + 2] = -2;
                            }
                        }
                        if (pos_pad) {
                            if (pad_mode == PAD_CONSTANT) {
                                img_col[dst_idx] = pad_ins[cidx * 2];
                                if (coordinate) {
                                    coordinate[dst_idx * 3 + 0] = -1;
                                    coordinate[dst_idx * 3 + 1] = -1;
                                    coordinate[dst_idx * 3 + 2] = -1;
                                }
                            } else if (pad_mode == PAD_REFLECTION) {
                                ih_pos = ih_pos < 0 ? -ih_pos : ih_pos;
                                ih_pos = ih_pos >= (ih_ext - pad_h_b - pad_h_t) ?
                                         2 * (ih_ext - pad_h_t - pad_h_b - 1) - ih_pos : ih_pos;
                                iw_pos = iw_pos < 0 ? -iw_pos : iw_pos;
                                iw_pos = iw_pos >= (iw_ext - pad_w_l - pad_w_r) ?
                                         2 * (iw_ext - pad_w_l - pad_w_r - 1) - iw_pos : iw_pos;
                                src_idx = (cidx * input_h + ih_pos / (ins_h + 1)) * input_w +
                                          iw_pos / (ins_w + 1);
                                bool ins = ih_pos % (ins_h + 1) || iw_pos % (ins_w + 1);
                                img_col[dst_idx] = ins ? pad_ins[cidx * 2 + 1] : img[src_idx];
                                if (coordinate) {
                                    if (ins) {
                                        coordinate[dst_idx * 3 + 0] = -3;
                                        coordinate[dst_idx * 3 + 1] = -3;
                                        coordinate[dst_idx * 3 + 2] = -3;
                                    } else {
                                        coordinate[dst_idx * 3 + 0] = cidx;
                                        coordinate[dst_idx * 3 + 1] = ih_pos / (ins_h + 1);
                                        coordinate[dst_idx * 3 + 2] = iw_pos / (ins_w + 1);
                                    }
                                }
                            } else if (pad_mode == PAD_REPLICATION) {
                                ih_pos = ih_pos < 0 ? 0 : ih_pos;
                                ih_pos = ih_pos >= (ih_ext - pad_h_b - pad_h_t) ?
                                         ih_ext - pad_h_b - pad_h_t - 1 : ih_pos;
                                iw_pos = iw_pos < 0 ? 0 : iw_pos;
                                iw_pos = iw_pos >= (iw_ext - pad_w_l - pad_w_r) ?
                                         iw_ext - pad_w_l - pad_w_r - 1 : iw_pos;
                                src_idx = (cidx * input_h + ih_pos / (ins_h + 1)) * input_w +
                                          iw_pos / (ins_w + 1);
                                bool ins = ih_pos % (ins_h + 1) || iw_pos % (ins_w + 1);
                                img_col[dst_idx] = ins ? pad_ins[cidx * 2 + 1] : img[src_idx];
                                if (coordinate) {
                                    if (ins) {
                                        coordinate[dst_idx * 3 + 0] = -3;
                                        coordinate[dst_idx * 3 + 1] = -3;
                                        coordinate[dst_idx * 3 + 2] = -3;
                                    } else {
                                        coordinate[dst_idx * 3 + 0] = cidx;
                                        coordinate[dst_idx * 3 + 1] = ih_pos / (ins_h + 1);
                                        coordinate[dst_idx * 3 + 2] = iw_pos / (ins_w + 1);
                                    }
                                }
                            } else if (pad_mode == PAD_CIRCULAR) {
                                ih_pos = ih_pos < 0 ? ih_ext - pad_h_t - pad_h_b + ih_pos : ih_pos;
                                ih_pos = ih_pos >= (ih_ext - pad_h_b - pad_h_t) ?
                                         ih_pos - (ih_ext - pad_h_t - pad_h_b) : ih_pos;
                                iw_pos = iw_pos < 0 ? iw_ext - pad_w_l - pad_w_r + iw_pos : iw_pos;
                                iw_pos = iw_pos >= (iw_ext - pad_w_l - pad_w_r) ?
                                         iw_pos - (iw_ext - pad_w_l - pad_w_r) : iw_pos;
                                src_idx = (cidx * input_h + ih_pos / (ins_h + 1)) * input_w +
                                          iw_pos / (ins_w + 1);
                                bool ins = ih_pos % (ins_h + 1) || iw_pos % (ins_w + 1);
                                img_col[dst_idx] = ins ? pad_ins[cidx * 2 + 1] : img[src_idx];
                                if (coordinate) {
                                    if (ins) {
                                        coordinate[dst_idx * 3 + 0] = -3;
                                        coordinate[dst_idx * 3 + 1] = -3;
                                        coordinate[dst_idx * 3 + 2] = -3;
                                    } else {
                                        coordinate[dst_idx * 3 + 0] = cidx;
                                        coordinate[dst_idx * 3 + 1] = ih_pos / (ins_h + 1);
                                        coordinate[dst_idx * 3 + 2] = iw_pos / (ins_w + 1);
                                    }
                                }
                            } else {
                                ASSERT(0);
                            }
                        }
                    }
                }
            }
        }
    }
#ifdef DEBUG_IMG2COL
    if (call_count == 1) {
        fprintf(img2col_file, "col_buf(%d * %d * %d)\n", oh, ow, conv_mac_size);
        for (int h = 0; h < oh; h++) {
            for (int w = 0; w < ow; w++) {
                for (int mac = 0; mac < conv_mac_size; mac++) {
                    fprintf(img2col_file, "%d  ", img_col[(h * ow + w) * conv_mac_size + mac]);
                }
                fprintf(img2col_file, "\n");
            }
        }
        fclose(img2col_file);
    }
#endif
}

template void img2col<signed char>(
        signed char* img,
        int input_c, int input_h, int input_w,
        int pad_h_t, int pad_h_b, int pad_w_l, int pad_w_r,
        int ins_h, int ins_w, signed char* pad_ins,
        int stride_h, int stride_w, int dh, int dw,
        int kh, int kw, PAD_MODE pad_mode,
        signed char *img_col, int* coordinate);

template void img2col<float>(
        float* img,
        int input_c, int input_h, int input_w,
        int pad_h_t, int pad_h_b, int pad_w_l, int pad_w_r,
        int ins_h, int ins_w, float* pad_ins,
        int stride_h, int stride_w, int dh, int dw,
        int kh, int kw, PAD_MODE pad_mode,
        float *img_col, int* coordinate);

void conv_core_quant(
        signed char *input_col, int *weight,
        void *output, short * bias, PREC out_prec,
        int input_c, int output_c, int oh, int ow,
        int kh, int kw, int with_bias, int result_add,
        int bReLU_EN, int r_shift_m,
        int opd0_sign,
        int opd2_sign, int out_sign);

void conv_core_float(
        void *input_col, void *weight,
        void *output, float *bias,
        int input_c, int output_c, int oh, int ow,
        int kh, int kw, int result_add,
        PREC in_prec, PREC out_prec,
        float *add_tree_res = NULL);

fp32 microp_nr(fp32 x, int iter);
float div_function(float a, float b, int iter);

long long get_first_zero(long long src, PREC prec);

fp32 fp32_to_fp32(fp32 src, ROUND_MODE round_mode);
fp16 fp16_to_fp16(fp16 src, ROUND_MODE round_mode);
bf16 bf16_to_bf16(bf16 src, ROUND_MODE round_mode);

fp16 fp32_to_fp16_all(fp32 src, ROUND_MODE round_mode);
bf16 fp32_to_bf16_all(fp32 src, ROUND_MODE round_mode);

int fp32_to_int(fp32 src, ROUND_MODE round_mode);
u32 fp32_to_u32(fp32 src, ROUND_MODE round_mode);
int fp16_to_int(fp16 src, ROUND_MODE round_mode);
u32 fp16_to_u32(fp16 src, ROUND_MODE round_mode);
int bf16_to_int(bf16 src, ROUND_MODE round_mode);
u32 bf16_to_u32(bf16 src, ROUND_MODE round_mode);

fp16 refine_fp16(fp16 src, ROUND_MODE round_mode);

fp32 int32_to_fp32(long long src, ROUND_MODE round_mode); // INT/UINT -> FP32
fp16 int32_to_fp16(long long src, ROUND_MODE round_mode); // INT/UINT -> FP16
bf16 int32_to_bf16(long long src, ROUND_MODE round_mode); // INT/UINT -> BFP16
fp32 int16_to_fp32(long long src); // INT16/UINT16 -> FP32
fp16 int16_to_fp16(long long src, ROUND_MODE round_mode); // INT16/UINT16 -> FP16
bf16 int16_to_bf16(long long src, ROUND_MODE round_mode); // INT16/UINT16 -> BFP16
fp32 int8_to_fp32(long long src); // INT8/UINT8 -> FP32
fp16 int8_to_fp16(long long src); // INT8/UINT8 -> FP16
bf16 int8_to_bf16(long long src); // INT8/UINT8 -> BFP16

#endif
