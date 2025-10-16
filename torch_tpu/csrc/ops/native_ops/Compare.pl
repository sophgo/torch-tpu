#include "ppl.h"
#include "ppl_wrapper_func.h"

using namespace ppl;

#ifdef __bm1690__
#define BLOCK_NUM 8
#elif defined __sg2260e__
#define BLOCK_NUM 4
#endif

template <typename T>
void cmp_const_kernel(uint8_t *ptr_output, T *ptr_input, float scalar, int mode, int outer_size, int inner_size, const int block_w) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();

    int C = outer_size;
    int W = inner_size;
    int block_c = LANE_NUM;
    int slice_per_core = div_up(outer_size, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, outer_size - core_offset);

    dim4 shape = {1, C, 1, W};
    auto res_gtensor = gtensor<uint8_t>(shape, GLOBAL, ptr_output);
    auto l_gtensor = gtensor<T>(shape, GLOBAL, ptr_input);

    dim4 block_shape = {1, block_c, 1, block_w};
    for (auto c_idx = 0; c_idx < slice_size_for_core; c_idx += block_c) {
        int c = min(block_c, slice_size_for_core - c_idx);
        for (auto w_idx = 0; w_idx < W; w_idx += block_w) {
            enable_pipeline();
            int w = min(block_w, W - w_idx);
            dim4 local_in_shape = {1, c, 1, w};
            dim4 input_offset = {0, core_offset + c_idx, 0, w_idx};
            auto left = make_tensor<T>(block_shape, local_in_shape);
            dma::load(left, l_gtensor.sub_view(local_in_shape, input_offset));

            auto res = make_tensor<uint8_t>(block_shape, local_in_shape);
            auto scalar_t = make_tensor<T>(block_shape, local_in_shape);
            tiu::fill(scalar_t, scalar);
            if (mode == 0) {
                tiu::eq(res, left, scalar_t, 1);
            } else if (mode == 1) {
                auto ones = make_tensor<T>(block_shape, local_in_shape);
                auto zeros = make_tensor<T>(block_shape, local_in_shape);
                tiu::fill(ones, 1.0f);
                tiu::fill(zeros, 0.0f);
                auto res_T = make_tensor<T>(block_shape, local_in_shape);
                tiu::eq_select(res_T, left, scalar_t, zeros, ones);
                tiu::cast(res, res_T);
            } else if (mode == 2) {
                tiu::gt(res, left, scalar_t, 1);
            } else if (mode == 3) {
                auto tmp_gt = tensor<uint8_t>(shape);
                auto tmp_eq = tensor<uint8_t>(shape);
                tiu::gt(tmp_gt, left, scalar_t, 1);
                tiu::eq(tmp_eq, left, scalar_t, 1);
                tiu::bitwise_or(res, tmp_gt, tmp_eq);
            } else if (mode == 4) {
                tiu::lt(res, left, scalar_t, 1);
            } else if (mode == 5) {
                auto tmp_lt = tensor<uint8_t>(shape);
                auto tmp_eq = tensor<uint8_t>(shape);
                tiu::lt(tmp_lt, left, scalar_t, 1);
                tiu::eq(tmp_eq, left, scalar_t, 1);
                tiu::bitwise_or(res, tmp_lt, tmp_eq);
            }

            dma::store(res_gtensor.sub_view(local_in_shape, input_offset) , res);
        }
    }
}

__KERNEL__ void compare_const_fp32(uint8_t *ptr_output, fp32 *ptr_input, float scalar,
           int mode, int outer_size, int inner_size, const int tile_size) {
    cmp_const_kernel<fp32>(ptr_output, ptr_input, scalar, mode, outer_size, inner_size, tile_size);
}

__KERNEL__ void compare_const_fp16(uint8_t *ptr_output, fp16 *ptr_input, float scalar,
           int mode, int outer_size, int inner_size, const int tile_size) {
    cmp_const_kernel<fp16>(ptr_output, ptr_input, scalar, mode, outer_size, inner_size, tile_size);
}

__KERNEL__ void compare_const_bf16(uint8_t *ptr_output, bf16 *ptr_input, float scalar,
           int mode, int outer_size, int inner_size, const int tile_size) {
    cmp_const_kernel<bf16>(ptr_output, ptr_input, scalar, mode, outer_size, inner_size, tile_size);
}

__KERNEL__ void compare_const_int32(uint8_t *ptr_output, int32 *ptr_input, float scalar,
           int mode, int outer_size, int inner_size, const int tile_size) {
    cmp_const_kernel<int32>(ptr_output, ptr_input, scalar, mode, outer_size, inner_size, tile_size);
}

template <typename T>
void cmp_kernel(uint8_t *ptr_output, T *ptr_input, T *ptr_other, int mode, int outer_size, int inner_size, const int block_w) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();

    int C = outer_size;
    int W = inner_size;
    int block_c = LANE_NUM;
    int slice_per_core = div_up(outer_size, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, outer_size - core_offset);

    dim4 shape = {1, C, 1, W};
    auto res_gtensor = gtensor<uint8_t>(shape, GLOBAL, ptr_output);
    auto l_gtensor = gtensor<T>(shape, GLOBAL, ptr_input);
    auto r_gtensor = gtensor<T>(shape, GLOBAL, ptr_other);

    dim4 block_shape = {1, block_c, 1, block_w};
    for (auto c_idx = 0; c_idx < slice_size_for_core; c_idx += block_c) {
        int c = min(block_c, slice_size_for_core - c_idx);
        for (auto w_idx = 0; w_idx < W; w_idx += block_w) {
            enable_pipeline();
            int w = min(block_w, W - w_idx);
            dim4 local_in_shape = {1, c, 1, w};
            dim4 input_offset = {0, core_offset + c_idx, 0, w_idx};
            auto left = make_tensor<T>(block_shape, local_in_shape);
            dma::load(left, l_gtensor.sub_view(local_in_shape, input_offset));
            auto right = make_tensor<T>(block_shape, local_in_shape);
            dma::load(right, r_gtensor.sub_view(local_in_shape, input_offset));

            auto res = make_tensor<uint8_t>(block_shape, local_in_shape);
            if (mode == 0) {
                tiu::eq(res, left, right, 1);
            } else if (mode == 1) {
                auto ones = make_tensor<T>(block_shape, local_in_shape);
                auto zeros = make_tensor<T>(block_shape, local_in_shape);
                tiu::fill(ones, 1.0f);
                tiu::fill(zeros, 0.0f);
                auto res_T = make_tensor<T>(block_shape, local_in_shape);
                tiu::eq_select(res_T, left, right, zeros, ones);
                tiu::cast(res, res_T);
            } else if (mode == 2) {
                tiu::gt(res, left, right, 1);
            } else if (mode == 3) {
                auto tmp_gt = tensor<uint8_t>(shape);
                auto tmp_eq = tensor<uint8_t>(shape);
                tiu::gt(tmp_gt, left, right, 1);
                tiu::eq(tmp_eq, left, right, 1);
                tiu::bitwise_or(res, tmp_gt, tmp_eq);
            } else if (mode == 4) {
                tiu::lt(res, left, right, 1);
            } else if (mode == 5) {
                auto tmp_lt = tensor<uint8_t>(shape);
                auto tmp_eq = tensor<uint8_t>(shape);
                tiu::lt(tmp_lt, left, right, 1);
                tiu::eq(tmp_eq, left, right, 1);
                tiu::bitwise_or(res, tmp_lt, tmp_eq);
            }

            dma::store(res_gtensor.sub_view(local_in_shape, input_offset), res);
        }
    }
}

__KERNEL__ void compare_fp32(uint8_t *ptr_output, fp32 *ptr_input, fp32 *ptr_other,
           int mode, int outer_size, int inner_size, const int tile_size) {
   cmp_kernel<fp32>(ptr_output, ptr_input, ptr_other, mode, outer_size, inner_size, tile_size);
}

__KERNEL__ void compare_fp16(uint8_t *ptr_output, fp16 *ptr_input, fp16 *ptr_other,
           int mode, int outer_size, int inner_size, const int tile_size) {
   cmp_kernel<fp16>(ptr_output, ptr_input, ptr_other, mode, outer_size, inner_size, tile_size);
}

__KERNEL__ void compare_bf16(uint8_t *ptr_output, bf16 *ptr_input, bf16 *ptr_other,
           int mode, int outer_size, int inner_size, const int tile_size) {
   cmp_kernel<bf16>(ptr_output, ptr_input, ptr_other, mode, outer_size, inner_size, tile_size);
}

__KERNEL__ void compare_int32(uint8_t *ptr_output, int32 *ptr_input, int32 *ptr_other,
           int mode, int outer_size, int inner_size, const int tile_size) {
  cmp_kernel<int32>(ptr_output, ptr_input, ptr_other, mode, outer_size, inner_size, tile_size);
}

template <typename T, typename U>
void binary_cmp_bcast_kernel(U* ptr_output, T* ptr_input, T* ptr_other,
                        int binary_type, const int block_n,
                        int in_N, int in_C, int in_H, int in_W,
                        int ot_N, int ot_C, int ot_H, int ot_W,
                        int is_float, bool is_const = 0, float alpha = 1.0f, int mode = -1) {

    int out_N = max(in_N, ot_N);
    int out_C = max(in_C, ot_C);
    int out_H = max(in_H, ot_H);
    int out_W = max(in_W, ot_W);

    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();
    int slice_per_core = div_up(out_N, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, out_N - core_offset);

    dim4 in_gshape = {in_N , in_C, in_H , in_W};
    dim4 out_shape = {out_N , out_C, out_H , out_W};

    auto dst_g = gtensor<U>(out_shape, GLOBAL, ptr_output);
    auto a_g = gtensor<T>(in_gshape, GLOBAL, ptr_input);
    dim4 ot_gshape = {ot_N, ot_C, ot_H, ot_W};
    auto b_g = gtensor<T>(ot_gshape, GLOBAL, ptr_other);

    dim4 a_local_shape_block = {block_n, in_C, in_H , in_W};
    dim4 b_local_shape_block = {ot_N>1?block_n:ot_N, out_C, out_H , out_W};
    dim4 out_local_shape_block = {block_n, out_C, out_H , out_W};

    for (auto n_idx = 0; n_idx < slice_size_for_core; n_idx += block_n) {
        enable_pipeline();
        int n = min(block_n, slice_size_for_core - n_idx);
        dim4 out_local_shape = {n, out_C, out_H , out_W};
        dim4 a_local_shape = {n, in_C, in_H , in_W};
        dim4 b_local_shape = {ot_N>1?n:ot_N, out_C, out_H , out_W};
        dim4 a_offset = {n_idx+core_offset, 0, 0, 0};
        dim4 b_offset = {ot_N>1?(n_idx+core_offset):0, 0, 0, 0};

        auto a_l = make_tensor<T>(a_local_shape_block, a_local_shape);
        dma::load(a_l, a_g.sub_view(a_local_shape, a_offset));
        auto b_l = make_tensor<T>(b_local_shape_block, b_local_shape);
        dma::load(b_l, b_g.sub_view(b_local_shape, b_offset));
        auto o_l = make_tensor<U>(out_local_shape_block, out_local_shape);

        dim4 a_stride_tmp;
        get_stride<T>(&a_stride_tmp, &a_local_shape, TPU_ALIGN);
        int a_sn = (out_N > in_N) ? 0 : a_stride_tmp.n;
        int a_sc = (out_C > in_C) ? 0 : a_stride_tmp.c;
        int a_sh = (out_H > in_H) ? 0 : a_stride_tmp.h;
        int a_sw = (out_W > in_W) ? 0 : a_stride_tmp.w;
        dim4 a_stride = {a_sn, a_sc, a_sh, a_sw};

        dim4 b_stride_tmp;
        get_stride<T>(&b_stride_tmp, &b_local_shape, TPU_ALIGN);
        int b_sn = (out_N > ot_N) ? 0 : b_stride_tmp.n;
        int b_sc = (out_C > ot_C) ? 0 : b_stride_tmp.c;
        int b_sh = (out_H > ot_H) ? 0 : b_stride_tmp.h;
        int b_sw = (out_W > ot_W) ? 0 : b_stride_tmp.w;
        dim4 b_stride = {b_sn, b_sc, b_sh, b_sw};
        if (mode == 0) {
            tiu::eq(o_l, a_l.view(out_local_shape, a_stride), b_l.view(out_local_shape, b_stride), 1);
        } else if (mode == 1) {
            auto tmp_eq = make_tensor<uint8_t>(out_local_shape_block, out_local_shape);
            tiu::eq(tmp_eq, a_l.view(out_local_shape, a_stride), b_l.view(out_local_shape, b_stride), 1);

            auto ones = make_tensor<uint8_t>(out_local_shape_block, out_local_shape);
            tiu::fill(ones, 1);
            tiu::sub(o_l, ones, tmp_eq);
        } else if (mode == 2) {
            tiu::gt(o_l, a_l.view(out_local_shape, a_stride), b_l.view(out_local_shape, b_stride), 1);
        }
        else if (mode == 3) {
            auto tmp_gt = make_tensor<uint8_t>(out_local_shape_block, out_local_shape);
            auto tmp_eq = make_tensor<uint8_t>(out_local_shape_block, out_local_shape);
            tiu::gt(tmp_gt, a_l.view(out_local_shape, a_stride), b_l.view(out_local_shape, b_stride), 1);
            tiu::eq(tmp_eq, a_l.view(out_local_shape, a_stride), b_l.view(out_local_shape, b_stride), 1);
            tiu::bitwise_or(o_l, tmp_gt, tmp_eq);
        } else if (mode == 4) {
            tiu::lt(o_l, a_l.view(out_local_shape, a_stride), b_l.view(out_local_shape, b_stride), 1);
        } else if (mode == 5) {
            auto tmp_lt = make_tensor<uint8_t>(out_local_shape_block, out_local_shape);
            auto tmp_eq = make_tensor<uint8_t>(out_local_shape_block, out_local_shape);
            tiu::lt(tmp_lt, a_l.view(out_local_shape, a_stride), b_l.view(out_local_shape, b_stride), 1);
            tiu::eq(tmp_eq, a_l.view(out_local_shape, a_stride), b_l.view(out_local_shape, b_stride), 1);
            tiu::bitwise_or(o_l, tmp_lt, tmp_eq);
        }

        dma::store(dst_g.sub_view(out_local_shape, a_offset), o_l);
    }
}

__KERNEL__ void compare_bcast_fp32(uint8_t *ptr_output, fp32 *ptr_input, fp32 *ptr_other,
           int mode, int outer_size, int inner_size, const int tile_size,
           int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
   binary_cmp_bcast_kernel<fp32, uint8_t>(ptr_output, ptr_input, ptr_other, -1, inner_size, iN, iC, iH, iW,
                    oN, oC, oH, oW, 1, 0, 0, mode);
}

__KERNEL__ void compare_bcast_fp16(uint8_t *ptr_output, fp16 *ptr_input, fp16 *ptr_other,
           int mode, int outer_size, int inner_size, const int tile_size,
           int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
   binary_cmp_bcast_kernel<fp16, uint8_t>(ptr_output, ptr_input, ptr_other, -1, inner_size, iN, iC, iH, iW,
                    oN, oC, oH, oW, 1, 0, 0, mode);
}

__KERNEL__ void compare_bcast_bf16(uint8_t *ptr_output, bf16 *ptr_input, bf16 *ptr_other,
           int mode, int outer_size, int inner_size, const int tile_size,
           int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
    binary_cmp_bcast_kernel<bf16, uint8_t>(ptr_output, ptr_input, ptr_other, -1, inner_size, iN, iC, iH, iW,
                oN, oC, oH, oW, 1, 0, 0, mode);
}

__KERNEL__ void compare_bcast_int32(uint8_t *ptr_output, int32 *ptr_input, int32 *ptr_other,
           int mode, int outer_size, int inner_size, const int tile_size,
           int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
    binary_cmp_bcast_kernel<int32, uint8_t>(ptr_output, ptr_input, ptr_other, -1, inner_size, iN, iC, iH, iW,
                oN, oC, oH, oW, 1, 0, 0, mode);
}

template <typename T>
void shift_const_kernel(T *ptr_output, T *ptr_input, char shift_c,
                        int outer_size, int inner_size, const int block_w) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();

    int C = outer_size;
    int W = inner_size;
    int block_c = LANE_NUM;
    int slice_per_core = div_up(outer_size, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, outer_size - core_offset);

    dim4 shape = {1, C, 1, W};
    auto res_gtensor = gtensor<T>(shape, GLOBAL, ptr_output);
    auto l_gtensor = gtensor<T>(shape, GLOBAL, ptr_input);

    dim4 block_shape = {1, block_c, 1, block_w};
    for (auto c_idx = 0; c_idx < slice_size_for_core; c_idx += block_c) {
        int c = min(block_c, slice_size_for_core - c_idx);
        for (auto w_idx = 0; w_idx < W; w_idx += block_w) {
            enable_pipeline();
            int w = min(block_w, W - w_idx);
            dim4 local_in_shape = {1, c, 1, w};
            dim4 input_offset = {0, core_offset + c_idx, 0, w_idx};
            auto left = make_tensor<T>(block_shape, local_in_shape);
            dma::load(left, l_gtensor.sub_view(local_in_shape, input_offset));

            auto res = make_tensor<T>(block_shape, local_in_shape);

            tiu::shift(res, left, shift_c, RM_TOWARDS_ZERO);

            dma::store(res_gtensor.sub_view(local_in_shape, input_offset) , res);
        }
    }
}

__KERNEL__ void shift_const_int32(int32 *ptr_output, int32 *ptr_input, char shift_c,
           int outer_size, int inner_size, const int tile_size) {
    shift_const_kernel<int32>(ptr_output, ptr_input, shift_c, outer_size, inner_size, tile_size);
}

template <typename T>
void shift_kernel(T *ptr_output, T *ptr_input, T *ptr_other, int mode, int outer_size, int inner_size, const int block_w) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();

    int C = outer_size;
    int W = inner_size;
    int block_c = LANE_NUM;
    int slice_per_core = div_up(outer_size, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, outer_size - core_offset);

    dim4 shape = {1, C, 1, W};
    auto res_gtensor = gtensor<T>(shape, GLOBAL, ptr_output);
    auto l_gtensor = gtensor<T>(shape, GLOBAL, ptr_input);
    auto r_gtensor = gtensor<T>(shape, GLOBAL, ptr_other);

    dim4 block_shape = {1, block_c, 1, block_w};
    for (auto c_idx = 0; c_idx < slice_size_for_core; c_idx += block_c) {
        int c = min(block_c, slice_size_for_core - c_idx);
        for (auto w_idx = 0; w_idx < W; w_idx += block_w) {
            enable_pipeline();
            int w = min(block_w, W - w_idx);
            dim4 local_in_shape = {1, c, 1, w};
            dim4 input_offset = {0, core_offset + c_idx, 0, w_idx};
            auto left = make_tensor<T>(block_shape, local_in_shape);
            dma::load(left, l_gtensor.sub_view(local_in_shape, input_offset));
            auto right = make_tensor<T>(block_shape, local_in_shape);
            dma::load(right, r_gtensor.sub_view(local_in_shape, input_offset));

            auto res = make_tensor<T>(block_shape, local_in_shape);
            auto right_i8 = make_tensor<int8_t>(block_shape, local_in_shape);
            tiu::cast(right_i8, right);
            tiu::shift(res, left, right_i8, RM_TOWARDS_ZERO);
            dma::store(res_gtensor.sub_view(local_in_shape, input_offset) , res);
        }
    }
}

__KERNEL__ void shift_forward_int32(int32 *ptr_output, int32 *ptr_input, int32 *ptr_other, int mode,
           int outer_size, int inner_size, const int tile_size) {
    shift_kernel<int32>(ptr_output, ptr_input, ptr_other, mode, outer_size, inner_size, tile_size);
}

template <typename T, typename U>
void shift_bcast_kernel(U* ptr_output, T* ptr_input, T* ptr_other,
                        const int block_n,
                        int in_N, int in_C, int in_H, int in_W,
                        int ot_N, int ot_C, int ot_H, int ot_W,
                        int mode) {

    int out_N = max(in_N, ot_N);
    int out_C = max(in_C, ot_C);
    int out_H = max(in_H, ot_H);
    int out_W = max(in_W, ot_W);

    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();
    int slice_per_core = div_up(out_N, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, out_N - core_offset);

    dim4 in_gshape = {in_N , in_C, in_H , in_W};
    dim4 out_shape = {out_N , out_C, out_H , out_W};

    auto dst_g = gtensor<U>(out_shape, GLOBAL, ptr_output);
    auto a_g = gtensor<T>(in_gshape, GLOBAL, ptr_input);
    dim4 ot_gshape = {ot_N, ot_C, ot_H, ot_W};
    auto b_g = gtensor<T>(ot_gshape, GLOBAL, ptr_other);

    dim4 a_local_shape_block = {block_n, in_C, in_H , in_W};
    dim4 b_local_shape_block = {ot_N>1?block_n:ot_N, out_C, out_H , out_W};
    dim4 out_local_shape_block = {block_n, out_C, out_H , out_W};

    for (auto n_idx = 0; n_idx < slice_size_for_core; n_idx += block_n) {
        enable_pipeline();
        int n = min(block_n, slice_size_for_core - n_idx);
        dim4 out_local_shape = {n, out_C, out_H , out_W};
        dim4 a_local_shape = {n, in_C, in_H , in_W};
        dim4 b_local_shape = {ot_N>1?n:ot_N, out_C, out_H , out_W};
        dim4 a_offset = {n_idx+core_offset, 0, 0, 0};
        dim4 b_offset = {ot_N>1?(n_idx+core_offset):0, 0, 0, 0};

        auto a_l = make_tensor<T>(a_local_shape_block, a_local_shape);
        dma::load(a_l, a_g.sub_view(a_local_shape, a_offset));
        auto b_l = make_tensor<T>(b_local_shape_block, b_local_shape);
        dma::load(b_l, b_g.sub_view(b_local_shape, b_offset));
        auto o_l = make_tensor<U>(out_local_shape_block, out_local_shape);

        dim4 a_stride_tmp;
        get_stride<T>(&a_stride_tmp, &a_local_shape, TPU_ALIGN);
        int a_sn = (out_N > in_N) ? 0 : a_stride_tmp.n;
        int a_sc = (out_C > in_C) ? 0 : a_stride_tmp.c;
        int a_sh = (out_H > in_H) ? 0 : a_stride_tmp.h;
        int a_sw = (out_W > in_W) ? 0 : a_stride_tmp.w;
        dim4 a_stride = {a_sn, a_sc, a_sh, a_sw};

        dim4 b_stride_tmp;
        get_stride<T>(&b_stride_tmp, &b_local_shape, TPU_ALIGN);
        int b_sn = (out_N > ot_N) ? 0 : b_stride_tmp.n;
        int b_sc = (out_C > ot_C) ? 0 : b_stride_tmp.c;
        int b_sh = (out_H > ot_H) ? 0 : b_stride_tmp.h;
        int b_sw = (out_W > ot_W) ? 0 : b_stride_tmp.w;
        dim4 b_stride = {b_sn, b_sc, b_sh, b_sw};
        auto b_i8 = make_tensor<int8_t>(out_local_shape_block, out_local_shape);
        tiu::cast(b_i8, b_l.view(out_local_shape, b_stride));
        tiu::shift(o_l, a_l.view(out_local_shape, a_stride), b_i8, RM_TOWARDS_ZERO);
        dma::store(dst_g.sub_view(out_local_shape, a_offset), o_l);
    }

}

__KERNEL__ void shift_bcast_int32(int32 *ptr_output, int32 *ptr_input, int32 *ptr_other,
           int mode, int outer_size, int inner_size, const int tile_size,
           int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
    shift_bcast_kernel<int32, int32>(ptr_output, ptr_input, ptr_other, inner_size, iN, iC, iH, iW,
                oN, oC, oH, oW, mode);
}

template <typename T>
void minmax_const_kernel(T *ptr_output, T *ptr_input, float minmax_C, int mode,
                        int outer_size, int inner_size, const int block_w) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();

    int C = outer_size;
    int W = inner_size;
    int block_c = LANE_NUM;
    int slice_per_core = div_up(outer_size, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, outer_size - core_offset);

    dim4 shape = {1, C, 1, W};
    auto res_gtensor = gtensor<T>(shape, GLOBAL, ptr_output);
    auto l_gtensor = gtensor<T>(shape, GLOBAL, ptr_input);

    dim4 block_shape = {1, block_c, 1, block_w};
    for (auto c_idx = 0; c_idx < slice_size_for_core; c_idx += block_c) {
        int c = min(block_c, slice_size_for_core - c_idx);
        for (auto w_idx = 0; w_idx < W; w_idx += block_w) {
            enable_pipeline();
            int w = min(block_w, W - w_idx);
            dim4 local_in_shape = {1, c, 1, w};
            dim4 input_offset = {0, core_offset + c_idx, 0, w_idx};
            auto left = make_tensor<T>(block_shape, local_in_shape);
            dma::load(left, l_gtensor.sub_view(local_in_shape, input_offset));

            auto res = make_tensor<T>(block_shape, local_in_shape);
            if constexpr (std::is_same<T, fp32>::value ||
                          std::is_same<T, fp16>::value ||
                          std::is_same<T, bf16>::value) {
                if (mode) {
                    tiu::fmin(res, left, minmax_C);
                } else {
                    tiu::fmax(res, left, minmax_C);
                }
            } else if constexpr (std::is_same<T, int32>::value) {
                if (mode) {
                    tiu::min(res, left, minmax_C);
                } else {
                    tiu::max(res, left, minmax_C);
                }
            }
            dma::store(res_gtensor.sub_view(local_in_shape, input_offset) , res);
        }
    }
}

__KERNEL__ void minmax_fp32(fp32 *ptr_output, fp32 *ptr_input, float minmax_c, int mode,
           int outer_size, int inner_size, const int tile_size) {
    minmax_const_kernel<fp32>(ptr_output, ptr_input, minmax_c, mode, outer_size, inner_size, tile_size);
}

__KERNEL__ void minmax_fp16(fp16 *ptr_output, fp16 *ptr_input, float minmax_c, int mode,
           int outer_size, int inner_size, const int tile_size) {
    minmax_const_kernel<fp16>(ptr_output, ptr_input, minmax_c, mode, outer_size, inner_size, tile_size);
}

__KERNEL__ void minmax_bf16(bf16 *ptr_output, bf16 *ptr_input, float minmax_c, int mode,
           int outer_size, int inner_size, const int tile_size) {
    minmax_const_kernel<bf16>(ptr_output, ptr_input, minmax_c, mode, outer_size, inner_size, tile_size);
}

__KERNEL__ void minmax_int32(int32 *ptr_output, int32 *ptr_input, float minmax_c, int mode,
           int outer_size, int inner_size, const int tile_size) {
    minmax_const_kernel<int32>(ptr_output, ptr_input, minmax_c, mode, outer_size, inner_size, tile_size);
}

template <typename T>
void minmax_kernel(T *ptr_output, T *ptr_input, T *ptr_other, int mode, int outer_size, int inner_size, const int block_w) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();

    int C = outer_size;
    int W = inner_size;
    int block_c = LANE_NUM;
    int slice_per_core = div_up(outer_size, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, outer_size - core_offset);

    dim4 shape = {1, C, 1, W};
    auto res_gtensor = gtensor<T>(shape, GLOBAL, ptr_output);
    auto l_gtensor = gtensor<T>(shape, GLOBAL, ptr_input);
    auto r_gtensor = gtensor<T>(shape, GLOBAL, ptr_other);

    dim4 block_shape = {1, block_c, 1, block_w};
    for (auto c_idx = 0; c_idx < slice_size_for_core; c_idx += block_c) {
        int c = min(block_c, slice_size_for_core - c_idx);
        for (auto w_idx = 0; w_idx < W; w_idx += block_w) {
            enable_pipeline();
            int w = min(block_w, W - w_idx);
            dim4 local_in_shape = {1, c, 1, w};
            dim4 input_offset = {0, core_offset + c_idx, 0, w_idx};
            auto left = make_tensor<T>(block_shape, local_in_shape);
            dma::load(left, l_gtensor.sub_view(local_in_shape, input_offset));
            auto right = make_tensor<T>(block_shape, local_in_shape);
            dma::load(right, r_gtensor.sub_view(local_in_shape, input_offset));

            auto res = make_tensor<T>(block_shape, local_in_shape);
            if constexpr (std::is_same<T, fp32>::value ||
                          std::is_same<T, fp16>::value ||
                          std::is_same<T, bf16>::value) {
                if (mode) {
                    tiu::fmin(res, left, right);
                } else {
                    tiu::fmax(res, left, right);
                }
            } else if constexpr (std::is_same<T, int32>::value) {
                if (mode) {
                    tiu::min(res, left, right);
                } else {
                    tiu::max(res, left, right);
                }
            }

            dma::store(res_gtensor.sub_view(local_in_shape, input_offset) , res);
        }
    }
}

__KERNEL__ void minmax_forward_int32(int32 *ptr_output, int32 *ptr_input, int32 *ptr_other, int mode,
           int outer_size, int inner_size, const int tile_size) {
    minmax_kernel<int32>(ptr_output, ptr_input, ptr_other, mode, outer_size, inner_size, tile_size);
}

__KERNEL__ void minmax_forward_fp32(fp32 *ptr_output, fp32 *ptr_input, fp32 *ptr_other, int mode,
           int outer_size, int inner_size, const int tile_size) {
    minmax_kernel<fp32>(ptr_output, ptr_input, ptr_other, mode, outer_size, inner_size, tile_size);
}

__KERNEL__ void minmax_forward_fp16(fp16 *ptr_output, fp16 *ptr_input, fp16 *ptr_other, int mode,
           int outer_size, int inner_size, const int tile_size) {
    minmax_kernel<fp16>(ptr_output, ptr_input, ptr_other, mode, outer_size, inner_size, tile_size);
}

__KERNEL__ void minmax_forward_bf16(bf16 *ptr_output, bf16 *ptr_input, bf16 *ptr_other, int mode,
           int outer_size, int inner_size, const int tile_size) {
    minmax_kernel<bf16>(ptr_output, ptr_input, ptr_other, mode, outer_size, inner_size, tile_size);
}

template <typename T, typename U>
void minmax_bcast_kernel(U* ptr_output, T* ptr_input, T* ptr_other,
                        const int block_n,
                        int in_N, int in_C, int in_H, int in_W,
                        int ot_N, int ot_C, int ot_H, int ot_W,
                        int mode) {

    int out_N = max(in_N, ot_N);
    int out_C = max(in_C, ot_C);
    int out_H = max(in_H, ot_H);
    int out_W = max(in_W, ot_W);

    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();
    int slice_per_core = div_up(out_N, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, out_N - core_offset);

    dim4 in_gshape = {in_N , in_C, in_H , in_W};
    dim4 out_shape = {out_N , out_C, out_H , out_W};

    auto dst_g = gtensor<U>(out_shape, GLOBAL, ptr_output);
    auto a_g = gtensor<T>(in_gshape, GLOBAL, ptr_input);
    dim4 ot_gshape = {ot_N, ot_C, ot_H, ot_W};
    auto b_g = gtensor<T>(ot_gshape, GLOBAL, ptr_other);

    dim4 a_local_shape_block = {block_n, in_C, in_H , in_W};
    dim4 b_local_shape_block = {ot_N>1?block_n:ot_N, out_C, out_H , out_W};
    dim4 out_local_shape_block = {block_n, out_C, out_H , out_W};

    for (auto n_idx = 0; n_idx < slice_size_for_core; n_idx += block_n) {
        enable_pipeline();
        int n = min(block_n, slice_size_for_core - n_idx);
        dim4 out_local_shape = {n, out_C, out_H , out_W};
        dim4 a_local_shape = {n, in_C, in_H , in_W};
        dim4 b_local_shape = {ot_N>1?n:ot_N, out_C, out_H , out_W};
        dim4 a_offset = {n_idx+core_offset, 0, 0, 0};
        dim4 b_offset = {ot_N>1?(n_idx+core_offset):0, 0, 0, 0};

        auto a_l = make_tensor<T>(a_local_shape_block, a_local_shape);
        dma::load(a_l, a_g.sub_view(a_local_shape, a_offset));
        auto b_l = make_tensor<T>(b_local_shape_block, b_local_shape);
        dma::load(b_l, b_g.sub_view(b_local_shape, b_offset));
        auto o_l = make_tensor<U>(out_local_shape_block, out_local_shape);

        dim4 a_stride_tmp;
        get_stride<T>(&a_stride_tmp, &a_local_shape, TPU_ALIGN);
        int a_sn = (out_N > in_N) ? 0 : a_stride_tmp.n;
        int a_sc = (out_C > in_C) ? 0 : a_stride_tmp.c;
        int a_sh = (out_H > in_H) ? 0 : a_stride_tmp.h;
        int a_sw = (out_W > in_W) ? 0 : a_stride_tmp.w;
        dim4 a_stride = {a_sn, a_sc, a_sh, a_sw};

        dim4 b_stride_tmp;
        get_stride<T>(&b_stride_tmp, &b_local_shape, TPU_ALIGN);
        int b_sn = (out_N > ot_N) ? 0 : b_stride_tmp.n;
        int b_sc = (out_C > ot_C) ? 0 : b_stride_tmp.c;
        int b_sh = (out_H > ot_H) ? 0 : b_stride_tmp.h;
        int b_sw = (out_W > ot_W) ? 0 : b_stride_tmp.w;
        dim4 b_stride = {b_sn, b_sc, b_sh, b_sw};

        if constexpr (std::is_same<T, fp32>::value ||
                        std::is_same<T, fp16>::value ||
                        std::is_same<T, bf16>::value) {
            if (mode) {
                tiu::fmin(o_l, a_l.view(out_local_shape, a_stride), b_l.view(out_local_shape, b_stride));
            } else {
                tiu::fmax(o_l, a_l.view(out_local_shape, a_stride), b_l.view(out_local_shape, b_stride));
            }
        } else if constexpr (std::is_same<T, int32>::value) {
            if (mode) {
                tiu::min(o_l, a_l.view(out_local_shape, a_stride), b_l.view(out_local_shape, b_stride));
            } else {
                tiu::max(o_l, a_l.view(out_local_shape, a_stride), b_l.view(out_local_shape, b_stride));
            }
        }

        dma::store(dst_g.sub_view(out_local_shape, a_offset), o_l);
    }

}

__KERNEL__ void minmax_bcast_int32(int32 *ptr_output, int32 *ptr_input, int32 *ptr_other,
           int mode, int outer_size, int inner_size, const int tile_size,
           int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
    minmax_bcast_kernel<int32, int32>(ptr_output, ptr_input, ptr_other, inner_size, iN, iC, iH, iW,
                oN, oC, oH, oW, mode);
}

__KERNEL__ void minmax_bcast_fp32(fp32 *ptr_output, fp32 *ptr_input, fp32 *ptr_other,
           int mode, int outer_size, int inner_size, const int tile_size,
           int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
    minmax_bcast_kernel<fp32, fp32>(ptr_output, ptr_input, ptr_other, inner_size, iN, iC, iH, iW,
                oN, oC, oH, oW, mode);
}

__KERNEL__ void minmax_bcast_fp16(fp16 *ptr_output, fp16 *ptr_input, fp16 *ptr_other,
           int mode, int outer_size, int inner_size, const int tile_size,
           int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
    minmax_bcast_kernel<fp16, fp16>(ptr_output, ptr_input, ptr_other, inner_size, iN, iC, iH, iW,
                oN, oC, oH, oW, mode);
}

__KERNEL__ void minmax_bcast_bf16(bf16 *ptr_output, bf16 *ptr_input, bf16 *ptr_other,
           int mode, int outer_size, int inner_size, const int tile_size,
           int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
    minmax_bcast_kernel<bf16, bf16>(ptr_output, ptr_input, ptr_other, inner_size, iN, iC, iH, iW,
                oN, oC, oH, oW, mode);
}

template <typename T>
void bitwise_const_kernel(T *ptr_output, T *ptr_input, T bitwise_c, int mode,
                        int outer_size, int inner_size, const int block_w) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();

    int C = outer_size;
    int W = inner_size;
    int block_c = LANE_NUM;
    int slice_per_core = div_up(outer_size, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, outer_size - core_offset);

    dim4 shape = {1, C, 1, W};
    auto res_gtensor = gtensor<T>(shape, GLOBAL, ptr_output);
    auto l_gtensor = gtensor<T>(shape, GLOBAL, ptr_input);

    dim4 block_shape = {1, block_c, 1, block_w};
    for (auto c_idx = 0; c_idx < slice_size_for_core; c_idx += block_c) {
        int c = min(block_c, slice_size_for_core - c_idx);
        for (auto w_idx = 0; w_idx < W; w_idx += block_w) {
            enable_pipeline();
            int w = min(block_w, W - w_idx);
            dim4 local_in_shape = {1, c, 1, w};
            dim4 input_offset = {0, core_offset + c_idx, 0, w_idx};
            auto left = make_tensor<T>(block_shape, local_in_shape);
            dma::load(left, l_gtensor.sub_view(local_in_shape, input_offset));

            auto res = make_tensor<T>(block_shape, local_in_shape);
            if (mode == 0) {
                tiu::bitwise_xor(res, left, bitwise_c);
            } else if (mode == 1) {
                tiu::bitwise_and(res, left, bitwise_c);
            } else if (mode == 2) {
                tiu::bitwise_or(res, left, bitwise_c);
            }

            dma::store(res_gtensor.sub_view(local_in_shape, input_offset) , res);
        }
    }
}

__KERNEL__ void bitwise_const_int32(int32 *ptr_output, int32 *ptr_input, int32 bitwise_c, int mode,
           int outer_size, int inner_size, const int tile_size) {
    bitwise_const_kernel<int32>(ptr_output, ptr_input, bitwise_c, mode, outer_size, inner_size, tile_size);
}

template <typename T>
void bitwise_kernel(T *ptr_output, T *ptr_input, T *ptr_other, int mode, int outer_size, int inner_size, const int block_w) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();

    int C = outer_size;
    int W = inner_size;
    int block_c = LANE_NUM;
    int slice_per_core = div_up(outer_size, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, outer_size - core_offset);

    dim4 shape = {1, C, 1, W};
    auto res_gtensor = gtensor<T>(shape, GLOBAL, ptr_output);
    auto l_gtensor = gtensor<T>(shape, GLOBAL, ptr_input);
    auto r_gtensor = gtensor<T>(shape, GLOBAL, ptr_other);

    dim4 block_shape = {1, block_c, 1, block_w};
    for (auto c_idx = 0; c_idx < slice_size_for_core; c_idx += block_c) {
        int c = min(block_c, slice_size_for_core - c_idx);
        for (auto w_idx = 0; w_idx < W; w_idx += block_w) {
            enable_pipeline();
            int w = min(block_w, W - w_idx);
            dim4 local_in_shape = {1, c, 1, w};
            dim4 input_offset = {0, core_offset + c_idx, 0, w_idx};
            auto left = make_tensor<T>(block_shape, local_in_shape);
            dma::load(left, l_gtensor.sub_view(local_in_shape, input_offset));
            auto right = make_tensor<T>(block_shape, local_in_shape);
            dma::load(right, r_gtensor.sub_view(local_in_shape, input_offset));

            auto res = make_tensor<T>(block_shape, local_in_shape);
            if (mode == 0)
                tiu::bitwise_xor(res, left, right);
            else if (mode == 1)
                tiu::bitwise_and(res, left, right);
            else if (mode == 2)
                tiu::bitwise_or(res, left, right);
            dma::store(res_gtensor.sub_view(local_in_shape, input_offset) , res);
        }
    }
}

__KERNEL__ void bitwise_forward_int32(int32 *ptr_output, int32 *ptr_input, int32 *ptr_other, int mode,
           int outer_size, int inner_size, const int tile_size) {
    bitwise_kernel<int32>(ptr_output, ptr_input, ptr_other, mode, outer_size, inner_size, tile_size);
}

template <typename T, typename U>
void bitwise_bcast_kernel(U* ptr_output, T* ptr_input, T* ptr_other,
                        const int block_n,
                        int in_N, int in_C, int in_H, int in_W,
                        int ot_N, int ot_C, int ot_H, int ot_W,
                        int mode) {

    int out_N = max(in_N, ot_N);
    int out_C = max(in_C, ot_C);
    int out_H = max(in_H, ot_H);
    int out_W = max(in_W, ot_W);

    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();
    int slice_per_core = div_up(out_N, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, out_N - core_offset);

    dim4 in_gshape = {in_N , in_C, in_H , in_W};
    dim4 out_shape = {out_N , out_C, out_H , out_W};

    auto dst_g = gtensor<U>(out_shape, GLOBAL, ptr_output);
    auto a_g = gtensor<T>(in_gshape, GLOBAL, ptr_input);
    dim4 ot_gshape = {ot_N, ot_C, ot_H, ot_W};
    auto b_g = gtensor<T>(ot_gshape, GLOBAL, ptr_other);

    dim4 a_local_shape_block = {block_n, in_C, in_H , in_W};
    dim4 b_local_shape_block = {ot_N>1?block_n:ot_N, out_C, out_H , out_W};
    dim4 out_local_shape_block = {block_n, out_C, out_H , out_W};

    for (auto n_idx = 0; n_idx < slice_size_for_core; n_idx += block_n) {
        enable_pipeline();
        int n = min(block_n, slice_size_for_core - n_idx);
        dim4 out_local_shape = {n, out_C, out_H , out_W};
        dim4 a_local_shape = {n, in_C, in_H , in_W};
        dim4 b_local_shape = {ot_N>1?n:ot_N, out_C, out_H , out_W};
        dim4 a_offset = {n_idx+core_offset, 0, 0, 0};
        dim4 b_offset = {ot_N>1?(n_idx+core_offset):0, 0, 0, 0};

        auto a_l = make_tensor<T>(a_local_shape_block, a_local_shape);
        dma::load(a_l, a_g.sub_view(a_local_shape, a_offset));
        auto b_l = make_tensor<T>(b_local_shape_block, b_local_shape);
        dma::load(b_l, b_g.sub_view(b_local_shape, b_offset));
        auto o_l = make_tensor<U>(out_local_shape_block, out_local_shape);

        dim4 a_stride_tmp;
        get_stride<T>(&a_stride_tmp, &a_local_shape, TPU_ALIGN);
        int a_sn = (out_N > in_N) ? 0 : a_stride_tmp.n;
        int a_sc = (out_C > in_C) ? 0 : a_stride_tmp.c;
        int a_sh = (out_H > in_H) ? 0 : a_stride_tmp.h;
        int a_sw = (out_W > in_W) ? 0 : a_stride_tmp.w;
        dim4 a_stride = {a_sn, a_sc, a_sh, a_sw};

        dim4 b_stride_tmp;
        get_stride<T>(&b_stride_tmp, &b_local_shape, TPU_ALIGN);
        int b_sn = (out_N > ot_N) ? 0 : b_stride_tmp.n;
        int b_sc = (out_C > ot_C) ? 0 : b_stride_tmp.c;
        int b_sh = (out_H > ot_H) ? 0 : b_stride_tmp.h;
        int b_sw = (out_W > ot_W) ? 0 : b_stride_tmp.w;
        dim4 b_stride = {b_sn, b_sc, b_sh, b_sw};

        if (mode == 0)
            tiu::bitwise_xor(o_l, a_l.view(out_local_shape, a_stride), b_l.view(out_local_shape, b_stride));
        else if (mode == 1)
            tiu::bitwise_and(o_l, a_l.view(out_local_shape, a_stride), b_l.view(out_local_shape, b_stride));
        else if (mode == 2)
            tiu::bitwise_or(o_l, a_l.view(out_local_shape, a_stride), b_l.view(out_local_shape, b_stride));

        dma::store(dst_g.sub_view(out_local_shape, a_offset), o_l);
    }

}

__KERNEL__ void bitwise_bcast_int32(int32 *ptr_output, int32 *ptr_input, int32 *ptr_other,
           int mode, int outer_size, int inner_size, const int tile_size,
           int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
    bitwise_bcast_kernel<int32, int32>(ptr_output, ptr_input, ptr_other, inner_size, iN, iC, iH, iW,
                oN, oC, oH, oW, mode);
}

template <typename T, typename U>
void powc_const_kernel(U *ptr_output, T *ptr_input, double pow_C,
                        int outer_size, int inner_size, const int block_w) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();

    int C = outer_size;
    int W = inner_size;
    int block_c = LANE_NUM;
    int slice_per_core = div_up(outer_size, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, outer_size - core_offset);

    dim4 shape = {1, C, 1, W};
    auto res_gtensor = gtensor<U>(shape, GLOBAL, ptr_output);
    auto l_gtensor = gtensor<T>(shape, GLOBAL, ptr_input);

    dim4 block_shape = {1, block_c, 1, block_w};
    for (auto c_idx = 0; c_idx < slice_size_for_core; c_idx += block_c) {
        int c = min(block_c, slice_size_for_core - c_idx);
        for (auto w_idx = 0; w_idx < W; w_idx += block_w) {
            enable_pipeline();
            int w = min(block_w, W - w_idx);
            dim4 local_in_shape = {1, c, 1, w};
            dim4 input_offset = {0, core_offset + c_idx, 0, w_idx};
            auto left = make_tensor<T>(block_shape, local_in_shape);
            dma::load(left, l_gtensor.sub_view(local_in_shape, input_offset));

            auto res = make_tensor<U>(block_shape, local_in_shape);

            if constexpr (std::is_same<T, fp32>::value) {
                pow_f32(res, left, pow_C, &block_shape, &local_in_shape);
            } else if constexpr ( std::is_same<T, fp16>::value ||
                        std::is_same<T, bf16>::value) {
                auto left_fp32 = make_tensor<fp32>(block_shape, local_in_shape);
                tiu::cast(left_fp32, left);
                auto res_fp32 = make_tensor<fp32>(block_shape, local_in_shape);
                pow_f32(res_fp32, left_fp32, pow_C, &block_shape, &local_in_shape);
                tiu::cast(res, res_fp32);
            } else if constexpr (std::is_same<T, int32>::value) {
                auto left_fp32 = make_tensor<fp32>(block_shape, local_in_shape);
                tiu::cast(left_fp32, left);
                if constexpr (std::is_same<U, fp32>::value){
                    pow_f32(res, left_fp32, pow_C, &block_shape, &local_in_shape);
                } else if constexpr (std::is_same<U, int32>::value){
                    auto res_fp32 = make_tensor<fp32>(block_shape, local_in_shape);
                    pow_f32(res_fp32, left_fp32, pow_C, &block_shape, &local_in_shape);
                    tiu::cast(res, res_fp32);
                }
            }
            dma::store(res_gtensor.sub_view(local_in_shape, input_offset) , res);
        }
    }
}

template <typename T, typename U>
void cpow_const_kernel(U *ptr_output, T *ptr_input, double pow_C,
                        int outer_size, int inner_size, const int block_w) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();

    int C = outer_size;
    int W = inner_size;
    int block_c = LANE_NUM;
    int slice_per_core = div_up(outer_size, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, outer_size - core_offset);

    dim4 shape = {1, C, 1, W};
    auto res_gtensor = gtensor<U>(shape, GLOBAL, ptr_output);
    auto l_gtensor = gtensor<T>(shape, GLOBAL, ptr_input);

    dim4 block_shape = {1, block_c, 1, block_w};
    for (auto c_idx = 0; c_idx < slice_size_for_core; c_idx += block_c) {
        int c = min(block_c, slice_size_for_core - c_idx);
        for (auto w_idx = 0; w_idx < W; w_idx += block_w) {
            enable_pipeline();
            int w = min(block_w, W - w_idx);
            dim4 local_in_shape = {1, c, 1, w};
            dim4 input_offset = {0, core_offset + c_idx, 0, w_idx};
            auto left = make_tensor<T>(block_shape, local_in_shape);
            dma::load(left, l_gtensor.sub_view(local_in_shape, input_offset));

            auto res = make_tensor<U>(block_shape, local_in_shape);

            if constexpr (std::is_same<T, fp32>::value) {
                pow_f32(res, pow_C, left, &block_shape, &local_in_shape);
            } else if constexpr ( std::is_same<T, fp16>::value ||
                        std::is_same<T, bf16>::value || std::is_same<T, int32>::value) {
                auto left_fp32 = make_tensor<fp32>(block_shape, local_in_shape);
                tiu::cast(left_fp32, left);
                auto res_fp32 = make_tensor<fp32>(block_shape, local_in_shape);
                pow_f32(res_fp32, pow_C, left_fp32, &block_shape, &local_in_shape);
                tiu::cast(res, res_fp32);
            }
            else if constexpr (std::is_same<T, int32>::value) {
                auto left_fp32 = make_tensor<fp32>(block_shape, local_in_shape);
                tiu::cast(left_fp32, left);
                if constexpr (std::is_same<U, fp32>::value){
                    pow_f32(res, pow_C, left_fp32, &block_shape, &local_in_shape);
                } else if constexpr (std::is_same<U, int32>::value){
                    auto res_fp32 = make_tensor<fp32>(block_shape, local_in_shape);
                    pow_f32(res_fp32, pow_C, left_fp32, &block_shape, &local_in_shape);
                    tiu::cast(res, res_fp32);
                }
            }
            dma::store(res_gtensor.sub_view(local_in_shape, input_offset) , res);
        }
    }
}

__KERNEL__ void powc_const_fp32(fp32 *ptr_output, fp32 *ptr_input, double pow_c,
           int outer_size, int inner_size, const int tile_size) {
    powc_const_kernel<fp32, fp32>(ptr_output, ptr_input, pow_c, outer_size, inner_size, tile_size);
}

__KERNEL__ void powc_const_fp16(fp16 *ptr_output, fp16 *ptr_input, double pow_c,
           int outer_size, int inner_size, const int tile_size) {
    powc_const_kernel<fp16, fp16>(ptr_output, ptr_input, pow_c, outer_size, inner_size, tile_size);
}

__KERNEL__ void powc_const_bf16(bf16 *ptr_output, bf16 *ptr_input, double pow_c,
           int outer_size, int inner_size, const int tile_size) {
    powc_const_kernel<bf16, bf16>(ptr_output, ptr_input, pow_c, outer_size, inner_size, tile_size);
}

__KERNEL__ void powc_const_int32(fp32 *ptr_output, int32 *ptr_input, double pow_c,
           int outer_size, int inner_size, const int tile_size) {
    powc_const_kernel<int32, fp32>(ptr_output, ptr_input, pow_c, outer_size, inner_size, tile_size);
}

__KERNEL__ void powc_const_int32_int32(int32 *ptr_output, int32 *ptr_input, double pow_c,
           int outer_size, int inner_size, const int tile_size) {
    powc_const_kernel<int32, int32>(ptr_output, ptr_input, pow_c, outer_size, inner_size, tile_size);
}

__KERNEL__ void cpow_const_fp32(fp32 *ptr_output, fp32 *ptr_input, double pow_c,
           int outer_size, int inner_size, const int tile_size) {
    cpow_const_kernel<fp32, fp32>(ptr_output, ptr_input, pow_c, outer_size, inner_size, tile_size);
}

__KERNEL__ void cpow_const_fp16(fp16 *ptr_output, fp16 *ptr_input, double pow_c,
           int outer_size, int inner_size, const int tile_size) {
    cpow_const_kernel<fp16, fp16>(ptr_output, ptr_input, pow_c, outer_size, inner_size, tile_size);
}

__KERNEL__ void cpow_const_bf16(bf16 *ptr_output, bf16 *ptr_input, double pow_c,
           int outer_size, int inner_size, const int tile_size) {
    cpow_const_kernel<bf16, bf16>(ptr_output, ptr_input, pow_c, outer_size, inner_size, tile_size);
}

__KERNEL__ void cpow_const_int32(fp32 *ptr_output, int32 *ptr_input, double pow_c,
           int outer_size, int inner_size, const int tile_size) {
    cpow_const_kernel<int32, fp32>(ptr_output, ptr_input, pow_c, outer_size, inner_size, tile_size);
}

__KERNEL__ void cpow_const_int32_int32(int32 *ptr_output, int32 *ptr_input, double pow_c,
           int outer_size, int inner_size, const int tile_size) {
    cpow_const_kernel<int32, int32>(ptr_output, ptr_input, pow_c, outer_size, inner_size, tile_size);
}

template <typename T>
void pow_kernel(T *ptr_output, T *ptr_input, T *ptr_other, int mode, int outer_size, int inner_size, const int block_w) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();

    int C = outer_size;
    int W = inner_size;
    int block_c = LANE_NUM;
    int slice_per_core = div_up(outer_size, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, outer_size - core_offset);

    dim4 shape = {1, C, 1, W};
    auto res_gtensor = gtensor<T>(shape, GLOBAL, ptr_output);
    auto l_gtensor = gtensor<T>(shape, GLOBAL, ptr_input);
    auto r_gtensor = gtensor<T>(shape, GLOBAL, ptr_other);

    dim4 block_shape = {1, block_c, 1, block_w};
    for (auto c_idx = 0; c_idx < slice_size_for_core; c_idx += block_c) {
        int c = min(block_c, slice_size_for_core - c_idx);
        for (auto w_idx = 0; w_idx < W; w_idx += block_w) {
            enable_pipeline();
            int w = min(block_w, W - w_idx);
            dim4 local_in_shape = {1, c, 1, w};
            dim4 input_offset = {0, core_offset + c_idx, 0, w_idx};
            auto left = make_tensor<T>(block_shape, local_in_shape);
            dma::load(left, l_gtensor.sub_view(local_in_shape, input_offset));
            auto right = make_tensor<T>(block_shape, local_in_shape);
            dma::load(right, r_gtensor.sub_view(local_in_shape, input_offset));

            auto res = make_tensor<T>(block_shape, local_in_shape);
            if constexpr (std::is_same<T, fp32>::value) {
                pow_f32(res, left, right, &block_shape, &local_in_shape);
            } else if constexpr (std::is_same<T, int32>::value ||
                                    std::is_same<T, fp16>::value ||
                                    std::is_same<T, bf16>::value) {
                auto left_fp32 = make_tensor<fp32>(block_shape, local_in_shape);
                tiu::cast(left_fp32, left);
                auto right_fp32 = make_tensor<fp32>(block_shape, local_in_shape);
                tiu::cast(right_fp32, right);
                auto res_fp32 = make_tensor<fp32>(block_shape, local_in_shape);
                pow_f32(res_fp32, left_fp32, right_fp32, &block_shape, &local_in_shape);
                tiu::cast(res, res_fp32);
            }

            dma::store(res_gtensor.sub_view(local_in_shape, input_offset) , res);
        }
    }
}

__KERNEL__ void pow_forward_int32(int32 *ptr_output, int32 *ptr_input, int32 *ptr_other, int mode,
           int outer_size, int inner_size, const int tile_size) {
    pow_kernel<int32>(ptr_output, ptr_input, ptr_other, mode, outer_size, inner_size, tile_size);
}

__KERNEL__ void pow_forward_fp32(fp32 *ptr_output, fp32 *ptr_input, fp32 *ptr_other, int mode,
           int outer_size, int inner_size, const int tile_size) {
    pow_kernel<fp32>(ptr_output, ptr_input, ptr_other, mode, outer_size, inner_size, tile_size);
}

__KERNEL__ void pow_forward_fp16(fp16 *ptr_output, fp16 *ptr_input, fp16 *ptr_other, int mode,
           int outer_size, int inner_size, const int tile_size) {
    pow_kernel<fp16>(ptr_output, ptr_input, ptr_other, mode, outer_size, inner_size, tile_size);
}

__KERNEL__ void pow_forward_bf16(bf16 *ptr_output, bf16 *ptr_input, bf16 *ptr_other, int mode,
           int outer_size, int inner_size, const int tile_size) {
    pow_kernel<bf16>(ptr_output, ptr_input, ptr_other, mode, outer_size, inner_size, tile_size);
}

template <typename T, typename U>
void pow_bcast_kernel(U* ptr_output, T* ptr_input, T* ptr_other,
                        const int block_n,
                        int in_N, int in_C, int in_H, int in_W,
                        int ot_N, int ot_C, int ot_H, int ot_W,
                        int mode) {

    int out_N = max(in_N, ot_N);
    int out_C = max(in_C, ot_C);
    int out_H = max(in_H, ot_H);
    int out_W = max(in_W, ot_W);

    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();
    int slice_per_core = div_up(out_N, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, out_N - core_offset);

    dim4 in_gshape = {in_N , in_C, in_H , in_W};
    dim4 out_shape = {out_N , out_C, out_H , out_W};

    auto dst_g = gtensor<U>(out_shape, GLOBAL, ptr_output);
    auto a_g = gtensor<T>(in_gshape, GLOBAL, ptr_input);
    dim4 ot_gshape = {ot_N, ot_C, ot_H, ot_W};
    auto b_g = gtensor<T>(ot_gshape, GLOBAL, ptr_other);

    dim4 a_local_shape_block = {block_n, in_C, in_H , in_W};
    dim4 b_local_shape_block = {ot_N>1?block_n:ot_N, out_C, out_H , out_W};
    dim4 out_local_shape_block = {block_n, out_C, out_H , out_W};

    for (auto n_idx = 0; n_idx < slice_size_for_core; n_idx += block_n) {
        enable_pipeline();
        int n = min(block_n, slice_size_for_core - n_idx);
        dim4 out_local_shape = {n, out_C, out_H , out_W};
        dim4 a_local_shape = {n, in_C, in_H , in_W};
        dim4 b_local_shape = {ot_N>1?n:ot_N, out_C, out_H , out_W};
        dim4 a_offset = {n_idx+core_offset, 0, 0, 0};
        dim4 b_offset = {ot_N>1?(n_idx+core_offset):0, 0, 0, 0};

        auto a_l = make_tensor<T>(a_local_shape_block, a_local_shape);
        dma::load(a_l, a_g.sub_view(a_local_shape, a_offset));
        auto b_l = make_tensor<T>(b_local_shape_block, b_local_shape);
        dma::load(b_l, b_g.sub_view(b_local_shape, b_offset));
        auto o_l = make_tensor<U>(out_local_shape_block, out_local_shape);

        dim4 a_stride_tmp;
        get_stride<T>(&a_stride_tmp, &a_local_shape, TPU_ALIGN);
        int a_sn = (out_N > in_N) ? 0 : a_stride_tmp.n;
        int a_sc = (out_C > in_C) ? 0 : a_stride_tmp.c;
        int a_sh = (out_H > in_H) ? 0 : a_stride_tmp.h;
        int a_sw = (out_W > in_W) ? 0 : a_stride_tmp.w;
        dim4 a_stride = {a_sn, a_sc, a_sh, a_sw};

        dim4 b_stride_tmp;
        get_stride<T>(&b_stride_tmp, &b_local_shape, TPU_ALIGN);
        int b_sn = (out_N > ot_N) ? 0 : b_stride_tmp.n;
        int b_sc = (out_C > ot_C) ? 0 : b_stride_tmp.c;
        int b_sh = (out_H > ot_H) ? 0 : b_stride_tmp.h;
        int b_sw = (out_W > ot_W) ? 0 : b_stride_tmp.w;
        dim4 b_stride = {b_sn, b_sc, b_sh, b_sw};

        auto left_fp32 = make_tensor<fp32>(out_local_shape_block, out_local_shape);
        tiu::cast(left_fp32, a_l.view(out_local_shape, a_stride));
        auto right_fp32 = make_tensor<fp32>(out_local_shape_block, out_local_shape);
        tiu::cast(right_fp32, b_l.view(out_local_shape, b_stride));
        auto res_fp32 = make_tensor<fp32>(out_local_shape_block, out_local_shape);
        pow_f32(res_fp32, left_fp32, right_fp32, &out_local_shape_block, &out_local_shape);
        tiu::cast(o_l, res_fp32);

        dma::store(dst_g.sub_view(out_local_shape, a_offset), o_l);
    }

}

__KERNEL__ void pow_bcast_int32(int32 *ptr_output, int32 *ptr_input, int32 *ptr_other,
           int mode, int outer_size, int inner_size, const int tile_size,
           int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
    pow_bcast_kernel<int32, int32>(ptr_output, ptr_input, ptr_other, inner_size, iN, iC, iH, iW,
                oN, oC, oH, oW, mode);
}

__KERNEL__ void pow_bcast_fp32(fp32 *ptr_output, fp32 *ptr_input, fp32 *ptr_other,
           int mode, int outer_size, int inner_size, const int tile_size,
           int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
    pow_bcast_kernel<fp32, fp32>(ptr_output, ptr_input, ptr_other, inner_size, iN, iC, iH, iW,
                oN, oC, oH, oW, mode);
}

__KERNEL__ void pow_bcast_fp16(fp16 *ptr_output, fp16 *ptr_input, fp16 *ptr_other,
           int mode, int outer_size, int inner_size, const int tile_size,
           int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
    pow_bcast_kernel<fp16, fp16>(ptr_output, ptr_input, ptr_other, inner_size, iN, iC, iH, iW,
                oN, oC, oH, oW, mode);
}

__KERNEL__ void pow_bcast_bf16(bf16 *ptr_output, bf16 *ptr_input, bf16 *ptr_other,
           int mode, int outer_size, int inner_size, const int tile_size,
           int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
    pow_bcast_kernel<bf16, bf16>(ptr_output, ptr_input, ptr_other, inner_size, iN, iC, iH, iW,
                oN, oC, oH, oW, mode);
}

template <typename DataType>
void atan2_fp32(tensor<DataType> &out, tensor<DataType> &y, tensor<DataType> &x,
                dim4 *shape, dim4 *real_shape) {
    auto div_res = make_tensor<DataType>(shape, real_shape);
    auto abs_res = make_tensor<DataType>(shape, real_shape);
    auto work1 = make_tensor<DataType>(shape, real_shape);
    auto work2 = make_tensor<DataType>(shape, real_shape);
    auto work3 = make_tensor<DataType>(shape, real_shape);
    auto mask = make_tensor<DataType>(shape, real_shape);
    auto mask2 = make_tensor<DataType>(shape, real_shape);
    auto zeros = make_tensor<DataType>(shape, real_shape);
    auto ones = make_tensor<DataType>(shape, real_shape);
    auto pi = make_tensor<DataType>(shape, real_shape);
    auto half_pi = make_tensor<DataType>(shape, real_shape);
    auto neg_one = make_tensor<DataType>(shape, real_shape);
    auto neg_pi = make_tensor<DataType>(shape, real_shape);
    auto x_safe = make_tensor<DataType>(shape, real_shape);

    const DataType PI_VAL = static_cast<DataType>(3.14159265f);
    const DataType HALF_PI_VAL = static_cast<DataType>(PI_VAL / 2.0f);
    const DataType NEG_PI_VAL = static_cast<DataType>(-3.14159265f);
    const DataType ONE = static_cast<DataType>(1.0f);
    const DataType NEG_ONE = static_cast<DataType>(-1.0f);
    const DataType ZERO = static_cast<DataType>(0.0f);
    const DataType EPSILON = static_cast<DataType>(1e-10f);

    tiu::fill(zeros, static_cast<DataType>(0.0f));
    tiu::fill(ones, ONE);
    tiu::fill(neg_one, NEG_ONE);
    tiu::fill(pi, PI_VAL);
    tiu::fill(half_pi, HALF_PI_VAL);
    tiu::fill(neg_pi, NEG_PI_VAL);

    // Step 1:  y/x
    {
        tiu::eq(mask, x, zeros, ONE);           // mask = (x == 0) ? 1 : 0
        tiu::fmul(work1, mask, EPSILON);         // work1 = (x == 0) ? ε : 0
        tiu::fadd(x_safe, x, work1);             // x_safe = x + (x==0 ? ε : 0)
        tiu::fdiv(div_res, y, x_safe);
    }

    // Step 2: atan(y/x) using arcsin
    tiu::abs(abs_res, div_res);
    tiu::fmul(work1, abs_res, abs_res);
    tiu::fadd(work2, work1, ones);
    tiu::frsqrt(work3, work2);
    tiu::fmul(work1, abs_res, work3);
    farcsin(out, work1, shape, real_shape);

    // Step 3: special case handling
    // Case 1: x = 0 , out = (y < 0) ? -π/2: π/2
    {
        tiu::eq(mask, x, zeros, ONE);  // mask = (x == 0) ? 1 : 0
        tiu::fsub(mask2, ones, mask); // mask2 = (x == 0) ? 0 : 1
        tiu::fmul(out, out, mask2);    // out = out * (x != 0 ? 1 : 0)

        // Subcase 3.1: y > 0 ⇒ π/2
        {
            tiu::gt(work1, y, zeros, ONE);
            tiu::fmul(work2, mask, work1);
            tiu::fmul(work3, work2, half_pi);
            tiu::fadd(out, out, work3);
        }

        // Subcase 3.2: y < 0 ⇒ -π/2
        {
            tiu::lt(work1, y, zeros, ONE);
            tiu::fmul(work2, mask, work1);
            tiu::fmul(work3, work2, half_pi);
            tiu::fmul(work3, work3, NEG_ONE);
            tiu::fadd(out, out, work3);
        }
        // Subcase 3.3: y == 0 ⇒ 0
        {
            tiu::eq(work1, y, zeros, ONE);
            tiu::fmul(work2, mask, work1);
            tiu::fmul(work3, out, work2);
            tiu::fsub(out, out, work3);
        }
    }
    // Case 2: x > 0 mask = (y < 0 ? -mask : mask); x < 0 out =  (y < 0 ? -mask : mask) * -1
    {
        tiu::lt_select(mask, y, zeros, neg_one, ones);  // mask = (y < 0) ? -1 : 1
        tiu::lt_select(work1, x, zeros, neg_one, ones); // work1 = (x < 0) ? -1 : 1
        tiu ::fmul(mask, mask, work1);              // mask = (y < 0 ? -1 : 1) * (x < 0 ? -1 : 1)
        tiu::fmul(out, out, mask);                 // out * mask
    }
    // Case 3: x < 0 out = out + (y < 0 ? -π : π)
    {
        tiu::lt_select(mask, x, zeros, ones, zeros);  // mask = (x < 0) ? 1 : 0
        tiu::lt_select(work1, y, zeros, neg_pi, pi);  // work1 = (y < 0) ? -π : π
        tiu::fmul(work1, work1, mask);                 // only apply if x < 0
        tiu::fadd(out, out, work1);                   //(y >= 0 ? π : -π) * mask
    }
}

template <typename T, typename U>
void atan2_const_kernel(U *ptr_output, T *ptr_input, float scalar_C, bool mode,
                        int outer_size, int inner_size, const int block_w) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();

    int C = outer_size;
    int W = inner_size;
    int block_c = LANE_NUM;
    int slice_per_core = div_up(outer_size, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, outer_size - core_offset);

    dim4 shape = {1, C, 1, W};
    auto res_gtensor = gtensor<U>(shape, GLOBAL, ptr_output);
    auto l_gtensor = gtensor<T>(shape, GLOBAL, ptr_input);

    dim4 block_shape = {1, block_c, 1, block_w};
    for (auto c_idx = 0; c_idx < slice_size_for_core; c_idx += block_c) {
        int c = min(block_c, slice_size_for_core - c_idx);
        for (auto w_idx = 0; w_idx < W; w_idx += block_w) {
            enable_pipeline();
            int w = min(block_w, W - w_idx);
            dim4 local_in_shape = {1, c, 1, w};
            dim4 input_offset = {0, core_offset + c_idx, 0, w_idx};
            auto left = make_tensor<T>(block_shape, local_in_shape);
            dma::load(left, l_gtensor.sub_view(local_in_shape, input_offset));
            auto x_tensor = make_tensor<fp32>(block_shape, local_in_shape);
            tiu::fill(x_tensor, static_cast<fp32>(scalar_C));
            auto res = make_tensor<U>(block_shape, local_in_shape);
            atan2_fp32<fp32>(res, left, x_tensor, &block_shape, &local_in_shape);

            dma::store(res_gtensor.sub_view(local_in_shape, input_offset) , res);
        }
    }
}

template <typename T, typename U>
void const_atan2_kernel(U *ptr_output, T *ptr_input, float scalar_C, bool mode,
                        int outer_size, int inner_size, const int block_w) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();

    int C = outer_size;
    int W = inner_size;
    int block_c = LANE_NUM;
    int slice_per_core = div_up(outer_size, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, outer_size - core_offset);

    dim4 shape = {1, C, 1, W};
    auto res_gtensor = gtensor<U>(shape, GLOBAL, ptr_output);
    auto l_gtensor = gtensor<T>(shape, GLOBAL, ptr_input);

    dim4 block_shape = {1, block_c, 1, block_w};
    for (auto c_idx = 0; c_idx < slice_size_for_core; c_idx += block_c) {
        int c = min(block_c, slice_size_for_core - c_idx);
        for (auto w_idx = 0; w_idx < W; w_idx += block_w) {
            enable_pipeline();
            int w = min(block_w, W - w_idx);
            dim4 local_in_shape = {1, c, 1, w};
            dim4 input_offset = {0, core_offset + c_idx, 0, w_idx};
            auto left = make_tensor<T>(block_shape, local_in_shape);
            dma::load(left, l_gtensor.sub_view(local_in_shape, input_offset));
            auto y_tensor = make_tensor<fp32>(block_shape, local_in_shape);
            tiu::fill(y_tensor, static_cast<fp32>(scalar_C));
            auto res = make_tensor<U>(block_shape, local_in_shape);
            atan2_fp32<fp32>(res, y_tensor, left, &block_shape, &local_in_shape);

            dma::store(res_gtensor.sub_view(local_in_shape, input_offset) , res);
        }
    }
}

__KERNEL__ void atan2_const_fp32(fp32 *ptr_output, fp32 *ptr_input, float scalar_C, bool mode,
           int outer_size, int inner_size, const int tile_size) {
    atan2_const_kernel<fp32, fp32>(ptr_output, ptr_input, scalar_C, mode, outer_size, inner_size, tile_size);
}

__KERNEL__ void const_atan2_fp32(fp32 *ptr_output, fp32 *ptr_input, float scalar_C, bool mode,
           int outer_size, int inner_size, const int tile_size) {
    const_atan2_kernel<fp32, fp32>(ptr_output, ptr_input, scalar_C, mode, outer_size, inner_size, tile_size);
}

template <typename T>
void atan2_kernel(T *ptr_output, T *ptr_input, T *ptr_other, int mode, int outer_size, int inner_size, const int block_w) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();

    int C = outer_size;
    int W = inner_size;
    int block_c = LANE_NUM;
    int slice_per_core = div_up(outer_size, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, outer_size - core_offset);

    dim4 shape = {1, C, 1, W};
    auto res_gtensor = gtensor<T>(shape, GLOBAL, ptr_output);
    auto l_gtensor = gtensor<T>(shape, GLOBAL, ptr_input);
    auto r_gtensor = gtensor<T>(shape, GLOBAL, ptr_other);

    dim4 block_shape = {1, block_c, 1, block_w};
    for (auto c_idx = 0; c_idx < slice_size_for_core; c_idx += block_c) {
        int c = min(block_c, slice_size_for_core - c_idx);
        for (auto w_idx = 0; w_idx < W; w_idx += block_w) {
            enable_pipeline();
            int w = min(block_w, W - w_idx);
            dim4 local_in_shape = {1, c, 1, w};
            dim4 input_offset = {0, core_offset + c_idx, 0, w_idx};
            auto left = make_tensor<T>(block_shape, local_in_shape);
            dma::load(left, l_gtensor.sub_view(local_in_shape, input_offset));
            auto right = make_tensor<T>(block_shape, local_in_shape);
            dma::load(right, r_gtensor.sub_view(local_in_shape, input_offset));

            auto res = make_tensor<T>(block_shape, local_in_shape);
            atan2_fp32<fp32>(res, left, right, &block_shape, &local_in_shape);
            dma::store(res_gtensor.sub_view(local_in_shape, input_offset) , res);
        }
    }
}

__KERNEL__ void atan2_forward_fp32(fp32 *ptr_output, fp32 *ptr_input, fp32 *ptr_other, int mode,
           int outer_size, int inner_size, const int tile_size) {
    atan2_kernel<fp32>(ptr_output, ptr_input, ptr_other, mode, outer_size, inner_size, tile_size);
}

template <typename T, typename U>
void atan2_bcast_kernel(U* ptr_output, T* ptr_input, T* ptr_other,
                        const int block_n,
                        int in_N, int in_C, int in_H, int in_W,
                        int ot_N, int ot_C, int ot_H, int ot_W,
                        int mode) {

    int out_N = max(in_N, ot_N);
    int out_C = max(in_C, ot_C);
    int out_H = max(in_H, ot_H);
    int out_W = max(in_W, ot_W);

    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();
    int slice_per_core = div_up(out_N, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, out_N - core_offset);

    dim4 in_gshape = {in_N , in_C, in_H , in_W};
    dim4 out_shape = {out_N , out_C, out_H , out_W};

    auto dst_g = gtensor<U>(out_shape, GLOBAL, ptr_output);
    auto a_g = gtensor<T>(in_gshape, GLOBAL, ptr_input);
    dim4 ot_gshape = {ot_N, ot_C, ot_H, ot_W};
    auto b_g = gtensor<T>(ot_gshape, GLOBAL, ptr_other);

    dim4 a_local_shape_block = {block_n, in_C, in_H , in_W};
    dim4 b_local_shape_block = {ot_N>1?block_n:ot_N, out_C, out_H , out_W};
    dim4 out_local_shape_block = {block_n, out_C, out_H , out_W};

    for (auto n_idx = 0; n_idx < slice_size_for_core; n_idx += block_n) {
        enable_pipeline();
        int n = min(block_n, slice_size_for_core - n_idx);
        dim4 out_local_shape = {n, out_C, out_H , out_W};
        dim4 a_local_shape = {n, in_C, in_H , in_W};
        dim4 b_local_shape = {ot_N>1?n:ot_N, out_C, out_H , out_W};
        dim4 a_offset = {n_idx+core_offset, 0, 0, 0};
        dim4 b_offset = {ot_N>1?(n_idx+core_offset):0, 0, 0, 0};

        auto a_l = make_tensor<T>(a_local_shape_block, a_local_shape);
        dma::load(a_l, a_g.sub_view(a_local_shape, a_offset));
        auto b_l = make_tensor<T>(b_local_shape_block, b_local_shape);
        dma::load(b_l, b_g.sub_view(b_local_shape, b_offset));
        auto o_l = make_tensor<U>(out_local_shape_block, out_local_shape);

        dim4 a_stride_tmp;
        get_stride<T>(&a_stride_tmp, &a_local_shape, TPU_ALIGN);
        int a_sn = (out_N > in_N) ? 0 : a_stride_tmp.n;
        int a_sc = (out_C > in_C) ? 0 : a_stride_tmp.c;
        int a_sh = (out_H > in_H) ? 0 : a_stride_tmp.h;
        int a_sw = (out_W > in_W) ? 0 : a_stride_tmp.w;
        dim4 a_stride = {a_sn, a_sc, a_sh, a_sw};

        dim4 b_stride_tmp;
        get_stride<T>(&b_stride_tmp, &b_local_shape, TPU_ALIGN);
        int b_sn = (out_N > ot_N) ? 0 : b_stride_tmp.n;
        int b_sc = (out_C > ot_C) ? 0 : b_stride_tmp.c;
        int b_sh = (out_H > ot_H) ? 0 : b_stride_tmp.h;
        int b_sw = (out_W > ot_W) ? 0 : b_stride_tmp.w;
        dim4 b_stride = {b_sn, b_sc, b_sh, b_sw};

        auto a_l_bcast_view = a_l.view(out_local_shape, a_stride);
        auto b_l_bcast_view = b_l.view(out_local_shape, b_stride);
        auto a_l_aligned = make_tensor<T>(out_local_shape_block, out_local_shape);
        auto b_l_aligned = make_tensor<T>(out_local_shape_block, out_local_shape);
        tiu::cast(a_l_aligned, a_l_bcast_view);
        tiu::cast(b_l_aligned, b_l_bcast_view);
        atan2_fp32<fp32>(o_l, a_l_aligned, b_l_aligned, &out_local_shape_block, &out_local_shape);
        dma::store(dst_g.sub_view(out_local_shape, a_offset), o_l);
    }

}

__KERNEL__ void atan2_bcast_fp32(fp32 *ptr_output, fp32 *ptr_input, fp32 *ptr_other,
           int mode, int outer_size, int inner_size, const int tile_size,
           int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
    atan2_bcast_kernel<fp32, fp32>(ptr_output, ptr_input, ptr_other, inner_size, iN, iC, iH, iW,
                oN, oC, oH, oW, mode);
}