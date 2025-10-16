#include "ppl.h"
#include "ppl_wrapper_func.h"

using namespace ppl;

#ifdef __bm1690__
#define BLOCK_NUM 8
#elif defined __sg2260e__
#define BLOCK_NUM 4
#endif

int get_max_common_div(int v, int max_v) {
  for (int i = min(max_v, v); i > 0; i--) {
    if (v % i == 0) {
      return i;
    }
  }
  return 1;
}

template <typename T, typename U, bool DivMode>
void binary_const_kernel(U *ptr_dst, T *ptr_src, float rhs, int inner_size,
                  const int block_w, bool is_float, int binary_type, bool is_scalar_left, int relu = 0) {
  // reshape src [N, C, H, W] -> [1, C, 1, W]
  // c_slice <= LANE_NUM
  ppl::set_block_num(BLOCK_NUM);
  int core_num = get_block_num();
  int core_idx = get_block_index();

  int C = get_max_common_div(inner_size, LANE_NUM);
  int W = div_up(inner_size, C);

  int slice_per_core = div_up(C, core_num);
  int core_offset = slice_per_core * core_idx;
  int slice_size_for_core = min(slice_per_core, C - core_offset);


  int block_c = LANE_NUM;
  int block_w_iter = max(min(block_w, W / 2), 1);
  dim4 src_shape = {1, C, 1, W};
  dim4 src_block_shape = {1, block_c, 1, block_w};

  auto dst_gtensor = gtensor<U>(src_shape, GLOBAL, ptr_dst);
  auto src_gtensor = gtensor<T>(src_shape, GLOBAL, ptr_src);

  for (int idx_c = 0; idx_c < slice_size_for_core; idx_c += block_c) {
    int cur_c = min(block_c, slice_size_for_core - idx_c);
    for (int idx_w = 0; idx_w < W; idx_w += block_w_iter) {
      ppl::enable_pipeline();
      int cur_w = min(block_w_iter, W - idx_w);
      dim4 src_real_shape = {1, cur_c, 1, cur_w};
      auto in_tensor = make_tensor<T>(src_block_shape, src_real_shape);
      auto out_tensor = make_tensor<U>(src_block_shape, src_real_shape);
      dim4 offset = {0, core_offset + idx_c, 0, idx_w};
      dma::load(in_tensor, src_gtensor.sub_view(src_real_shape, offset));
      if constexpr (DivMode){ // div
        if constexpr (std::is_same<T, fp32>::value) {
            if (!is_scalar_left)
              tiu::fdiv(out_tensor, rhs, in_tensor);
            else
              tiu::fdiv(out_tensor, in_tensor, rhs);
        } else if constexpr (std::is_same<T, fp16>::value || std::is_same<T, bf16>::value) {
          auto in_tensor_fp32 = make_tensor<fp32>(src_block_shape, src_real_shape);
          tiu::cast(in_tensor_fp32, in_tensor, RM_HALF_TO_EVEN);
          auto out_tensor_fp32 = make_tensor<fp32>(src_block_shape, src_real_shape);
          if (!is_scalar_left)
            tiu::fdiv(out_tensor_fp32, rhs, in_tensor_fp32);
          else
            tiu::fdiv(out_tensor_fp32, in_tensor_fp32, rhs);
          tiu::cast(out_tensor, out_tensor_fp32, RM_HALF_TO_EVEN);
        } else if constexpr (std::is_same<T, int32_t>::value
                              || std::is_same<T, int16_t>::value
                              || std::is_same<T, int8_t>::value
                              || std::is_same<T, uint8_t>::value) {
          auto in_tensor_fp32 = make_tensor<fp32>(src_block_shape, src_real_shape);
          tiu::cast(in_tensor_fp32, in_tensor, RM_HALF_TO_EVEN);
          if (!is_scalar_left)
            tiu::fdiv(out_tensor, rhs, in_tensor_fp32);
          else
            tiu::fdiv(out_tensor, in_tensor_fp32, rhs);
        }
      } else { // add, sub, mul
        if (is_float) {
          if (binary_type == 0){
            tiu::fadd(out_tensor, in_tensor, rhs);
          } else if (binary_type == 1){
            if (is_scalar_left)
              tiu::fsub(out_tensor, in_tensor, rhs);
            else
              tiu::fsub(out_tensor, rhs, in_tensor);
          } else if (binary_type == 2){
            tiu::fmul(out_tensor, in_tensor, rhs);
          }
        } else {
          if (binary_type == 0){
            tiu::add(out_tensor, in_tensor, rhs, 0, RM_HALF_TO_EVEN, 0);
          } else if (binary_type == 1){
            if (is_scalar_left)
              tiu::sub(out_tensor, in_tensor, rhs, 0, RM_HALF_TO_EVEN, 0);
            else
              tiu::sub(out_tensor, rhs, in_tensor, 0, RM_HALF_TO_EVEN, 0);
          } else if (binary_type == 2){
            tiu::mul(out_tensor, in_tensor, rhs, 0, RM_HALF_TO_EVEN, 0);
          }
        }
      }
      if (relu) {
        tiu::max(out_tensor, out_tensor, 0.0f);
      }
      dma::store(dst_gtensor.sub_view(src_real_shape, offset), out_tensor);
    }
  }
}

__KERNEL__ void binary_async_fp32_scalar(fp32* ptr_output, fp32* ptr_input, int binary_type, const int block_w,
int inner_size, float alpha, bool is_scalar_left) {
  binary_const_kernel<fp32, fp32, 0>(ptr_output, ptr_input, alpha, inner_size,
  block_w, 1, binary_type, is_scalar_left);
}

__KERNEL__ void binary_async_fp16_scalar(fp16* ptr_output, fp16* ptr_input, int binary_type, const int block_w,
int inner_size, float alpha, bool is_scalar_left) {
  binary_const_kernel<fp16, fp16, 0>(ptr_output, ptr_input, alpha, inner_size,
  block_w, 1, binary_type, is_scalar_left);
}

__KERNEL__ void binary_async_bf16_scalar(bf16* ptr_output, bf16* ptr_input, int binary_type, const int block_w,
int inner_size, float alpha, bool is_scalar_left) {
  binary_const_kernel<bf16, bf16, 0>(ptr_output, ptr_input, alpha, inner_size,
  block_w, 1, binary_type, is_scalar_left);
}

__KERNEL__ void binary_div_fp32_scalar(fp32* ptr_output, fp32* ptr_input, int binary_type, const int block_w,
int inner_size, float alpha, bool is_scalar_left) {
  binary_const_kernel<fp32, fp32, 1>(ptr_output, ptr_input, alpha, inner_size,
  block_w, 1, binary_type, is_scalar_left);
}

__KERNEL__ void binary_div_fp16_scalar(fp16* ptr_output, fp16* ptr_input, int binary_type, const int block_w,
int inner_size, float alpha, bool is_scalar_left) {
  binary_const_kernel<fp16, fp16, 1>(ptr_output, ptr_input, alpha, inner_size,
  block_w, 1, binary_type, is_scalar_left);
}

__KERNEL__ void binary_div_bf16_scalar(bf16* ptr_output, bf16* ptr_input, int binary_type, const int block_w,
int inner_size, float alpha, bool is_scalar_left) {
  binary_const_kernel<bf16, bf16, 1>(ptr_output, ptr_input, alpha, inner_size,
  block_w, 1, binary_type, is_scalar_left);
}

__KERNEL__ void binary_async_int32_scalar(int32* ptr_output, int32* ptr_input, int binary_type, const int block_w,
int inner_size, float alpha, bool is_scalar_left) {
  binary_const_kernel<int32, int32, 0>(ptr_output, ptr_input, alpha, inner_size,
    block_w, 0, binary_type, is_scalar_left);
}

__KERNEL__ void div_int32_fp32_scalar(fp32* ptr_output, int32* ptr_input, int binary_type, const int block_w,
int inner_size, float alpha, bool is_scalar_left) {
  binary_const_kernel<int32, fp32, 1>(ptr_output, ptr_input, alpha, inner_size,
    block_w, 0, binary_type, is_scalar_left);
}

__KERNEL__ void binary_async_int16_scalar(int16* ptr_output, int16* ptr_input, int binary_type, const int block_w,
int inner_size, float alpha, bool is_scalar_left) {
  binary_const_kernel<int16, int16, 0>(ptr_output, ptr_input, alpha, inner_size,
    block_w, 0, binary_type, is_scalar_left);
}

__KERNEL__ void div_int16_fp32_scalar(fp32* ptr_output, int16* ptr_input, int binary_type, const int block_w,
int inner_size, float alpha, bool is_scalar_left) {
  binary_const_kernel<int16, fp32, 1>(ptr_output, ptr_input, alpha, inner_size,
    block_w, 0, binary_type, is_scalar_left);
}

__KERNEL__ void binary_async_int8_scalar(int8* ptr_output, int8* ptr_input, int binary_type, const int block_w,
int inner_size, float alpha, bool is_scalar_left) {
  binary_const_kernel<int8, int8, 0>(ptr_output, ptr_input, alpha, inner_size,
    block_w, 0, binary_type, is_scalar_left);
}

__KERNEL__ void div_int8_fp32_scalar(fp32* ptr_output, int8* ptr_input, int binary_type, const int block_w,
int inner_size, float alpha, bool is_scalar_left) {
  binary_const_kernel<int8, fp32, 1>(ptr_output, ptr_input, alpha, inner_size,
    block_w, 0, binary_type, is_scalar_left);
}

__KERNEL__ void binary_async_uint8_scalar(uint8* ptr_output, uint8* ptr_input, int binary_type, const int block_w,
int inner_size, float alpha, bool is_scalar_left) {
  binary_const_kernel<uint8, uint8, 0>(ptr_output, ptr_input, alpha, inner_size,
    block_w, 0, binary_type, is_scalar_left);
}

__KERNEL__ void div_uint8_fp32_scalar(fp32* ptr_output, uint8* ptr_input, int binary_type, const int block_w,
int inner_size, float alpha, bool is_scalar_left) {
  binary_const_kernel<uint8, fp32, 1>(ptr_output, ptr_input, alpha, inner_size,
    block_w, 0, binary_type, is_scalar_left);
}

template <typename T, typename U, bool DivMode>
void binary_async_kernel(U* ptr_output, T* ptr_input, T* ptr_other,
                        int binary_type, const int block_w,
                        int outer_size, int inner_size,
                        int is_float){
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

          auto out_tensor = make_tensor<U>(block_shape, local_in_shape);
          if constexpr (DivMode){ // div
            if constexpr (std::is_same<T, fp32>::value) {
                tiu::fdiv(out_tensor, left, right);
            } else if constexpr (std::is_same<T, fp16>::value || std::is_same<T, bf16>::value) {
              auto left_fp32 = make_tensor<fp32>(block_shape, local_in_shape);
              tiu::cast(left_fp32, left, RM_HALF_TO_EVEN);
              auto right_fp32 = make_tensor<fp32>(block_shape, local_in_shape);
              tiu::cast(right_fp32, right, RM_HALF_TO_EVEN);
              auto out_tensor_fp32 = make_tensor<fp32>(block_shape, local_in_shape);
              tiu::fdiv(out_tensor_fp32, left_fp32, right_fp32);
              tiu::cast(out_tensor, out_tensor_fp32, RM_HALF_TO_EVEN);
            } else if constexpr (std::is_same<T, int32_t>::value
                                  || std::is_same<T, int16_t>::value
                                  || std::is_same<T, int8_t>::value
                                  || std::is_same<T, uint8_t>::value) {
              auto left_fp32 = make_tensor<fp32>(block_shape, local_in_shape);
              tiu::cast(left_fp32, left, RM_HALF_TO_EVEN);
              auto right_fp32 = make_tensor<fp32>(block_shape, local_in_shape);
              tiu::cast(right_fp32, right, RM_HALF_TO_EVEN);
              tiu::fdiv(out_tensor, left_fp32, right_fp32);
            }
          } else { // add, sub, mul
            if (is_float) {
              if (binary_type == 0){
                tiu::fadd(out_tensor, left, right);
              } else if (binary_type == 1){
                tiu::fsub(out_tensor, left, right);
              } else if (binary_type == 2){
                tiu::fmul(out_tensor, left, right);
              }
            } else {
              if (binary_type == 0){
                tiu::add(out_tensor, left, right, 0, RM_HALF_TO_EVEN, 0);
              } else if (binary_type == 1){
                tiu::sub(out_tensor, left, right, 0, RM_HALF_TO_EVEN, 0);
              } else if (binary_type == 2){
                tiu::mul(out_tensor, left, right, 0, RM_HALF_TO_EVEN, 0);
              }
            }
          }
          dma::store(res_gtensor.sub_view(local_in_shape, input_offset), out_tensor);
        }
    }
}

template <typename T, typename U, bool DivMode>
void binary_bcast_kernel(U* ptr_output, T* ptr_input, T* ptr_other,
                        int binary_type, const int block_n,
                        int in_N, int in_C, int in_H, int in_W,
                        int ot_N, int ot_C, int ot_H, int ot_W,
                        int is_float) {
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

    dim4 in_gshape = {in_N, in_C, in_H, in_W};
    dim4 out_shape = {out_N, out_C, out_H, out_W};

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
        if constexpr (DivMode){
          get_stride<fp32>(&a_stride_tmp, &a_local_shape, TPU_ALIGN);
        } else {
          get_stride<T>(&a_stride_tmp, &a_local_shape, TPU_ALIGN);
        }
        int a_sn = (out_N > in_N) ? 0 : a_stride_tmp.n;
        int a_sc = (out_C > in_C) ? 0 : a_stride_tmp.c;
        int a_sh = (out_H > in_H) ? 0 : a_stride_tmp.h;
        int a_sw = (out_W > in_W) ? 0 : a_stride_tmp.w;
        dim4 a_stride = {a_sn, a_sc, a_sh, a_sw};

        dim4 b_stride_tmp;
        if constexpr (DivMode){
          get_stride<fp32>(&b_stride_tmp, &b_local_shape, TPU_ALIGN);
        } else {
          get_stride<T>(&b_stride_tmp, &b_local_shape, TPU_ALIGN);
        }
        int b_sn = (out_N > ot_N) ? 0 : b_stride_tmp.n;
        int b_sc = (out_C > ot_C) ? 0 : b_stride_tmp.c;
        int b_sh = (out_H > ot_H) ? 0 : b_stride_tmp.h;
        int b_sw = (out_W > ot_W) ? 0 : b_stride_tmp.w;
        dim4 b_stride = {b_sn, b_sc, b_sh, b_sw};
        if constexpr (DivMode){ // div
          if constexpr (std::is_same<T, fp32>::value) {
              tiu::fdiv(o_l, a_l.view(out_local_shape, a_stride), b_l.view(out_local_shape, b_stride));
          } else if constexpr (std::is_same<T, fp16>::value || std::is_same<T, bf16>::value) {
            auto left_fp32 = make_tensor<fp32>(a_local_shape_block, a_local_shape);
            tiu::cast(left_fp32, a_l);
            auto right_fp32 = make_tensor<fp32>(b_local_shape_block, b_local_shape);
            tiu::cast(right_fp32, b_l);
            auto out_tensor_fp32 = make_tensor<fp32>(out_local_shape_block, out_local_shape);
            tiu::fdiv(out_tensor_fp32, left_fp32.view(out_local_shape, a_stride), right_fp32.view(out_local_shape, b_stride));
            tiu::cast(o_l, out_tensor_fp32);
          } else if constexpr (std::is_same<T, int32_t>::value
                                || std::is_same<T, int16_t>::value
                                || std::is_same<T, int8_t>::value
                                || std::is_same<T, uint8_t>::value) {
            auto left_fp32 = make_tensor<fp32>(a_local_shape_block, a_local_shape);
            tiu::cast(left_fp32, a_l);
            auto right_fp32 = make_tensor<fp32>(b_local_shape_block, b_local_shape);
            tiu::cast(right_fp32, b_l);
            tiu::fdiv(o_l, left_fp32.view(out_local_shape, a_stride), right_fp32.view(out_local_shape, b_stride));
          }
        } else { // add, sub, mul
          if (is_float) {
            if (binary_type == 0){
              tiu::fadd(o_l, a_l.view(out_local_shape, a_stride), b_l.view(out_local_shape, b_stride));
            } else if (binary_type == 1){
              tiu::fsub(o_l, a_l.view(out_local_shape, a_stride), b_l.view(out_local_shape, b_stride));
            } else if (binary_type == 2){
              tiu::fmul(o_l, a_l.view(out_local_shape, a_stride), b_l.view(out_local_shape, b_stride));
            }
          } else {
            if (binary_type == 0){
              tiu::add(o_l, a_l.view(out_local_shape, a_stride), b_l.view(out_local_shape, b_stride), 0, RM_HALF_TO_EVEN, 0);
            } else if (binary_type == 1){
              tiu::sub(o_l, a_l.view(out_local_shape, a_stride), b_l.view(out_local_shape, b_stride), 0, RM_HALF_TO_EVEN, 0);
            } else if (binary_type == 2){
              tiu::mul(o_l, a_l.view(out_local_shape, a_stride), b_l.view(out_local_shape, b_stride), 0, RM_HALF_TO_EVEN, 0);
            }
          }
        }
        dma::store(dst_g.sub_view(out_local_shape, a_offset), o_l);
    }
}

__KERNEL__ void binary_async_fp32(fp32* ptr_output, fp32* ptr_input, fp32* ptr_other,
    int binary_type, const int block_w, int outer_size, int inner_size) {
  binary_async_kernel<fp32, fp32, 0>(ptr_output, ptr_input, ptr_other,
                      binary_type, block_w,
                      outer_size, inner_size, 1);
}

__KERNEL__ void binary_async_fp16(fp16* ptr_output, fp16* ptr_input, fp16* ptr_other,
    int binary_type, const int block_w, int outer_size, int inner_size) {
  binary_async_kernel<fp16, fp16, 0>(ptr_output, ptr_input, ptr_other,
                      binary_type, block_w,
                      outer_size, inner_size, 1);
}

__KERNEL__ void binary_async_bf16(bf16* ptr_output, bf16* ptr_input, bf16* ptr_other,
    int binary_type, const int block_w, int outer_size, int inner_size) {
  binary_async_kernel<bf16, bf16, 0>(ptr_output, ptr_input, ptr_other,
                      binary_type, block_w,
                      outer_size, inner_size, 1);
}

__KERNEL__ void binary_div_fp32(fp32* ptr_output, fp32* ptr_input, fp32* ptr_other,
    int binary_type, const int block_w, int outer_size, int inner_size) {
  binary_async_kernel<fp32, fp32, 1>(ptr_output, ptr_input, ptr_other, 3, block_w,
                      outer_size, inner_size, 1);
}

__KERNEL__ void binary_div_fp16(fp16* ptr_output, fp16* ptr_input, fp16* ptr_other,
    int binary_type, const int block_w, int outer_size, int inner_size) {
  binary_async_kernel<fp16, fp16, 1>(ptr_output, ptr_input, ptr_other, 3, block_w,
                      outer_size, inner_size, 1);
}

__KERNEL__ void binary_div_bf16(bf16* ptr_output, bf16* ptr_input, bf16* ptr_other,
    int binary_type, const int block_w, int outer_size, int inner_size) {
  binary_async_kernel<bf16, bf16, 1>(ptr_output, ptr_input, ptr_other, 3, block_w,
                      outer_size, inner_size, 1);
}

__KERNEL__ void binary_async_int32(int32_t* ptr_output, int32_t* ptr_input, int32_t* ptr_other,
    int binary_type, const int block_w, int outer_size, int inner_size) {
    binary_async_kernel<int32_t, int32_t, 0>(ptr_output, ptr_input, ptr_other,
                    binary_type, block_w,
                    outer_size, inner_size, 0);

}

__KERNEL__ void div_int32_fp32(fp32* ptr_output, int32_t* ptr_input, int32_t* ptr_other,
    const int block_w, int outer_size, int inner_size) {
    binary_async_kernel<int32_t, fp32, 1>(ptr_output, ptr_input, ptr_other,
                    3, block_w, outer_size, inner_size, 0);
}

__KERNEL__ void binary_async_int16(int16_t* ptr_output, int16_t* ptr_input, int16_t* ptr_other,
    int binary_type, const int block_w, int outer_size, int inner_size) {
    binary_async_kernel<int16_t, int16_t, 0>(ptr_output, ptr_input, ptr_other,
                              binary_type, block_w, outer_size, inner_size, 0);
}

__KERNEL__ void div_int16_fp32(fp32* ptr_output, int16_t* ptr_input, int16_t* ptr_other,
    const int block_w, int outer_size, int inner_size) {
    binary_async_kernel<int16_t, fp32, 1>(ptr_output, ptr_input, ptr_other,
                              3, block_w, outer_size, inner_size, 0);
}

__KERNEL__ void binary_async_int8(int8_t* ptr_output, int8_t* ptr_input, int8_t* ptr_other,
    int binary_type, const int block_w, int outer_size, int inner_size) {
    binary_async_kernel<int8_t, int8_t, 0>(ptr_output, ptr_input, ptr_other,
                      binary_type, block_w, outer_size, inner_size, 0);
}

__KERNEL__ void div_int8_fp32(fp32* ptr_output, int8_t* ptr_input, int8_t* ptr_other,
    const int block_w, int outer_size, int inner_size) {
    binary_async_kernel<int8_t, fp32, 1>(ptr_output, ptr_input, ptr_other,
                      3, block_w, outer_size, inner_size, 0);
}

__KERNEL__ void binary_async_uint8(uint8_t* ptr_output, uint8_t* ptr_input, uint8_t* ptr_other,
    int binary_type, const int block_w, int outer_size, int inner_size) {
  binary_async_kernel<uint8_t, uint8_t, 0>(ptr_output, ptr_input, ptr_other,
                          binary_type, block_w, outer_size, inner_size, 0);
}

__KERNEL__ void div_uint8_fp32(fp32* ptr_output, uint8_t* ptr_input, uint8_t* ptr_other,
    const int block_w, int outer_size, int inner_size) {
  binary_async_kernel<uint8_t, fp32, 1>(ptr_output, ptr_input, ptr_other,
                          3, block_w, outer_size, inner_size, 0);
}

__KERNEL__ void binary_bcast_fp32(fp32* ptr_output, fp32* ptr_input, fp32* ptr_other,
    int binary_type, const int block_w, int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
  binary_bcast_kernel<fp32, fp32, 0>(ptr_output, ptr_input, ptr_other,
                      binary_type, block_w, iN, iC, iH, iW, oN, oC, oH, oW, 1);
}

__KERNEL__ void binary_bcast_fp16(fp16* ptr_output, fp16* ptr_input, fp16* ptr_other,
    int binary_type, const int block_w, int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
  binary_bcast_kernel<fp16, fp16, 0>(ptr_output, ptr_input, ptr_other,
                      binary_type, block_w,
                      iN, iC, iH, iW, oN, oC, oH, oW, 1);
}

__KERNEL__ void binary_bcast_bf16(bf16* ptr_output, bf16* ptr_input, bf16* ptr_other,
    int binary_type, const int block_w, int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
  binary_bcast_kernel<bf16, bf16, 0>(ptr_output, ptr_input, ptr_other,
                      binary_type, block_w,
                      iN, iC, iH, iW, oN, oC, oH, oW, 1);
}

__KERNEL__ void binary_div_bcast_fp32(fp32* ptr_output, fp32* ptr_input, fp32* ptr_other,
    int binary_type, const int block_w, int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
  binary_bcast_kernel<fp32, fp32, 1>(ptr_output, ptr_input, ptr_other, 3, block_w,
                                      iN, iC, iH, iW, oN, oC, oH, oW, 1);
}

__KERNEL__ void binary_div_bcast_fp16(fp16* ptr_output, fp16* ptr_input, fp16* ptr_other,
    int binary_type, const int block_w, int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
  binary_bcast_kernel<fp16, fp16, 1>(ptr_output, ptr_input, ptr_other, 3, block_w,
                      iN, iC, iH, iW, oN, oC, oH, oW, 1);
}

__KERNEL__ void binary_div_bcast_bf16(bf16* ptr_output, bf16* ptr_input, bf16* ptr_other,
    int binary_type, const int block_w, int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
  binary_bcast_kernel<bf16, bf16, 1>(ptr_output, ptr_input, ptr_other, 3, block_w,
                      iN, iC, iH, iW, oN, oC, oH, oW, 1);
}

__KERNEL__ void binary_bcast_int32(int32_t* ptr_output, int32_t* ptr_input, int32_t* ptr_other,
    int binary_type, const int block_w, int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
  binary_bcast_kernel<int32_t, int32_t, 0>(ptr_output, ptr_input, ptr_other,
                  binary_type, block_w,
                  iN, iC, iH, iW, oN, oC, oH, oW, 0);

}

__KERNEL__ void div_bcast_int32_fp32(fp32* ptr_output, int32_t* ptr_input, int32_t* ptr_other,
    const int block_w, int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
    binary_bcast_kernel<int32_t, fp32, 1>(ptr_output, ptr_input, ptr_other,
                    3, block_w,
                    iN, iC, iH, iW, oN, oC, oH, oW, 0);
}

__KERNEL__ void binary_bcast_int16(int16_t* ptr_output, int16_t* ptr_input, int16_t* ptr_other,
    int binary_type, const int block_w, int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
    binary_bcast_kernel<int16_t, int16_t, 0>(ptr_output, ptr_input, ptr_other,
                              binary_type, block_w,
                              iN, iC, iH, iW, oN, oC, oH, oW, 0);
}

__KERNEL__ void div_bcast_int16_fp32(fp32* ptr_output, int16_t* ptr_input, int16_t* ptr_other,
    const int block_w, int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
    binary_bcast_kernel<int16_t, fp32, 1>(ptr_output, ptr_input, ptr_other,
                              3, block_w,
                              iN, iC, iH, iW, oN, oC, oH, oW, 0);
}

__KERNEL__ void binary_bcast_int8(int8_t* ptr_output, int8_t* ptr_input, int8_t* ptr_other,
    int binary_type, const int block_w, int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
    binary_bcast_kernel<int8_t, int8_t, 0>(ptr_output, ptr_input, ptr_other,
                      binary_type, block_w,
                      iN, iC, iH, iW, oN, oC, oH, oW, 0);
}

__KERNEL__ void div_bcast_int8_fp32(fp32* ptr_output, int8_t* ptr_input, int8_t* ptr_other,
    const int block_w, int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
    binary_bcast_kernel<int8_t, fp32, 1>(ptr_output, ptr_input, ptr_other,
                      3, block_w,
                      iN, iC, iH, iW, oN, oC, oH, oW, 0);
}

__KERNEL__ void binary_bcast_uint8(uint8_t* ptr_output, uint8_t* ptr_input, uint8_t* ptr_other,
    int binary_type, const int block_w, int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
  binary_bcast_kernel<uint8_t, uint8_t, 0>(ptr_output, ptr_input, ptr_other,
                          binary_type, block_w, iN, iC, iH, iW, oN, oC, oH, oW, 0);
}

__KERNEL__ void div_bcast_uint8_fp32(fp32* ptr_output, uint8_t* ptr_input, uint8_t* ptr_other,
    const int block_w, int iN, int iC, int iH, int iW, int oN, int oC, int oH, int oW) {
  binary_bcast_kernel<uint8_t, fp32, 1>(ptr_output, ptr_input, ptr_other,
                          3, block_w, iN, iC, iH, iW, oN, oC, oH, oW, 0);
}
