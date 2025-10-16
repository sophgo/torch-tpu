#include "ppl.h"
#include <vector>

using namespace ppl;

#ifdef __bm1690__
#define BLOCK_NUM 8
#elif defined __sg2260e__
#define BLOCK_NUM 4
#endif

template <typename T>
void gdmad2d_kernel(T *ptr_output, T *ptr_input, const int outer_size, const int block_w) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();
    int slice_per_core = div_up(outer_size, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, outer_size - core_offset);

    dim4 shape = {1, 1, 1, outer_size};
    auto in_gt = gtensor<T>(shape, GLOBAL, ptr_input);
    auto out_gt = gtensor<T>(shape, GLOBAL, ptr_output);

    for (auto w_idx = 0; w_idx < slice_size_for_core; w_idx += block_w) {
        int w = min(block_w, slice_size_for_core - w_idx);
        dim4 input_global_offset = {0, 0, 0, core_offset + w_idx};
        dim4 local_shape = {1, 1, 1, w};

        dma::move(out_gt.sub_view(local_shape, input_global_offset),
                  in_gt.sub_view(local_shape, input_global_offset));
    }
}

__KERNEL__ void gdmad2d_fp32(fp32 *ptr_output, fp32 *ptr_input, const int outer_size, const int block_w) {
  gdmad2d_kernel<fp32>(ptr_output, ptr_input, outer_size, block_w);
}
__KERNEL__ void gdmad2d_fp16(fp16 *ptr_output, fp16 *ptr_input, const int outer_size, const int block_w) {
  gdmad2d_kernel<fp16>(ptr_output, ptr_input, outer_size, block_w);
}
__KERNEL__ void gdmad2d_bf16(bf16 *ptr_output, bf16 *ptr_input, const int outer_size, const int block_w) {
  gdmad2d_kernel<bf16>(ptr_output, ptr_input, outer_size, block_w);
}
__KERNEL__ void gdmad2d_int32(int32 *ptr_output, int32 *ptr_input, const int outer_size, const int block_w) {
  gdmad2d_kernel<int32>(ptr_output, ptr_input, outer_size, block_w);
}
__KERNEL__ void gdmad2d_int16(int16 *ptr_output, int16 *ptr_input, const int outer_size, const int block_w) {
  gdmad2d_kernel<int16>(ptr_output, ptr_input, outer_size, block_w);
}
__KERNEL__ void gdmad2d_int8(int8 *ptr_output, int8 *ptr_input, const int outer_size, const int block_w) {
  gdmad2d_kernel<int8>(ptr_output, ptr_input, outer_size, block_w);
}
__KERNEL__ void gdmad2d_uint8(uint8 *ptr_output, uint8 *ptr_input, const int outer_size, const int block_w) {
  gdmad2d_kernel<uint8>(ptr_output, ptr_input, outer_size, block_w);
}

template <typename T>
void stridedcopy_move_kernel_singlecore(T *ptr_output, T *ptr_intput, const int shape0, const int shape1, const int shape2, const int shape3,
                          const int src_stride0, const int src_stride1, const int src_stride2, const int src_stride3,
                          const int dst_stride0, const int dst_stride1, const int dst_stride2, const int dst_stride3) {

    dim4 shape = {shape0, shape1, shape2, shape3};
    dim4 src_stride = {src_stride0, src_stride1, src_stride2, src_stride3};
    dim4 dst_stride = {dst_stride0, dst_stride1, dst_stride2, dst_stride3};

    auto in_gt = gtensor<T>(shape, GLOBAL, ptr_intput);
    auto in_g = in_gt.view(shape, src_stride);
    auto out_gt = gtensor<T>(shape, GLOBAL, ptr_output);
    auto out_g = out_gt.view(shape, dst_stride);
    dma::move(out_g, in_g);
}

template <typename T>
void stridedcopy_move_kernel(T *ptr_output, T *ptr_input,
                          const int shape0, const int shape1, const int shape2, const int shape3,
                          const int src_stride0, const int src_stride1, const int src_stride2, const int src_stride3,
                          const int dst_stride0, const int dst_stride1, const int dst_stride2, const int dst_stride3,
                          const int block_n) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();
    int slice_per_core = div_up(shape0, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, shape0 - core_offset);

    dim4 shape = {shape0, shape1, shape2, shape3};
    dim4 src_stride = {src_stride0, src_stride1, src_stride2, src_stride3};
    dim4 dst_stride = {dst_stride0, dst_stride1, dst_stride2, dst_stride3};

    auto in_gt = gtensor<T>(shape, GLOBAL, ptr_input);
    auto in_g = in_gt.view(shape, src_stride);
    auto out_gt = gtensor<T>(shape, GLOBAL, ptr_output);
    auto out_g = out_gt.view(shape, dst_stride);

    for (auto n_idx = 0; n_idx < slice_size_for_core; n_idx += block_n) {
        int n = min(block_n, slice_size_for_core - n_idx);
        dim4 input_global_offset = {core_offset + n_idx, 0, 0, 0};
        dim4 local_shape = {n, shape1, shape2, shape3};

        dma::move(out_g.sub_view(local_shape, input_global_offset),
                  in_g.sub_view(local_shape, input_global_offset));
    }
}


__KERNEL__ void stridedcopy_fp32(fp32 *ptr_output, fp32 *ptr_input,
    const int shape0, const int shape1, const int shape2, const int shape3,
    const int shape4,
    const int src_stride0, const int src_stride1, const int src_stride2, const int src_stride3,
    const int src_stride4,
    const int dst_stride0, const int dst_stride1, const int dst_stride2, const int dst_stride3,
    const int dst_stride4,
    int dim, const int block_w)
{
  if (dim <= 4){
    stridedcopy_move_kernel<fp32>(ptr_output, ptr_input,
            shape0, shape1, shape2, shape3,
            src_stride0, src_stride1, src_stride2, src_stride3,
            dst_stride0, dst_stride1, dst_stride2, dst_stride3, block_w);
  } else {
    stridedcopy_move_kernel<fp32>(
            ptr_output, ptr_input,
            shape1, shape2, shape3, shape4,
            src_stride1, src_stride2, src_stride3, src_stride4,
            dst_stride1, dst_stride2, dst_stride3, dst_stride4, block_w);
  }
}

__KERNEL__ void stridedcopy_fp16(fp16 *ptr_output, fp16 *ptr_input,
    const int shape0, const int shape1, const int shape2, const int shape3,
    const int shape4,
    const int src_stride0, const int src_stride1, const int src_stride2, const int src_stride3,
    const int src_stride4,
    const int dst_stride0, const int dst_stride1, const int dst_stride2, const int dst_stride3,
    const int dst_stride4,
    int dim, const int block_w)
{
  if (dim <= 4){
    stridedcopy_move_kernel<fp16>(ptr_output, ptr_input,
            shape0, shape1, shape2, shape3,
            src_stride0, src_stride1, src_stride2, src_stride3,
            dst_stride0, dst_stride1, dst_stride2, dst_stride3, block_w);
  } else {
    stridedcopy_move_kernel<fp16>(
            ptr_output, ptr_input,
            shape1, shape2, shape3, shape4,
            src_stride1, src_stride2, src_stride3, src_stride4,
            dst_stride1, dst_stride2, dst_stride3, dst_stride4, block_w);
  }
}

__KERNEL__ void stridedcopy_bf16(bf16 *ptr_output, bf16 *ptr_input,
    const int shape0, const int shape1, const int shape2, const int shape3,
    const int shape4,
    const int src_stride0, const int src_stride1, const int src_stride2, const int src_stride3,
    const int src_stride4,
    const int dst_stride0, const int dst_stride1, const int dst_stride2, const int dst_stride3,
    const int dst_stride4,
    int dim, const int block_w)
{
  if (dim <= 4){
    stridedcopy_move_kernel<bf16>(ptr_output, ptr_input,
            shape0, shape1, shape2, shape3,
            src_stride0, src_stride1, src_stride2, src_stride3,
            dst_stride0, dst_stride1, dst_stride2, dst_stride3, block_w);
  } else {
    stridedcopy_move_kernel<bf16>(
            ptr_output, ptr_input,
            shape1, shape2, shape3, shape4,
            src_stride1, src_stride2, src_stride3, src_stride4,
            dst_stride1, dst_stride2, dst_stride3, dst_stride4, block_w);
  }
}

__KERNEL__ void stridedcopy_int32(int32 *ptr_output, int32 *ptr_input,
    const int shape0, const int shape1, const int shape2, const int shape3,
    const int shape4,
    const int src_stride0, const int src_stride1, const int src_stride2, const int src_stride3,
    const int src_stride4,
    const int dst_stride0, const int dst_stride1, const int dst_stride2, const int dst_stride3,
    const int dst_stride4,
    int dim, const int block_w)
{
  if (dim <= 4){
    stridedcopy_move_kernel<int32>(ptr_output, ptr_input,
            shape0, shape1, shape2, shape3,
            src_stride0, src_stride1, src_stride2, src_stride3,
            dst_stride0, dst_stride1, dst_stride2, dst_stride3, block_w);
  } else {
    stridedcopy_move_kernel<int32>(
            ptr_output, ptr_input,
            shape1, shape2, shape3, shape4,
            src_stride1, src_stride2, src_stride3, src_stride4,
            dst_stride1, dst_stride2, dst_stride3, dst_stride4, block_w);
  }
}

__KERNEL__ void stridedcopy_int16(int16 *ptr_output, int16 *ptr_input,
    const int shape0, const int shape1, const int shape2, const int shape3,
    const int shape4,
    const int src_stride0, const int src_stride1, const int src_stride2, const int src_stride3,
    const int src_stride4,
    const int dst_stride0, const int dst_stride1, const int dst_stride2, const int dst_stride3,
    const int dst_stride4,
    int dim, const int block_w)
{
  if (dim <= 4){
    stridedcopy_move_kernel<int16>(ptr_output, ptr_input,
            shape0, shape1, shape2, shape3,
            src_stride0, src_stride1, src_stride2, src_stride3,
            dst_stride0, dst_stride1, dst_stride2, dst_stride3, block_w);
  } else {
    stridedcopy_move_kernel<int16>(
            ptr_output, ptr_input,
            shape1, shape2, shape3, shape4,
            src_stride1, src_stride2, src_stride3, src_stride4,
            dst_stride1, dst_stride2, dst_stride3, dst_stride4, block_w);
  }
}

__KERNEL__ void stridedcopy_int8(int8 *ptr_output, int8 *ptr_input,
    const int shape0, const int shape1, const int shape2, const int shape3,
    const int shape4,
    const int src_stride0, const int src_stride1, const int src_stride2, const int src_stride3,
    const int src_stride4,
    const int dst_stride0, const int dst_stride1, const int dst_stride2, const int dst_stride3,
    const int dst_stride4,
    int dim, const int block_w)
{
  if (dim <= 4){
    stridedcopy_move_kernel<int8>(ptr_output, ptr_input,
            shape0, shape1, shape2, shape3,
            src_stride0, src_stride1, src_stride2, src_stride3,
            dst_stride0, dst_stride1, dst_stride2, dst_stride3, block_w);
  } else {
    stridedcopy_move_kernel<int8>(
            ptr_output, ptr_input,
            shape1, shape2, shape3, shape4,
            src_stride1, src_stride2, src_stride3, src_stride4,
            dst_stride1, dst_stride2, dst_stride3, dst_stride4, block_w);
  }
}

__KERNEL__ void stridedcopy_uint8(uint8 *ptr_output, uint8 *ptr_input,
    const int shape0, const int shape1, const int shape2, const int shape3,
    const int shape4,
    const int src_stride0, const int src_stride1, const int src_stride2, const int src_stride3,
    const int src_stride4,
    const int dst_stride0, const int dst_stride1, const int dst_stride2, const int dst_stride3,
    const int dst_stride4,
    int dim, const int block_w)
{
  if (dim <= 4){
    stridedcopy_move_kernel<uint8>(ptr_output, ptr_input,
            shape0, shape1, shape2, shape3,
            src_stride0, src_stride1, src_stride2, src_stride3,
            dst_stride0, dst_stride1, dst_stride2, dst_stride3, block_w);
  } else {
    stridedcopy_move_kernel<uint8>(
            ptr_output, ptr_input,
            shape1, shape2, shape3, shape4,
            src_stride1, src_stride2, src_stride3, src_stride4,
            dst_stride1, dst_stride2, dst_stride3, dst_stride4, block_w);
  }
}