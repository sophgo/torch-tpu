
#include "ppl.h"
#include "ppl_wrapper_func.h"
#include <vector>

using namespace ppl;

#ifdef __bm1690__
#define BLOCK_NUM 8
#elif defined __sg2260e__
#define BLOCK_NUM 4
#endif

template <typename T>
void scatter_slice_kernel(T *ptr_output, T *ptr_param, T *ptr_input, uint32 *ptr_index,
                          const int outer_size, const int axis, const int inner_size, const int param_h) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();

    int C = outer_size;
    int H = axis;
    int W = inner_size;
    int block_c = LANE_NUM;

    int slice_per_core = div_up(C, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, outer_size - core_offset);

    dim4 output_shape = {1, C, H, W};
    dim4 input_shape = {1, C, H, W};
    dim4 param_shape = {1, C, param_h, W};
    dim4 index_shape = {1, C, param_h, 1};

    auto out_g = gtensor<T>(output_shape, GLOBAL, ptr_output);
    auto input_g = gtensor<T>(input_shape, GLOBAL, ptr_input);
    auto param_g = gtensor<T>(param_shape, GLOBAL, ptr_param);
    auto index_g = gtensor<uint32>(index_shape, GLOBAL, ptr_index);

    for (auto c_idx = 0; c_idx < slice_size_for_core; c_idx += block_c) {
        int c = min(block_c, slice_size_for_core - c_idx);
        dim4 input_global_offset = {0, core_offset + c_idx, 0, 0};
        dim4 local_shape = {1, c, H, W};
        dim4 local_param_shape = {1, c, param_h, W};
        dim4 local_index_shape = {1, c, param_h, 1};

        auto out_sub_g = out_g.sub_view(local_shape, input_global_offset);
        auto input_sub_g = input_g.sub_view(local_shape, input_global_offset);
        auto param_sub_g = param_g.sub_view(local_param_shape, input_global_offset);
        auto index_sub_g = index_g.sub_view(local_index_shape, input_global_offset);

        dma::move(out_sub_g, input_sub_g);

        dma::scatter_h(out_sub_g, param_sub_g, index_sub_g);
    }
}

__KERNEL__ void scatter_fp32(fp32 *ptr_output, fp32 *ptr_param, fp32 *ptr_input, uint32 *ptr_index,
                    const int outer_size, const int axis, const int inner_size, const int param_h) {

  scatter_slice_kernel<fp32>(ptr_output, ptr_param, ptr_input, ptr_index, outer_size, axis, inner_size, param_h);

}

__KERNEL__ void scatter_fp16(fp16 *ptr_output, fp16 *ptr_param, fp16 *ptr_input, uint32 *ptr_index,
                    const int outer_size, const int axis, const int inner_size, const int param_h) {

  scatter_slice_kernel<fp16>(ptr_output, ptr_param, ptr_input, ptr_index, outer_size, axis, inner_size, param_h);

}

__KERNEL__ void scatter_bf16(bf16 *ptr_output, bf16 *ptr_param, bf16 *ptr_input, uint32 *ptr_index,
                    const int outer_size, const int axis, const int inner_size, const int param_h) {

  scatter_slice_kernel<bf16>(ptr_output, ptr_param, ptr_input, ptr_index, outer_size, axis, inner_size, param_h);

}

__KERNEL__ void scatter_int32(int32 *ptr_output, int32 *ptr_param, int32 *ptr_input, uint32 *ptr_index,
                    const int outer_size, const int axis, const int inner_size, const int param_h) {

  scatter_slice_kernel<int32>(ptr_output, ptr_param, ptr_input, ptr_index, outer_size, axis, inner_size, param_h);

}

template <typename T>
void scatter_add_kernel(T *ptr_output, T *ptr_src, uint32 *ptr_index,
                        const int outer_size, const int inner_size, const int param_h) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();

    int C = outer_size;
    int H = inner_size;
    int W = 1;
    int block_c = LANE_NUM;

    int slice_per_core = div_up(C, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, C - core_offset);

    dim4 output_shape = {1, C, H, W};
    dim4 param_shape = {1, C, param_h, W};
    dim4 index_shape = {1, C, param_h, 1};

    auto out_g = gtensor<T>(output_shape, GLOBAL, ptr_output);
    auto param_g = gtensor<T>(param_shape, GLOBAL, ptr_src);
    auto index_g = gtensor<uint32>(index_shape, GLOBAL, ptr_index);

    for (auto c_idx = 0; c_idx < slice_size_for_core; c_idx += block_c) {
        int c = min(block_c, slice_size_for_core - c_idx);
        dim4 input_global_offset = {0, core_offset + c_idx, 0, 0};
        dim4 local_output_shape = {1, c, H, W};
        dim4 local_param_shape = {1, c, param_h, W};
        dim4 local_index_shape = {1, c, param_h, 1};

        auto out_sub_g = out_g.sub_view(local_output_shape, input_global_offset);
        auto param_sub_g = param_g.sub_view(local_param_shape, input_global_offset);
        auto index_sub_g = index_g.sub_view(local_index_shape, input_global_offset);

        // tiu::fill(out_sub_g, 0);
        dma::scatter_h(out_sub_g, param_sub_g, index_sub_g, 0, 1);
    }
}


__KERNEL__ void scatter_add_fp32(fp32 *ptr_output, fp32 *ptr_src, uint32 *ptr_index,
                    const int outer_size, const int inner_size, const int param_h) {
  scatter_add_kernel<fp32>(ptr_output, ptr_src, ptr_index, outer_size, inner_size, param_h);
}

__KERNEL__ void scatter_add_fp16(fp16 *ptr_output, fp16 *ptr_src, uint32 *ptr_index,
                    const int outer_size, const int inner_size, const int param_h) {
  scatter_add_kernel<fp16>(ptr_output, ptr_src, ptr_index, outer_size, inner_size, param_h);
}

__KERNEL__ void scatter_add_bf16(bf16 *ptr_output, bf16 *ptr_src, uint32 *ptr_index,
                    const int outer_size, const int inner_size, const int param_h) {
  scatter_add_kernel<bf16>(ptr_output, ptr_src, ptr_index, outer_size, inner_size, param_h);
}

__KERNEL__ void scatter_add_int32(int32 *ptr_output, int32 *ptr_src, uint32 *ptr_index,
                    const int outer_size, const int inner_size, const int param_h) {
  scatter_add_kernel<int32>(ptr_output, ptr_src, ptr_index, outer_size, inner_size, param_h);
}

__KERNEL__ void scatter_add_int16(int16 *ptr_output, int16 *ptr_src, uint32 *ptr_index,
                    const int outer_size, const int inner_size, const int param_h) {
  scatter_add_kernel<int16>(ptr_output, ptr_src, ptr_index, outer_size, inner_size, param_h);
}

__KERNEL__ void scatter_add_int8(int8 *ptr_output, int8 *ptr_src, uint32 *ptr_index,
                    const int outer_size, const int inner_size, const int param_h) {
  scatter_add_kernel<int8>(ptr_output, ptr_src, ptr_index, outer_size, inner_size, param_h);
}