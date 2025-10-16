
#include "ppl.h"
#include <vector>

using namespace ppl;

#ifdef __bm1690__
#define BLOCK_NUM 8
#elif defined __sg2260e__
#define BLOCK_NUM 4
#endif

template <typename T>
void fill_kernel(T *ptr_output, int32_t value, int outer_size, const int block_w) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();

    int slice_per_core = div_up(outer_size, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, outer_size - core_offset);

    dim4 output_shape = {1, 1, 1, outer_size};
    auto out_g = gtensor<T>(output_shape, GLOBAL, ptr_output);
    dim4 local_out_block = {1, 1, 1, block_w};

    for (auto w_idx = 0; w_idx < slice_size_for_core; w_idx += block_w) {
        enable_pipeline();
        int w = min(block_w, slice_size_for_core - w_idx);
        dim4 local_out_shape = {1, 1, 1, w};
        dim4 output_global_offset = {0, 0, 0, core_offset + w_idx};

        auto out_buffer = make_tensor<T>(local_out_block, local_out_shape);
        tiu::fill(out_buffer, value);

        dma::store(
            out_g.sub_view(local_out_shape, output_global_offset),
            out_buffer);
    }

}

__KERNEL__ void fill_fp32(float *ptr_output, int32_t value, int output_size, const int block_w) {
  fill_kernel<float>(ptr_output, value, output_size, block_w);
}

__KERNEL__ void fill_fp16(fp16 *ptr_output, int32_t value, int output_size, const int block_w) {
  fill_kernel<fp16>(ptr_output, value, output_size, block_w);
}

__KERNEL__ void fill_bf16(bf16 *ptr_output, int32_t value, int output_size, const int block_w) {
  fill_kernel<bf16>(ptr_output, value, output_size, block_w);
}

__KERNEL__ void fill_int32(int32_t *ptr_output, int32_t value, int output_size, const int block_w) {
  fill_kernel<int32_t>(ptr_output, value, output_size, block_w);
}

__KERNEL__ void fill_uint8(uint8_t *ptr_output, int32_t value, int output_size, const int block_w) {
  fill_kernel<uint8_t>(ptr_output, value, output_size, block_w);
}
