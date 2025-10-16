
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
void indexselect_kernel(T *ptr_output, uint32 *ptr_index, T *ptr_intput,
                        const int outer_size, const int inner_size,
                        const int gather_num, const int gathered_num) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();

    int C = outer_size;
    int H = gathered_num;
    int W = inner_size;
    int block_c = LANE_NUM;

    int slice_per_core = div_up(C, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, outer_size - core_offset);

    dim4 output_shape = {1, C, H, W};
    dim4 input_shape = {1, C, gather_num, W};
    dim4 index_shape = {1, 1, H, 1};

    auto out_g = gtensor<T>(output_shape, GLOBAL, ptr_output);
    auto input_g = gtensor<T>(input_shape, GLOBAL, ptr_intput);
    auto index_g = gtensor<uint32>(index_shape, GLOBAL, ptr_index);

    dim4 offset = {0, core_offset, 0, 0};
    dma::gather_h(out_g.sub_view(output_shape, offset),
                  input_g.sub_view(input_shape, offset),
                  index_g.sub_view(index_shape, offset), 0);
}

__KERNEL__ void indexselect_fp32(fp32 *ptr_output, uint32 *ptr_index, fp32 *ptr_input,
                    const int outer_size, const int inner_size, const int gather_num, const int gathered_num) {
  indexselect_kernel<fp32>(ptr_output, ptr_index, ptr_input, outer_size, inner_size, gather_num, gathered_num);
}

__KERNEL__ void indexselect_fp16(fp16 *ptr_output, uint32 *ptr_index, fp16 *ptr_input,
                    const int outer_size, const int inner_size, const int gather_num, const int gathered_num) {
  indexselect_kernel<fp16>(ptr_output, ptr_index, ptr_input, outer_size, inner_size, gather_num, gathered_num);
}

__KERNEL__ void indexselect_bf16(bf16 *ptr_output, uint32 *ptr_index, bf16 *ptr_input,
                    const int outer_size, const int inner_size, const int gather_num, const int gathered_num) {
  indexselect_kernel<bf16>(ptr_output, ptr_index, ptr_input, outer_size, inner_size, gather_num, gathered_num);
}

__KERNEL__ void indexselect_int32(int32 *ptr_output, uint32 *ptr_index, int32 *ptr_input,
                    const int outer_size, const int inner_size, const int gather_num, const int gathered_num) {
  indexselect_kernel<int32>(ptr_output, ptr_index, ptr_input, outer_size, inner_size, gather_num, gathered_num);
}

__KERNEL__ void indexselect_int16(int16 *ptr_output, uint32 *ptr_index, int16 *ptr_input,
                    const int outer_size, const int inner_size, const int gather_num, const int gathered_num) {
  indexselect_kernel<int16>(ptr_output, ptr_index, ptr_input, outer_size, inner_size, gather_num, gathered_num);
}

__KERNEL__ void indexselect_int8(int8 *ptr_output, uint32 *ptr_index, int8 *ptr_input,
                    const int outer_size, const int inner_size, const int gather_num, const int gathered_num) {
  indexselect_kernel<int8>(ptr_output, ptr_index, ptr_input, outer_size, inner_size, gather_num, gathered_num);
}