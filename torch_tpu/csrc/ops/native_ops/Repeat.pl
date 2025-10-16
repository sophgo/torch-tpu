#include "ppl.h"

using namespace ppl;

#ifdef __bm1690__
#define BLOCK_NUM 8
#elif defined __sg2260e__
#define BLOCK_NUM 4
#endif

template <typename T>
void repeat_kernel(T *ptr_output, T *ptr_input, int row, int column, int repeat) {
    ppl::set_block_num(BLOCK_NUM);
    const int core_num = get_block_num();
    const int core_idx = get_block_index();
    const int rows_per_core = div_up(row, core_num);
    const int row_start = core_idx * rows_per_core;
    const int row_end = min(row_start + rows_per_core, row);
    const int valid_rows = row_end - row_start;

    if (valid_rows <= 0) {
        return;
    }

    dim4 local_src_shape = {1, 1, valid_rows, column};
    dim4 local_dst_shape = {1, 1, valid_rows, column * repeat};
    dim4 src_offset = {0, 0, row_start, 0};
    dim4 dst_offset = {0, 0, row_start, 0};

    auto input = gtensor<T>(local_src_shape, GLOBAL, ptr_input).sub_view(local_src_shape, src_offset);
    auto output = gtensor<T>(local_dst_shape, GLOBAL, ptr_output).sub_view(local_dst_shape, dst_offset);

    dim4 src_stride  = { 0, 0, column, 1};
    for (int r = 0; r < repeat; ++r) {
        enable_pipeline();
        dim4 repeat_offset = {0, 0, 0, r * column};
        auto dst_view = output.sub_view(local_src_shape, repeat_offset);
        dma::move(dst_view, input.view(local_src_shape, src_stride));
    }
}

__KERNEL__ void repeat_fp32(fp32 *ptr_output, fp32 *ptr_input,
                             int row, int column, int repeat) {
    repeat_kernel<fp32>(ptr_output, ptr_input, row, column, repeat);
}

__KERNEL__ void repeat_fp16(fp16 *ptr_output, fp16 *ptr_input,
                             int row, int column, int repeat) {
    repeat_kernel<fp16>(ptr_output, ptr_input, row, column, repeat);
}

__KERNEL__ void repeat_bf16(bf16 *ptr_output, bf16 *ptr_input,
                             int row, int column, int repeat) {
    repeat_kernel<bf16>(ptr_output, ptr_input, row, column, repeat);
}

__KERNEL__ void repeat_int32(int32 *ptr_output, int32 *ptr_input,
                             int row, int column, int repeat) {
    repeat_kernel<int32>(ptr_output, ptr_input, row, column, repeat);
}

__KERNEL__ void repeat_int16(int16 *ptr_output, int16 *ptr_input,
                             int row, int column, int repeat) {
    repeat_kernel<int16>(ptr_output, ptr_input, row, column, repeat);
}

__KERNEL__ void repeat_int8(int8 *ptr_output, int8 *ptr_input,
                             int row, int column, int repeat) {
    repeat_kernel<int8>(ptr_output, ptr_input, row, column, repeat);
}

__KERNEL__ void repeat_uint8(uint8 *ptr_output, uint8 *ptr_input,
                             int row, int column, int repeat) {
    repeat_kernel<uint8>(ptr_output, ptr_input, row, column, repeat);
}