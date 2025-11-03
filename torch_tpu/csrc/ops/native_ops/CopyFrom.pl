#include "ppl.h"
using namespace ppl;

#ifdef __bm1690__
#define BLOCK_NUM 8
#elif defined __sg2260e__
#define BLOCK_NUM 4
#endif

template <typename T, typename U>
void convert_kernel(T *ptr_rst, U *ptr_inp, const int tile_size,
                    int inner_size, int outer_size) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();

    int N = 1;
    int C = outer_size;
    int H = 1;
    int W = inner_size;

    int block_c = LANE_NUM;
    int block_w_iter = min(tile_size, W);

    int slice_per_core = div_up(outer_size, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, outer_size - core_offset);

    dim4 global_shape = {N, C, H, W};
    auto g_inp = gtensor<U>(global_shape, GLOBAL, ptr_inp);
    auto g_rst = gtensor<T>(global_shape, GLOBAL, ptr_rst);

    dim4 local_in_block_shape = {1, block_c, 1, tile_size};

    for (int c_idx = 0; c_idx < slice_size_for_core; c_idx += block_c) {
        int c = min(block_c, slice_size_for_core - c_idx);

        for (int w_idx = 0; w_idx < W; w_idx += block_w_iter) {
            enable_pipeline();
            int w = min(block_w_iter, W - w_idx);

            dim4 local_in_shape = {N, c, H, w};
            auto local_in = make_tensor<U>(local_in_block_shape, local_in_shape);
            auto local_rst = make_tensor<T>(local_in_block_shape, local_in_shape);
            dim4 global_offset = {0, core_offset + c_idx, 0, w_idx};

            dma::load(local_in, g_inp.sub_view(local_in_shape, global_offset));
            tiu::cast(local_rst, local_in);
            dma::store(g_rst.sub_view(local_in_shape, global_offset), local_rst);
        }
    }
}


// Half -> Float
__KERNEL__ void convert_impl_half_to_float(fp32 *ptr_output, fp16 *ptr_input,
                                            const int tile_size,int inner_size, int outer_size) {
    convert_kernel<fp32, fp16>(ptr_output, ptr_input, tile_size, inner_size, outer_size);
}

// BFloat16 -> Float
__KERNEL__ void convert_impl_bf16_to_float(fp32 *ptr_output, bf16 *ptr_input,
                                            const int tile_size,int inner_size, int outer_size) {
    convert_kernel<fp32, bf16>(ptr_output, ptr_input, tile_size, inner_size, outer_size);
}

// Float -> Half
__KERNEL__ void convert_impl_float_to_half(fp16 *ptr_output, fp32 *ptr_input,
                                            const int tile_size,int inner_size, int outer_size) {
    convert_kernel<fp16, fp32>(ptr_output, ptr_input, tile_size, inner_size, outer_size);
}

// Float -> BFloat16
__KERNEL__ void convert_impl_float_to_bf16(bf16 *ptr_output, fp32 *ptr_input,
                                            const int tile_size,int inner_size, int outer_size) {
    convert_kernel<bf16, fp32>(ptr_output, ptr_input, tile_size, inner_size, outer_size);
}

// Float -> Float
__KERNEL__ void convert_impl_float_to_float(fp32 *ptr_output, fp32 *ptr_input,
                                            const int tile_size,int inner_size, int outer_size) {
    convert_kernel<fp32, fp32>(ptr_output, ptr_input, tile_size, inner_size, outer_size);
}

// Half -> Half
__KERNEL__ void convert_impl_half_to_half(fp16 *ptr_output, fp16 *ptr_input,
                                            const int tile_size,int inner_size, int outer_size) {
    convert_kernel<fp16, fp16>(ptr_output, ptr_input, tile_size, inner_size, outer_size);
}

// BFloat16 -> BFloat16
__KERNEL__ void convert_impl_bf16_to_bf16(bf16 *ptr_output, bf16 *ptr_input,
                                            const int tile_size,int inner_size, int outer_size) {
    convert_kernel<bf16, bf16>(ptr_output, ptr_input, tile_size, inner_size, outer_size);
}

// Int32 -> Float
__KERNEL__ void convert_impl_int32_to_float(fp32 *ptr_output, int32 *ptr_input,
                                            const int tile_size,int inner_size, int outer_size) {
    convert_kernel<fp32, int32>(ptr_output, ptr_input, tile_size, inner_size, outer_size);
}

// Int32 -> Half
__KERNEL__ void convert_impl_int32_to_half(fp16 *ptr_output, int32 *ptr_input,
                                            const int tile_size,int inner_size, int outer_size) {
    convert_kernel<fp16, int32>(ptr_output, ptr_input, tile_size, inner_size, outer_size);
}

// Int32 -> BFloat16
__KERNEL__ void convert_impl_int32_to_bf16(bf16 *ptr_output, int32 *ptr_input,
                                            const int tile_size,int inner_size, int outer_size) {
    convert_kernel<bf16, int32>(ptr_output, ptr_input, tile_size, inner_size, outer_size);
}

// Int32 -> Int32
__KERNEL__ void convert_impl_int32_to_int32(int32 *ptr_output, int *ptr_input,
                                            const int tile_size,int inner_size, int outer_size) {
    convert_kernel<int32, int32>(ptr_output, ptr_input, tile_size, inner_size, outer_size);
}
