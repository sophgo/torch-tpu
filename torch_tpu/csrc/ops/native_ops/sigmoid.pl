
#include "ppl.h"
#include "ppl_tpu.h"
#include "ppl_wrapper_func.h"
using namespace ppl;

#ifdef __bm1690__
#define BLOCK_NUM 8
#elif defined __sg2260e__
#define BLOCK_NUM 4
#endif
#define GDMA_MAX_C 65535

template <typename T>
void sigmoid_kernel(T *sigmoid_output, T *sigmoid_input, int inner_size, const int block_size) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();
    auto slice = div_up(inner_size, core_num);
    int core_offset = slice * core_idx;
    int slice_size = min(slice, inner_size - core_offset);
    if (slice_size <= 0) return;

    const int w_dim = get_eu_num<T>();
    const int max_m_dim = (GDMA_MAX_C + 1) >> 1;
    int m_dim = min(inner_size, (int)max_m_dim);
    int n_dim = (inner_size + m_dim - 1) / m_dim;
    dim4 global_shape = {n_dim, div_up(m_dim, w_dim), 1, w_dim};
    auto sigmoid_gtensor_in = gtensor<T>(global_shape, GLOBAL, sigmoid_input);
    auto sigmoid_gtensor_out = gtensor<T>(global_shape, GLOBAL, sigmoid_output);
    const int block_iter = min(block_size, inner_size);

    int m_dim_block = min(block_iter, (int)max_m_dim);
    int n_dim_block = (block_iter + m_dim_block - 1) / m_dim_block;
    dim4 local_block_shape = {n_dim_block, div_up(m_dim_block, w_dim), 1, w_dim};
    for (int c_idx = 0; c_idx < slice_size; c_idx += block_iter) {
        enable_pipeline();
        int c = min(block_iter, slice_size - c_idx);
        dim4 local_shape = {1, div_up(c, w_dim), 1, w_dim};
        int flat_offset = core_offset + c_idx;
        int n_off = flat_offset / m_dim;
        int m_off = flat_offset % m_dim;
        dim4 offset = {n_off, m_off / w_dim, 0, m_off % w_dim};

        auto local_in  = make_tensor<T>(local_block_shape, local_shape);
        auto local_out = make_tensor<T>(local_block_shape, local_shape);
        dma::load(local_in, sigmoid_gtensor_in.sub_view(local_shape, offset));
        sigmoid_fp32<T>(local_out, local_in, &local_block_shape, &local_shape);
        dma::store(sigmoid_gtensor_out.sub_view(local_shape, offset), local_out);
    }
}

__KERNEL__ void activation_sigmoid_impl_float32(fp32 *ptr_output, fp32 *ptr_input,
                                             int inner_size, const int block_size) {
    sigmoid_kernel<fp32>(ptr_output, ptr_input, inner_size, block_size);
}

__KERNEL__ void activation_sigmoid_impl_float16(fp16 *ptr_output, fp16 *ptr_input,
                                             int inner_size, const int block_size) {
    sigmoid_kernel<fp16>(ptr_output, ptr_input, inner_size, block_size);
}

__KERNEL__ void activation_sigmoid_impl_bf16(bf16 *ptr_output, bf16 *ptr_input,
                                             int inner_size, const int block_size) {
    sigmoid_kernel<bf16>(ptr_output, ptr_input, inner_size, block_size);
}