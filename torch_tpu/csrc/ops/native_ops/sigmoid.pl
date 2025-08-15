
#include "ppl.h"
#include "ppl_tpu.h"
using namespace ppl;

#define EXP_NO_OVERFLOW(out, in, shape, real_shape, T, MIN_C, MAX_C, MUL1, MUL2, INT_MAX, INT_MIN, ADD_CONST, ADD_SHIFT, EXP_TYPE, VIEW_TYPE) \
    auto maxc_tensor = make_tensor<T>(shape, real_shape); \
    tiu::fmax(maxc_tensor, in, MIN_C); \
    auto minc_tensor1 = make_tensor<T>(shape, real_shape); \
    if constexpr (std::is_same<T, fp16>::value) { \
        tiu::fmin(minc_tensor1, maxc_tensor, MAX_C); \
    } else { \
        tiu::move(minc_tensor1, maxc_tensor); \
    } \
    auto fp_mulc_tensor = make_tensor<T>(shape, real_shape); \
    tiu::fmul(fp_mulc_tensor, minc_tensor1, MUL1); \
    auto fp_floor_tensor = make_tensor<T>(shape, real_shape); \
    tiu::floor(fp_floor_tensor, fp_mulc_tensor); \
    auto fp_mulc_tensor2 = make_tensor<T>(shape, real_shape); \
    tiu::fmul(fp_mulc_tensor2, fp_floor_tensor, MUL2); \
    auto fp_sub = make_tensor<T>(shape, real_shape); \
    tiu::fsub(fp_sub, maxc_tensor, fp_mulc_tensor2); \
    auto cast_out = make_tensor<EXP_TYPE>(shape, real_shape); \
    tiu::cast(cast_out, fp_floor_tensor, RM_HALF_AWAY_FROM_ZERO); \
    auto minc_tensor = make_tensor<EXP_TYPE>(shape, real_shape); \
    tiu::min(minc_tensor, cast_out, (int16)INT_MAX); \
    auto maxc_tensor2 = make_tensor<EXP_TYPE>(shape, real_shape); \
    tiu::max(maxc_tensor2, minc_tensor, (int16)INT_MIN); \
    auto add_intc_tensor = make_tensor<VIEW_TYPE>(shape, real_shape); \
    tiu::add(add_intc_tensor, maxc_tensor2, (int16)ADD_CONST, ADD_SHIFT, RM_HALF_AWAY_FROM_ZERO, true); \
    auto exp_out = make_tensor<T>(shape, real_shape); \
    tiu::fexp(exp_out, fp_sub); \
    auto cast_intc_tensor = add_intc_tensor.view<T>(); \
    tiu::fmul(out, exp_out, cast_intc_tensor);

#ifndef BLOCK_NUM
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
        int c = min(block_iter, slice_size - c_idx);
        dim4 local_shape = {1, div_up(c, w_dim), 1, w_dim};
        int flat_offset = core_offset + c_idx;
        int n_off = flat_offset / m_dim;
        int m_off = flat_offset % m_dim;
        dim4 offset = {n_off, m_off / w_dim, 0, m_off % w_dim};

        auto local_in  = make_tensor<T>(local_block_shape, local_shape);
        auto local_out = make_tensor<T>(local_block_shape, local_shape);
        dma::load(local_in, sigmoid_gtensor_in.sub_view(local_shape, offset));

        auto local_input_exp = make_tensor<T>(local_block_shape, local_shape);
        if constexpr (std::is_same<T, fp32>::value) {
            EXP_NO_OVERFLOW(local_input_exp, local_in, local_block_shape, local_shape, fp32,
                -3.40282e35f, 0.0f, 1.4426950f, 0.69314718f, 127, -127, 127, 23, int16, int32);
        } else if constexpr (std::is_same<T, fp16>::value) {
            EXP_NO_OVERFLOW(local_input_exp, local_in, local_block_shape, local_shape, fp16,
                -45403.f, 45403.f, 1.4426950f, 0.69314718f, 15, -15, 15, 10, int8, int16);
        } else if constexpr (std::is_same<T, bf16>::value) {
            EXP_NO_OVERFLOW(local_input_exp, local_in, local_block_shape, local_shape, bf16,
                -3.40282e35f, 0.0f, 1.4426950f, 0.69314718f, 127, -127, 127, 7, int16, int16);
        }

        auto local_input_exp_reciprocal = make_tensor<T>(local_block_shape, local_shape);
        tiu::fdiv(local_input_exp_reciprocal, 1, local_input_exp, 3);

        auto local_output_pre = make_tensor<T>(local_block_shape, local_shape);
        tiu::fadd(local_output_pre, local_input_exp_reciprocal, 1);
        tiu::fdiv(local_out, 1, local_output_pre, 3);

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