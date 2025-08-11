#include "ppl.h"
#include "ppl_wrapper_func.h"

using namespace ppl;

#ifndef BLOCK_NUM
#define BLOCK_NUM 4
#endif

template <typename T>
void rmsnorm_kernel(T *ptr_output, T *ptr_input, T *ptr_weight, T *ptr_bias,  float eps,
                    bool affine, bool with_bias, int outer_size, int inner_size, const int block_w) {
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();

    int N = 1;
    int C = outer_size;
    int H = 1;
    int W = inner_size;
    // Make sure to loop at least twice so that you can run the pipeline in
    // parallel
    int block_w_iter = max(min(block_w, W / 2), 1);
    int block_c = LANE_NUM;

    int slice_per_core = div_up(outer_size, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, outer_size - core_offset);

    dim4 global_shape = {1, C, 1, W};
    auto in_gtensor = gtensor<T>(global_shape, GLOBAL, ptr_input);
    auto weight_gtensor = gtensor<T>(global_shape, GLOBAL, ptr_weight);
    auto bias_gtensor = gtensor<T>(global_shape, GLOBAL, ptr_bias);
    auto out_gtensor = gtensor<T>(global_shape, GLOBAL, ptr_output);

    dim4 local_in_block_shape = {1, block_c, 1, block_w};
    dim4 local_avg_block_shape = {1, block_c, 1, 1};
    dim4 local_weight_shape = {1, 1, 1, block_w};
    dim4 local_bias_shape = {1, 1, 1, block_w};

    for (auto c_idx = 0; c_idx < slice_size_for_core; c_idx += block_c) {
        int c = min(block_c, slice_size_for_core - c_idx);
        dim4 local_avg_shape = {1, c, 1, 1};
        auto avg_buffer = make_tensor<T>(local_avg_block_shape, local_avg_shape);
        tiu::fill(avg_buffer, eps);

        for (auto w_idx = 0; w_idx < W; w_idx += block_w_iter) {
            enable_pipeline();
            int w = min(block_w_iter, W - w_idx);
            dim4 local_in_shape = {1, c, 1, w};
            dim4 input_global_offset = {0, core_offset + c_idx, 0, w_idx};

            auto local_in = make_tensor<T>(local_in_block_shape, local_in_shape);

            dma::load(local_in,
                    in_gtensor.sub_view(local_in_shape, input_global_offset));
            auto local_in_tmp = make_tensor<T>(local_in_block_shape, local_in_shape);

            // tmp = x^2
            tiu::fmul(local_in_tmp, local_in, local_in);
            auto sub_avg = make_tensor<T>(local_avg_block_shape, local_avg_shape);
            // avg(x^2)
            quick_pooling(sub_avg, local_in_tmp, &local_in_block_shape,
                    &local_in_shape, 0, 1, 1.f / W);
            // avg(x^2) + exp
            tiu::fadd(avg_buffer, avg_buffer, sub_avg);
        }

        tensor<T> local_mu = make_tensor<T>(local_avg_block_shape, local_avg_shape);
        // 1/sqrt(avg(x^2) + exp)
        if (std::is_same_v<T, bf16> || std::is_same_v<T, fp16>) {
            auto local_mu_tmp =
                    make_tensor<fp32>(local_avg_block_shape, local_avg_shape);
            auto avg_buffer_fp32 =
                    make_tensor<fp32>(local_avg_block_shape, local_avg_shape);
            tiu::cast(avg_buffer_fp32, avg_buffer);
            tiu::frsqrt(local_mu_tmp, avg_buffer_fp32, 4);
            tiu::cast(local_mu, local_mu_tmp);
        } else {
            tiu::frsqrt(local_mu, avg_buffer, 4);
        }

        for (auto w_idx = 0; w_idx < W; w_idx += block_w_iter) {
            enable_pipeline();
            int w = min(block_w_iter, W - w_idx);
            dim4 local_in_shape = {1, c, 1, w};
            dim4 input_global_offset = {0, core_offset + c_idx, 0, w_idx};

            auto local_in = make_tensor<T>(local_in_block_shape, local_in_shape);

            dma::load(local_in,
                    in_gtensor.sub_view(local_in_shape, input_global_offset));

            auto out = make_tensor<T>(local_in_block_shape, local_in_shape);
            // 1/sqrt(avg(x^2)) * x
            tiu::fmul(out, local_in, local_mu);

            if (affine) {
                dim4 weight_offset = {0, 0, 0, w_idx};
                dim4 weight_real_shape = {1, 1, 1, w};
                auto local_weight_sub =
                        make_tensor<T>(local_weight_shape, weight_real_shape);
                dma::load(local_weight_sub,
                        weight_gtensor.sub_view(weight_real_shape, weight_offset));
                tiu::broadcast(local_in, local_weight_sub);
                tiu::fmul(out, out, local_in);
                if (with_bias) {
                    dim4 bias_offset = {0, 0, 0, w_idx};
                    auto local_bias_sub = make_tensor<T>(local_bias_shape, weight_real_shape);
                    dma::load(local_bias_sub, bias_gtensor.sub_view(weight_real_shape, bias_offset));
                    tiu::broadcast(local_bias_sub, local_bias_sub);
                    tiu::fadd(out, out, local_bias_sub);
                }
            }

            dma::store(out_gtensor.sub_view(local_in_shape, input_global_offset),
                    out);
        }
    }
}

__KERNEL__ void rmsnorm_fp32(fp32 *ptr_output, fp32 *ptr_input,
                             fp32 *ptr_weight, fp32 *ptr_bias, float eps, bool affine, bool with_bias,
                             int outer_size, int inner_size, const int block_w) {
    rmsnorm_kernel<fp32>(ptr_output, ptr_input, ptr_weight, ptr_bias, eps, affine, with_bias, outer_size,
                         inner_size, block_w);
}

__KERNEL__ void rmsnorm_fp16(fp16 *ptr_output, fp16 *ptr_input,
                             fp16 *ptr_weight, fp16 *ptr_bias, float eps, bool affine, bool with_bias,
                             int outer_size, int inner_size, const int block_w) {
    rmsnorm_kernel<fp16>(ptr_output, ptr_input, ptr_weight, ptr_bias, eps, affine, with_bias, outer_size,
                         inner_size, block_w);
}

__KERNEL__ void rmsnorm_bf16(bf16 *ptr_output, bf16 *ptr_input,
                             bf16 *ptr_weight, bf16 *ptr_bias, float eps, bool affine, bool with_bias,
                             int outer_size, int inner_size, const int block_w) {
    rmsnorm_kernel<bf16>(ptr_output, ptr_input, ptr_weight, ptr_bias, eps, affine, with_bias, outer_size,
                         inner_size, block_w);
}

