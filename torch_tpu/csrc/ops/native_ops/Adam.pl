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
void adambackward_kernel(
        T* weight_out, T* m_out, T* v_out, T* vmax_out,
        T* grad_in, T* weight_in, T* m_in, T* v_in, T* vmax_in, T* t,
        float lr, float beta1, float beta2, float eps, const float weight_decay, bool amsgrad,
        bool maximize, const int inner_size, const int block_w
) {
    const int W = inner_size;
    ppl::set_block_num(BLOCK_NUM);
    int core_num = get_block_num();
    int core_idx = get_block_index();
    int slice_per_core = div_up(W, core_num);
    int core_offset = slice_per_core * core_idx;
    int slice_size_for_core = min(slice_per_core, W - core_offset);

    dim4 in_gshape = {1, 1, 1, W};
    dim4 out_gshape = {1, 1, 1, W};
    auto in_weight_g  = gtensor<T>(in_gshape, GLOBAL, weight_in);
    auto in_m_g  = gtensor<T>(in_gshape, GLOBAL, m_in);
    auto in_v_g  = gtensor<T>(in_gshape, GLOBAL, v_in);
    auto in_vmax_g  = gtensor<T>(in_gshape, GLOBAL, vmax_in);
    auto in_t_g  = gtensor<T>(in_gshape, GLOBAL, t);
    auto in_grad_g = gtensor<T>(in_gshape, GLOBAL, grad_in);

    auto out_weight_g = gtensor<T>(out_gshape, GLOBAL, weight_out);
    auto out_m_g = gtensor<T>(out_gshape, GLOBAL, m_out);
    auto out_v_g = gtensor<T>(out_gshape, GLOBAL, v_out);
    auto out_vmax_g = gtensor<T>(out_gshape, GLOBAL, vmax_out);

    dim4 in_blockshape = {1, 1, 1, block_w};
    dim4 out_blockshape = {1, 1, 1, block_w};
    for (auto w_idx = 0; w_idx < slice_size_for_core; w_idx += block_w) {
        enable_pipeline();
        int w = min(block_w, slice_size_for_core - w_idx);
        dim4 input_offset = {0, 0, 0, core_offset + w_idx};
        dim4 in_lshape = {1, 1, 1, w};
        dim4 out_lshape = {1, 1, 1, w};
        auto in_weight_l  = make_tensor<T>(in_blockshape, in_lshape);
        auto in_m_l  = make_tensor<T>(in_blockshape, in_lshape);
        auto in_v_l  = make_tensor<T>(in_blockshape, in_lshape);
        auto in_vmax_l  = make_tensor<T>(in_blockshape, in_lshape);
        auto in_t_l  = make_tensor<T>(in_blockshape, in_lshape);
        auto in_grad_l = make_tensor<T>(in_blockshape, in_lshape);

        auto out_weight_l = make_tensor<T>(out_blockshape, out_lshape);
        auto out_m_l = make_tensor<T>(out_blockshape, out_lshape);
        auto out_v_l = make_tensor<T>(out_blockshape, out_lshape);
        auto out_vmax_l = make_tensor<T>(out_blockshape, out_lshape);

        dma::load(in_weight_l, in_weight_g.sub_view(in_lshape, input_offset));
        dma::load(in_m_l, in_m_g.sub_view(in_lshape, input_offset));
        dma::load(in_v_l, in_v_g.sub_view(in_lshape, input_offset));
        dma::load(in_vmax_l, in_vmax_g.sub_view(in_lshape, input_offset));
        dma::load(in_t_l, in_t_g.sub_view(in_lshape, input_offset));
        dma::load(in_grad_l, in_grad_g.sub_view(in_lshape, input_offset));

        if (maximize){
            // grad = -grad
            tiu::fmul(in_grad_l, in_grad_l, -1.0f);
        }
        // grad = grad + weight_decay * weight
        auto temp1 = make_tensor<T>(in_blockshape, in_lshape);
        tiu::fmul(temp1, in_weight_l, weight_decay);
        tiu::fadd(in_grad_l, in_grad_l, temp1);

        // m_t = beta1 * m + (1 - beta1) * grad
        tiu::fmul(out_m_l, in_m_l, beta1);
        tiu::fmul(temp1, in_grad_l, 1.0f - beta1);
        tiu::fadd(out_m_l, out_m_l, temp1);

        // v_t = beta2 * v + (1 - beta2) * grad * grad
        tiu::fmul(out_v_l, in_v_l, beta2);
        tiu::fmul(temp1, in_grad_l, in_grad_l);
        tiu::fmul(temp1, temp1, 1.0f - beta2);
        tiu::fadd(out_v_l, out_v_l, temp1);
        if (amsgrad){
            // vmax_t = max(vmax, v_t)
            tiu::fmax(out_vmax_l, in_vmax_l, out_v_l);
            // denom = sqrt(vmax_t) + eps
            // tiu::fsqrt(out_v_l, out_vmax_l);
            exp_no_overflow(out_v_l, out_vmax_l, &in_blockshape, &in_lshape);
            tiu::fadd(out_v_l, out_v_l, eps);
        } else {
            // denom = sqrt(v_t) + eps
            // tiu::fsqrt(out_v_l, out_v_l);
            exp_no_overflow(out_v_l, out_v_l, &in_blockshape, &in_lshape);
            tiu::fadd(out_v_l, out_v_l, eps);
        }

        // bias_correction1 = 1 - beta1^t
        // bias_correction2 = 1 - beta2^t
        auto bias_correction1_fp = make_tensor<fp32>(in_blockshape, in_lshape);
        auto bias_correction2_fp = make_tensor<fp32>(in_blockshape, in_lshape);

        auto t_float = make_tensor<fp32>(in_blockshape, in_lshape);
        tiu::cast(t_float, in_t_l);
        auto beta1_t = make_tensor<fp32>(in_blockshape, in_lshape);
        auto beta2_t = make_tensor<fp32>(in_blockshape, in_lshape);
        pow_f32(beta1_t, beta1, t_float, &in_blockshape, &in_lshape);
        pow_f32(beta2_t, beta2, t_float, &in_blockshape, &in_lshape);
        tiu::fsub(bias_correction1_fp, 1.0f, beta1_t);
        tiu::fsub(bias_correction2_fp, 1.0f, beta2_t);

        // step_size = lr * sqrt(bias_correction2) / bias_correction1
        auto step_size = make_tensor<fp32>(in_blockshape, in_lshape);
        // tiu::fsqrt(step_size, bias_correction2);
        exp_no_overflow(step_size, bias_correction2_fp, &in_blockshape, &in_lshape);
        tiu::fdiv(step_size, step_size, bias_correction1_fp);
        tiu::fmul(step_size, step_size, lr);

        // weight = weight - step_size * m_t / denom
        auto m_fp32 = make_tensor<fp32>(in_blockshape, in_lshape);
        auto denom_fp32 = make_tensor<fp32>(in_blockshape, in_lshape);
        tiu::cast(m_fp32, out_m_l);
        tiu::cast(denom_fp32, out_v_l);
        tiu::fdiv(m_fp32, m_fp32, denom_fp32);
        tiu::fmul(m_fp32, m_fp32, step_size);
        tiu::cast(temp1, m_fp32);
        tiu::fsub(out_weight_l, in_weight_l, temp1);

        dma::store(out_weight_g.sub_view(out_lshape, input_offset), out_weight_l);
        dma::store(out_m_g.sub_view(out_lshape, input_offset), out_m_l);
        dma::store(out_v_g.sub_view(out_lshape, input_offset), out_v_l);
        if (amsgrad){
            dma::store(out_vmax_g.sub_view(out_lshape, input_offset), out_vmax_l);
        }
    }
}

__KERNEL__ void adambackward_fp32(fp32 *weight_out, fp32 *m_out, fp32 *v_out, fp32 *vmax_out,
    fp32 *grad_in, fp32 *weight_in, fp32 *m_in, fp32 *v_in, fp32 *vmax_in, fp32 *t,
    float lr, float beta1, float beta2, float eps, const float weight_decay, bool amsgrad,
    bool maximize, const int inner_size, const int block_w) {
    adambackward_kernel<fp32>(weight_out, m_out, v_out, vmax_out,
        grad_in, weight_in, m_in, v_in, vmax_in, t,
        lr, beta1, beta2, eps, weight_decay, amsgrad,
        maximize, inner_size, block_w);
}

__KERNEL__ void adambackward_fp16(fp16 *weight_out, fp16 *m_out, fp16 *v_out, fp16 *vmax_out,
    fp16 *grad_in, fp16 *weight_in, fp16 *m_in, fp16 *v_in, fp16 *vmax_in, fp16 *t,
    float lr, float beta1, float beta2, float eps, const float weight_decay, bool amsgrad,
    bool maximize, const int inner_size, const int block_w) {
    adambackward_kernel<fp16>(weight_out, m_out, v_out, vmax_out,
        grad_in, weight_in, m_in, v_in, vmax_in, t,
        lr, beta1, beta2, eps, weight_decay, amsgrad,
        maximize, inner_size, block_w);
}

__KERNEL__ void adambackward_bf16(bf16 *weight_out, bf16 *m_out, bf16 *v_out, bf16 *vmax_out,
    bf16 *grad_in, bf16 *weight_in, bf16 *m_in, bf16 *v_in, bf16 *vmax_in, bf16 *t,
    float lr, float beta1, float beta2, float eps, const float weight_decay, bool amsgrad,
    bool maximize, const int inner_size, const int block_w) {
    adambackward_kernel<bf16>(weight_out, m_out, v_out, vmax_out,
        grad_in, weight_in, m_in, v_in, vmax_in, t,
        lr, beta1, beta2, eps, weight_decay, amsgrad,
        maximize, inner_size, block_w);
}