#include <iostream>
#include <memory>
#include <type_traits>
#include <unordered_map>
#include <math.h>
#include <omp.h>
#include <torch/extension.h>

#include "cpu_adam.h"

static std::unordered_map<int, std::shared_ptr<void>> s_optimizers;

// C++ interface

void Adam_Optimizer::Step_1(
    float* _params,
    float* grads,
    float* _exp_avg,
    float* _exp_avg_sq,
    size_t _param_size,
    ds_half_precision_t* dev_params,
    bool half_precision
){
    size_t rounded_size = 0;

    if (_param_size > rounded_size) {
        float betta1_minus1 = 1 - _betta1;
        float betta2_minus1 = 1 - _betta2;

        float step_size = -1 * _alpha / _bias_correction1;
        float w_decay = -1 * _alpha * _weight_decay;
        ds_half_precision_t* grads_cast_h;
        ds_half_precision_t* params_cast_h;

        if (half_precision) {
            grads_cast_h = reinterpret_cast<ds_half_precision_t*>(grads);
            params_cast_h = reinterpret_cast<ds_half_precision_t*>(_params);
        }

        for (size_t t = rounded_size; t < _param_size; t += TILE) {
            size_t copy_size = TILE;
            if ((t + TILE) > _param_size){
                copy_size = _param_size - t;
            }
            size_t offset = copy_size + t;
            // if ((t / TILE) >= 2) {
            //     aclrtSynchronizeStream(_streams[_buf_index]);
            // }

#pragma omp parallel for
            for (size_t k = t; k < offset; k++) {
                float grad = half_precision ? (float)grads_cast_h[k] : grads[k];
                float param = half_precision ? (float)params_cast_h[k] : _params[k];
                float momentum = _exp_avg[k];
                float variance = _exp_avg_sq[k];
                if (_weight_decay > 0 && !_adamw_mode) {
                    grad = param * _weight_decay + grad;
                }
                momentum = momentum * _betta1;
                momentum = grad * betta1_minus1 + momentum;

                variance = variance * _betta2;
                grad = grad * grad;
                variance = grad * betta2_minus1 + variance;

                grad = sqrt(variance);
                grad = grad * _bias_correction2 + _eps;
                grad = momentum / grad;
                if (_weight_decay > 0 && _adamw_mode) {
                    param += w_decay * param;
                }
                param = grad * step_size + param;
                if (dev_params){
                    _doubled_buffer[_buf_index][k - t] = (ds_half_precision_t)param;
                }

                if (half_precision){
                    params_cast_h[k] = (ds_half_precision_t)param;
                }else{
                    _params[k] = param;
                }
                _exp_avg[k] = momentum;
                _exp_avg_sq[k] = variance;
            }
            if (dev_params) {
                size_t MemoryCpySize = copy_size * sizeof(_doubled_buffer[_buf_index][0]);
                tpu::TPUCopyHostToDevice(dev_params + t, _doubled_buffer[_buf_index], MemoryCpySize);
                _buf_index = !_buf_index;
            }
        }
    }
}

void Adam_Optimizer::Step_4(
    float* _params,
    float* grads,
    float* _exp_avg,
    float* _exp_avg_sq,
    size_t _param_size,
    ds_half_precision_t* dev_params,
    bool half_precision
){
    size_t rounded_size = 0;

    if (_param_size > rounded_size)
        Step_1((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size),
               (dev_params != nullptr ? (dev_params + rounded_size) : dev_params),
               half_precision);
}

void Adam_Optimizer::Step_8(
    float* _params,
    float* grads,
    float* _exp_avg,
    float* _exp_avg_sq,
    size_t _param_size,
    ds_half_precision_t* dev_params,
    bool half_precision
){
    size_t rounded_size = 0;

    if (_param_size > rounded_size)
        Step_4((_params + rounded_size),
               (grads + rounded_size),
               (_exp_avg + rounded_size),
               (_exp_avg_sq + rounded_size),
               (_param_size - rounded_size),
               (dev_params != nullptr ? (dev_params + rounded_size) : dev_params),
               half_precision);
}

int create_adam_optimizer(
    int optimizer_id,
    float alpha = 1e-3,
    float betta1 = 0.9,
    float betta2 = 0.999,
    float eps = 1e-8,
    float weight_decay = 0,
    bool adamw_mode = true,
    bool should_log = false
){
    auto opt = std::make_shared<Adam_Optimizer>(alpha, betta1, betta2, eps, weight_decay, adamw_mode);

    s_optimizers[optimizer_id] = opt;

    if (should_log) {
        std::string avx_type = "";
        avx_type = "scalar";

        printf("Adam Optimizer #%d is created with %s arithmetic capability.\n",
               optimizer_id,
               avx_type.c_str());
        printf("Config: alpha=%f, betas=(%f, %f), weight_decay=%f, adam_w=%d\n",
               alpha,
               betta1,
               betta2,
               weight_decay,
               (int)adamw_mode);
    }

    return 0;
}



int ds_adam_step(
    int optimizer_id,
    size_t step,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    bool bias_correction,
    torch::Tensor& params,
    torch::Tensor& grads,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq
){
    auto params_c = params.contiguous();
    auto grads_c = grads.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();

    // assert(params.options().dtype() == grads.options().dtype());

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    std::shared_ptr<Adam_Optimizer> opt = std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, epsilon, weight_decay, bias_correction);

    opt->Step_8(params_ptr,
                grads_ptr,
                exp_avg_ptr,
                exp_avg_sq_ptr,
                params_c.size(0),
                nullptr,
                (params.options().dtype() == at::kHalf));

    opt->SynchronizeStreams();
    return 0;
}

int ds_adam_step_plus_copy(
    int optimizer_id,
    size_t step,
    float lr,
    float beta1,
    float beta2,
    float epsilon,
    float weight_decay,
    bool bias_correction,
    torch::Tensor& params,
    torch::Tensor& grads,
    torch::Tensor& exp_avg,
    torch::Tensor& exp_avg_sq,
    torch::Tensor& npu_params
){
    auto params_c = params.contiguous();
    auto npu_params_c = npu_params.contiguous();
    auto exp_avg_c = exp_avg.contiguous();
    auto exp_avg_sq_c = exp_avg_sq.contiguous();
    auto grads_c = grads.contiguous();

    float* params_ptr = (float*)params_c.data_ptr();
    float* grads_ptr = (float*)grads_c.data_ptr();
    ds_half_precision_t* npu_params_ptr = (ds_half_precision_t*)npu_params_c.data_ptr();
    float* exp_avg_ptr = (float*)exp_avg_c.data_ptr();
    float* exp_avg_sq_ptr = (float*)exp_avg_sq_c.data_ptr();

    std::shared_ptr<Adam_Optimizer> opt = std::static_pointer_cast<Adam_Optimizer>(s_optimizers[optimizer_id]);
    opt->IncrementStep(step, beta1, beta2);
    opt->update_state(lr, epsilon, weight_decay, bias_correction);
    opt->Step_8(params_ptr,
                grads_ptr,
                exp_avg_ptr,
                exp_avg_sq_ptr,
                params_c.size(0),
                npu_params_ptr,
                (params.options().dtype() == at::kHalf));

    opt->SynchronizeStreams();
    return 0;
}

int destroy_adam_optimizer(int optimizer_id){
    s_optimizers.erase(optimizer_id);
    return 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("adam_update", &ds_adam_step, "DeepSpeed CPU Adam update (C++)");
    m.def("adam_update_copy", &ds_adam_step_plus_copy, "DeepSpeed CPU Adam update and param copy (C++)");
    m.def("create_adam", &create_adam_optimizer, "DeepSpeed CPU Adam (C++)");
    m.def("destroy_adam", &destroy_adam_optimizer, "DeepSpeed CPU Adam destroy (C++)");
}
