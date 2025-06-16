#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include "TPUTorchUtils.h"
#include "common/config.h"

namespace at {
    Tensor noaux_tc_topk(
        Tensor &values,
        Tensor &indices,
        const Tensor &scores,
        int64_t n_groups,
        int64_t topk_groups,
        int64_t top_k) {
        
        // 第一步：输入验证
        CHECK_TENSOR_IN_DEVICE(values);
        CHECK_TENSOR_IN_DEVICE(indices);
        CHECK_TENSOR_IN_DEVICE(scores);
        
        // 检查数据类型 - values 和 scores 应该有相同的数据类型
        TORCH_CHECK(values.dtype() == scores.dtype(), 
                   "values and scores must have the same dtype");
        
        // 检查 indices 的数据类型应该是 int32
        TORCH_CHECK(indices.dtype() == torch::kInt32, 
                   "indices must be int32");
        
        // 第二步：获取张量信息
        auto scores_shape = scores.sizes();
        int64_t batch_size = scores_shape[0];
        int64_t total_size = scores_shape[1];
        
        // 验证形状是否符合预期
        TORCH_CHECK(scores.dim() == 2, "scores must be 2D tensor");
        TORCH_CHECK(values.size(0) == batch_size && values.size(1) == top_k,
                   "values shape must be [batch_size, top_k]");
        TORCH_CHECK(indices.size(0) == batch_size && indices.size(1) == top_k,
                   "indices shape must be [batch_size, top_k]");
        
        // 验证参数合理性
        TORCH_CHECK(total_size % n_groups == 0, 
                   "total_size must be divisible by n_groups");
        TORCH_CHECK(topk_groups <= n_groups, 
                   "topk_groups must be <= n_groups");
        TORCH_CHECK(top_k > 0, "top_k must be positive");
        
        // 第三步：调用底层 DNN 接口
        TIMING_START;
        auto stream = c10_tpu::getCurrentTPUStream();
        auto status = tpudnnNoauxTcTopkAsync(
            stream,
            tpu::TPUGenerateTpudnnTensor(stream, values),
            tpu::TPUGenerateTpudnnTensor(stream, indices),
            tpu::TPUGenerateTpudnnTensor(stream, scores),
            batch_size,
            total_size,
            n_groups,
            topk_groups,
            top_k);
        TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS, 
                   "tpudnnNoauxTcTopkAsync failed");
        TIMING_END(tpu::NOAUX_TC_TOPK);
        
        return values;
    }
} 