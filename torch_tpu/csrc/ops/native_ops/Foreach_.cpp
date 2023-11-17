#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/native/ForeachUtils.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>
#include "common/config.h"
namespace at {

using namespace native;

#define FOREACH_UNARY_OP(OP)                                           \
  std::vector<Tensor> foreach_tensor_##OP##_tpu(TensorList tensors) {  \
    check_foreach_api_restrictions(tensors);                           \
                                                                       \
    std::vector<Tensor> result;                                        \
    result.reserve(tensors.size());                                    \
    for (const auto& t : tensors) {                                    \
      result.emplace_back(t.OP());                                     \
    }                                                                  \
                                                                       \
    return result;                                                     \
  }                                                                    \
                                                                       \
  void foreach_tensor_##OP##_tpu_(TensorList tensors) {                \
    check_foreach_api_restrictions(tensors);                           \
                                                                       \
    for (auto& t : tensors) {                                          \
      t.OP##_();                                                       \
    }                                                                  \
  }                                                                    \
TORCH_LIBRARY_IMPL ( aten, TPU, m )                                    \
{                                                                      \
  m.impl ( "_foreach_"#OP , foreach_tensor_##OP##_tpu);                \
  m.impl ( "_foreach_"#OP"_", foreach_tensor_##OP##_tpu_ );            \
}

#define FOREACH_BINARY_OP_TENSOR(OP)                            \
  void foreach_tensor_##OP##_tensor_kernel_tpu_(               \
      TensorList tensors, const Tensor& scalar) {               \
    TORCH_CHECK(                                                \
        scalar.dim() == 0 && scalar.numel() == 1,               \
        "scalar tensor expected to be 0 dim but it has ",       \
        scalar.dim(),                                           \
        " dimensions and ",                                     \
        scalar.numel(),                                         \
        " elements.");                                          \
    check_foreach_api_restrictions(tensors);                    \
                                                                \
    for (auto& t : tensors) {                                   \
      t.OP##_(scalar);                                          \
    }                                                           \
  }                                                             \
                                                                \
  std::vector<Tensor> foreach_tensor_##OP##_tensor_kernel_tpu( \
      TensorList tensors, const Tensor& scalar) {               \
    TORCH_CHECK(                                                \
        scalar.dim() == 0 && scalar.numel() == 1,               \
        "scalar tensor expected to be 0 dim but it has ",       \
        scalar.dim(),                                           \
        " dimensions and ",                                     \
        scalar.numel(),                                         \
        " elements.");                                          \
    check_foreach_api_restrictions(tensors);                    \
                                                                \
    std::vector<Tensor> result;                                 \
    result.reserve(tensors.size());                             \
    for (const auto& t : tensors) {                             \
      result.emplace_back(t.OP(scalar));                        \
    }                                                           \
                                                                \
    return result;                                              \
  }                                                             \
TORCH_LIBRARY_IMPL ( aten, TPU, m )                                             \
{                                                                               \
  m.impl ( "_foreach_"#OP".Tensor" , foreach_tensor_##OP##_tensor_kernel_tpu);  \
  m.impl ( "_foreach_"#OP"_.Tensor", foreach_tensor_##OP##_tensor_kernel_tpu_); \
}


#define FOREACH_BINARY_OP_SCALAR(OP)                            \
  void foreach_tensor_##OP##_scalar_kernel_tpu_(                \
      TensorList tensors, const Scalar& scalar) {               \
    check_foreach_api_restrictions(tensors);                    \
                                                                \
    for (auto& t : tensors) {                                   \
      t.OP##_(scalar);                                          \
    }                                                           \
  }                                                             \
                                                                \
  std::vector<Tensor> foreach_tensor_##OP##_scalar_kernel_tpu(  \
      TensorList tensors, const Scalar& scalar) {               \
    check_foreach_api_restrictions(tensors);                    \
                                                                \
    std::vector<Tensor> result;                                 \
    result.reserve(tensors.size());                             \
    for (const auto& t : tensors) {                             \
      result.emplace_back(t.OP(scalar));                        \
    }                                                           \
                                                                \
    return result;                                              \
  }                                                             \
TORCH_LIBRARY_IMPL ( aten, TPU, m )                                             \
{                                                                               \
  m.impl ( "_foreach_"#OP".Scalar" , foreach_tensor_##OP##_scalar_kernel_tpu);  \
  m.impl ( "_foreach_"#OP"_.Scalar", foreach_tensor_##OP##_scalar_kernel_tpu_); \
}


#define FOREACH_BINARY_OP_LIST(OP)                            \
  std::vector<Tensor> foreach_tensor_##OP##_list_kernel_tpu(  \
      TensorList tensors1, TensorList tensors2) {             \
    check_foreach_api_restrictions(tensors1, tensors2);       \
                                                              \
    std::vector<Tensor> result;                               \
    result.reserve(tensors1.size());                          \
    for (const auto i : c10::irange(tensors1.size())) {       \
      result.emplace_back(tensors1[i].OP(tensors2[i]));       \
    }                                                         \
                                                              \
    return result;                                            \
  }                                                           \
                                                              \
  void foreach_tensor_##OP##_list_kernel_tpu_(               \
      TensorList tensors1, TensorList tensors2) {             \
    check_foreach_api_restrictions(tensors1, tensors2);       \
                                                              \
    for (const auto i : c10::irange(tensors1.size())) {       \
      tensors1[i].OP##_(tensors2[i]);                         \
    }                                                         \
  }                                                           \
TORCH_LIBRARY_IMPL ( aten, TPU, m )                                         \
{                                                                           \
  m.impl ( "_foreach_"#OP".List" , foreach_tensor_##OP##_list_kernel_tpu);  \
  m.impl ( "_foreach_"#OP"_.List", foreach_tensor_##OP##_list_kernel_tpu_); \
}


#define FOREACH_BINARY_OP_LIST_ALPHA(OP)                               \
  std::vector<Tensor> foreach_tensor_##OP##_list_kernel_tpu(          \
      TensorList tensors1, TensorList tensors2, const Scalar& alpha) { \
    check_foreach_api_restrictions(tensors1, tensors2);                \
                                                                       \
    std::vector<Tensor> result;                                        \
    result.reserve(tensors1.size());                                   \
    for (const auto i : c10::irange(tensors1.size())) {                \
      result.emplace_back(tensors1[i].OP(tensors2[i], alpha));         \
    }                                                                  \
                                                                       \
    return result;                                                     \
  }                                                                    \
                                                                       \
  void foreach_tensor_##OP##_list_kernel_tpu_(                         \
      TensorList tensors1, TensorList tensors2, const Scalar& alpha) { \
    check_foreach_api_restrictions(tensors1, tensors2);                \
                                                                       \
    for (const auto i : c10::irange(tensors1.size())) {                \
      tensors1[i].OP##_(tensors2[i], alpha);                           \
    }                                                                  \
  }                                                                    \
TORCH_LIBRARY_IMPL ( aten, TPU, m )                                         \
{                                                                           \
  m.impl ( "_foreach_"#OP".List" , foreach_tensor_##OP##_list_kernel_tpu);  \
  m.impl ( "_foreach_"#OP"_.List", foreach_tensor_##OP##_list_kernel_tpu_); \
}


#define FOREACH_BINARY_OP_SCALARLIST(OP)                            \
  void foreach_tensor_##OP##_scalarlist_kernel_tpu_(               \
      TensorList tensors, at::ArrayRef<Scalar> scalars) {           \
    check_foreach_api_restrictions(tensors, scalars);               \
                                                                    \
    for (const auto i : c10::irange(tensors.size())) {              \
      tensors[i].OP##_(scalars[i]);                                 \
    }                                                               \
  }                                                                 \
                                                                    \
  std::vector<Tensor> foreach_tensor_##OP##_scalarlist_kernel_tpu( \
      TensorList tensors, at::ArrayRef<Scalar> scalars) {           \
    check_foreach_api_restrictions(tensors, scalars);               \
    std::vector<Tensor> result;                                     \
    result.reserve(tensors.size());                                 \
    for (const auto i : c10::irange(tensors.size())) {              \
      result.emplace_back(tensors[i].OP(scalars[i]));               \
    }                                                               \
                                                                    \
    return result;                                                  \
  }                                                                 \
TORCH_LIBRARY_IMPL ( aten, TPU, m )                                                     \
{                                                                                       \
  m.impl ( "_foreach_"#OP".ScalarList" , foreach_tensor_##OP##_scalarlist_kernel_tpu);  \
  m.impl ( "_foreach_"#OP"_.ScalarList", foreach_tensor_##OP##_scalarlist_kernel_tpu_); \
}

#define FOREACH_POINTWISE_OP_SCALAR(OP)                                   \
  std::vector<Tensor> foreach_tensor_##OP##_scalar_tpu(                  \
      TensorList input,                                                   \
      TensorList tensors1,                                                \
      TensorList tensors2,                                                \
      const Scalar& scalar) {                                             \
    check_foreach_api_restrictions(input, tensors1, tensors2);            \
                                                                          \
    std::vector<Tensor> result;                                           \
    for (const auto i : c10::irange(input.size())) {                      \
      result.emplace_back(input[i].OP(tensors1[i], tensors2[i], scalar)); \
    }                                                                     \
                                                                          \
    return result;                                                        \
  }                                                                       \
                                                                          \
  void foreach_tensor_##OP##_scalar_tpu_(                                \
      TensorList input,                                                   \
      TensorList tensors1,                                                \
      TensorList tensors2,                                                \
      const Scalar& scalar) {                                             \
    check_foreach_api_restrictions(input, tensors1, tensors2);            \
                                                                          \
    for (const auto i : c10::irange(input.size())) {                      \
      input[i].OP##_(tensors1[i], tensors2[i], scalar);                   \
    }                                                                     \
  }                                                                       \
TORCH_LIBRARY_IMPL ( aten, TPU, m )                                                     \
{                                                                                       \
  m.impl ( "_foreach_"#OP".Scalar" , foreach_tensor_##OP##_scalar_tpu);  \
  m.impl ( "_foreach_"#OP"_.Scalar", foreach_tensor_##OP##_scalar_tpu_); \
}

#define FOREACH_POINTWISE_OP_SCALARLIST(OP)                                   \
  std::vector<Tensor> foreach_tensor_##OP##_scalarlist_tpu(                  \
      TensorList input,                                                       \
      TensorList tensors1,                                                    \
      TensorList tensors2,                                                    \
      at::ArrayRef<Scalar> scalars) {                                         \
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);       \
                                                                              \
    std::vector<Tensor> result;                                               \
    for (const auto i : c10::irange(input.size())) {                          \
      result.emplace_back(input[i].OP(tensors1[i], tensors2[i], scalars[i])); \
    }                                                                         \
                                                                              \
    return result;                                                            \
  }                                                                           \
                                                                              \
  void foreach_tensor_##OP##_scalarlist_tpu_(                                \
      TensorList input,                                                       \
      TensorList tensors1,                                                    \
      TensorList tensors2,                                                    \
      at::ArrayRef<Scalar> scalars) {                                         \
    check_foreach_api_restrictions(input, tensors1, tensors2, scalars);       \
                                                                              \
    for (const auto i : c10::irange(input.size())) {                          \
      input[i].OP##_(tensors1[i], tensors2[i], scalars[i]);                   \
    }                                                                         \
  }                                                                           \
TORCH_LIBRARY_IMPL ( aten, TPU, m )                                                     \
{                                                                                       \
  m.impl ( "_foreach_"#OP".ScalarList" , foreach_tensor_##OP##_scalarlist_tpu);  \
  m.impl ( "_foreach_"#OP"_.ScalarList", foreach_tensor_##OP##_scalarlist_tpu_); \
}


FOREACH_UNARY_OP(neg);
FOREACH_UNARY_OP(sqrt);
FOREACH_UNARY_OP(reciprocal);

FOREACH_BINARY_OP_TENSOR(mul);

FOREACH_BINARY_OP_SCALAR(mul);
FOREACH_BINARY_OP_SCALAR(div);
FOREACH_BINARY_OP_SCALAR(pow);
FOREACH_BINARY_OP_SCALAR(add);

FOREACH_BINARY_OP_LIST(mul);
FOREACH_BINARY_OP_LIST(div);
FOREACH_BINARY_OP_LIST(pow);


FOREACH_BINARY_OP_LIST_ALPHA(add);
FOREACH_BINARY_OP_LIST_ALPHA(sub);

FOREACH_BINARY_OP_SCALARLIST(div);
FOREACH_BINARY_OP_SCALARLIST(pow);


FOREACH_POINTWISE_OP_SCALAR(addcmul);
FOREACH_POINTWISE_OP_SCALAR(addcdiv);

FOREACH_POINTWISE_OP_SCALARLIST(addcmul);
FOREACH_POINTWISE_OP_SCALARLIST(addcdiv);



#define FOREACH_BINARY_OP_LIST_SCALAR(OP)                               \
  std::vector<Tensor> foreach_tensor_##OP##_list_kernel_tpu(          \
      TensorList tensors1, TensorList tensors2, const Scalar& alpha) { \
    check_foreach_api_restrictions(tensors1, tensors2);                \
                                                                       \
    std::vector<Tensor> result;                                        \
    result.reserve(tensors1.size());                                   \
    for (const auto i : c10::irange(tensors1.size())) {                \
      result.emplace_back(tensors1[i].OP(tensors2[i], alpha));         \
    }                                                                  \
                                                                       \
    return result;                                                     \
  }                                                                    \
                                                                       \
  void foreach_tensor_##OP##_list_kernel_tpu_(                         \
      TensorList tensors1, TensorList tensors2, const Scalar& alpha) { \
    check_foreach_api_restrictions(tensors1, tensors2);                \
                                                                       \
    for (const auto i : c10::irange(tensors1.size())) {                \
      tensors1[i].OP##_(tensors2[i], alpha);                           \
    }                                                                  \
  }                                                                    \
TORCH_LIBRARY_IMPL ( aten, TPU, m )                                         \
{                                                                           \
  m.impl ( "_foreach_"#OP".Sclar" , foreach_tensor_##OP##_list_kernel_tpu);  \
  m.impl ( "_foreach_"#OP"_.Scalar", foreach_tensor_##OP##_list_kernel_tpu_); \
}

FOREACH_BINARY_OP_LIST_SCALAR(lerp);
}