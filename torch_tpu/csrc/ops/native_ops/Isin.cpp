#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"
#include "common/config.h"

namespace at 
{
Tensor & isin_Tensor_Tensor_out_tpu(const Tensor & elements, const Tensor & test_elements,
                                    bool assume_unique, bool invert, Tensor & out)
{
#if 1
  CPU_IMPL_WARNING();
  TIMING_START;
  auto out_cpu = isin(elements.cpu(), test_elements.cpu(), assume_unique, invert);
  out = out_cpu.to(elements.device());
  TIMING_END(tpu::CPU_LAYER);
#else
#endif
  SHOW_TENSOR_OP(out);
  return out;
}

Tensor & isin_Tensor_Scalar_out_tpu(const Tensor & elements, const Scalar & test_element,
                                    bool assume_unique, bool invert, Tensor & out)
{
#if 1
  CPU_IMPL_WARNING();
  TIMING_START;
  auto out_cpu = isin(elements.cpu(), test_element, assume_unique, invert);
  out = out_cpu.to(elements.device());
  TIMING_END(tpu::CPU_LAYER);
#else
#endif
  SHOW_TENSOR_OP(out);
  return out;
}

Tensor & isin_Scalar_Tensor_out_tpu(const Scalar & element, const Tensor & test_elements,
                                    bool assume_unique, bool invert, Tensor & out)
{
#if 1
  CPU_IMPL_WARNING();
  TIMING_START;
  auto out_cpu = isin(element, test_elements.cpu(), assume_unique, invert);
  out = out_cpu.to(test_elements.device());
  TIMING_END(tpu::CPU_LAYER);
#else
#endif
  SHOW_TENSOR_OP(out);
  return out;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "isin.Tensor_Tensor_out", isin_Tensor_Tensor_out_tpu );
  m.impl ( "isin.Tensor_Scalar_out", isin_Tensor_Scalar_out_tpu );
  m.impl ( "isin.Scalar_Tensor_out", isin_Scalar_Tensor_out_tpu );
}
} // namespace at