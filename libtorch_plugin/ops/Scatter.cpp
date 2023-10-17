#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/EmptyTensor.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/DimVector.h>
#include <ATen/core/TensorBase.h>
#include <ATen/native/Copy.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/cpu/CatKernel.h>
#include <ATen/native/cpu/StackKernel.h>
#include <ATen/quantized/QTensorImpl.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/SmallVector.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <sgdnn_api.h>
#include <torch/library.h>
#include <torch/torch.h>

#include "common/config.h"

#define CONSTANT (0)
#define REFLECT (1)
#define SYMMETRIC (2)
#define REPLICATE (3)
#define CIRCULAR (4)

namespace at {
Tensor &slice_scatter_out_tpu(const Tensor &self, const Tensor &src,
                              int64_t dim, c10::optional<int64_t> start,
                              c10::optional<int64_t> end, int64_t step,
                              Tensor &out) {
  if (self.dim() > 0) {
    CHECK_TENSOR_IN_DEVICE(self);
  }
  CHECK_TENSOR_IN_DEVICE(out);
#if 0

#else
  TIMING_START;

  int start_value = start.has_value() ? start.value() : 0;
  int end_value = end.has_value() ? end.value() : self.size(dim);
  int num_c = 1;
  for (int i = 0; i < dim; i++){
    num_c *= self.size(i);
  }

  Tensor indices = arange(start_value,end_value,step,at::kInt).unsqueeze(0).unsqueeze(1).unsqueeze(-1).expand({1,num_c,-1,1}).to(self.device());
  bm_status_t status = sgdnnSliceScatter(
      tpu::TPUGetDeviceHandle(), tpu::TPUGenerateSgdnnTensor(self),
      tpu::TPUGenerateSgdnnTensor(src), tpu::TPUGenerateSgdnnTensor(indices),dim, 
      tpu::TPUGenerateSgdnnTensor(out));

  TORCH_CHECK(status == BM_SUCCESS);
  TIMING_END(tpu::SLICE_SCATTER);
#endif
  return out;
}

Tensor slice_scatter_tpu(const Tensor &self, const Tensor &src,
                              int64_t dim, c10::optional<int64_t> start,
                              c10::optional<int64_t> end, int64_t step) {
  Tensor out = empty_like(self);                              
  out = slice_scatter_out_tpu(self,src,dim,start,end,step,out);
  return out;
}

TORCH_LIBRARY_IMPL(aten, TPU, m) {
  m.impl("slice_scatter", slice_scatter_tpu);
  // m.impl("slice_scatter.out", slice_scatter_out_tpu);
}

}  // namespace at
