#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <ATen/ATen.h>
#include <ATen/AccumulateType.h>
#include <ATen/ExpandUtils.h>
#include <ATen/InferSize.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/NamedTensorUtils.h>
#include <ATen/NativeFunctions.h>
#include <ATen/SparseCsrTensorUtils.h>
#include <ATen/SparseTensorUtils.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/core/DimVector.h>
#include <ATen/core/IListRef.h>
#include <ATen/native/Copy.h>
#include <ATen/native/Resize.h>
#include <ATen/native/TensorIterator.h>
#include <ATen/native/TensorShape.h>
#include <ATen/native/TypeProperties.h>
#include <ATen/native/cpu/CatKernel.h>
#include <ATen/native/cpu/SerialStackImpl.h>
#include <ATen/native/cpu/StackKernel.h>
#include <ATen/quantized/QTensorImpl.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <c10/util/SmallVector.h>
#include <c10/util/accumulate.h>
#include <c10/util/irange.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#define TPU_OP_TIMING

namespace at
{
//
// templated for ArrayRef<int64_t> and SmallVector<int64_t> use cases
//
template <typename Vec>
Tensor alias_with_sizes_and_strides (
const Tensor& self,
const Vec& sizes,
const Vec& strides ) {
  //caller should make sure that sizes and strides are valid for self
  //(storage is sufficient, strides are non-negative, strides and sizes array size is the same)
  Tensor self_;
  if ( self.is_quantized() ) {
    self_ = at::detail::make_tensor<QTensorImpl> (
            c10::TensorImpl::VIEW, Storage ( self.storage() ), self.key_set(), self.dtype(), get_qtensorimpl ( self )->quantizer() );
    auto* self_tmp_ = self_.unsafeGetTensorImpl();
    self_tmp_->set_storage_offset ( self.storage_offset() );
    self_tmp_->set_sizes_and_strides ( sizes, strides );
  } else {
    self_ = at::detail::make_tensor<TensorImpl> (
            c10::TensorImpl::VIEW, Storage ( self.storage() ), self.key_set(), self.dtype() );
    auto* self_tmp_ = self_.unsafeGetTensorImpl();
    self_tmp_->set_storage_offset ( self.storage_offset() );
    self_tmp_->set_sizes_and_strides ( sizes, strides );
  }
  namedinference::propagate_names ( self_, self );
  return self_;
}

Tensor view_tpu ( const Tensor & self, c10::IntArrayRef size )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  TORCH_CHECK ( self.is_contiguous() );
  at::DimVector inferred_size = at::infer_size_dv ( size, self.numel() );
  auto stride = at::detail::computeStride ( self.sizes(), self.strides(), inferred_size );
  TORCH_CHECK ( stride.has_value(), "view size is "
                "not compatible with input tensor's size and stride (at least one dimension"
                " spans across two contiguous subspaces). Use .reshape(...) instead." );
  return alias_with_sizes_and_strides ( self, inferred_size, *stride );
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "view", view_tpu );
}

Tensor _reshape_alias_tpu ( const Tensor & input,
                            IntArrayRef    sizes,
                            IntArrayRef    strides )
{
  return view_tpu ( input, sizes );
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "_reshape_alias", _reshape_alias_tpu );
}

Tensor as_strided_tpu ( const Tensor & self, IntArrayRef size, IntArrayRef stride, c10::optional<int64_t> storage_offset )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  Tensor out;
  if ( self.sizes() == size && self.strides() == stride )
  {
    out = self.detach();
  }
  else
  {
#ifdef TPU_OP_TIMING
    auto timer = tpu::Timer().Start();
#endif
    auto out_cpu = as_strided ( self.cpu(), size, stride, storage_offset );
    out = out_cpu.contiguous().to ( tpu::TPUGetCurrentDevice() );
#ifdef TPU_OP_TIMING
    tpu::OpTimer::Instance().AddTime ( tpu::PERMUTE, timer.ElapsedUS() );
#endif
#if 0
    std::cout << "self.shape = " << self.sizes() << std::endl;
    std::cout << "self.stride = " << self.strides() << std::endl;
    std::cout << "size = " << size << std::endl;
    std::cout << "stride = " << stride << std::endl;
    std::cout << "out.shape = " << out.sizes() << std::endl;
    std::cout << "**********************************************" << std::endl;
#endif
  }
  return out;
}
TORCH_LIBRARY_IMPL ( aten, PrivateUse1, m )
{
  m.impl ( "as_strided", as_strided_tpu );
}

} // namespace at
