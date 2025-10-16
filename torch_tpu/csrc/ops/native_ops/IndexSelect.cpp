#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>

#include "TPUTorchUtils.h"


#include "common/config.h"
#ifdef USING_PPL
#include "IndexSelect.h"
#define AT_DISPATCH_FLOAT_INT_TYPES(scalar_type, name, func)  \
AT_DISPATCH_SWITCH(                   \
scalar_type, name,                    \
AT_DISPATCH_CASE(at::kFloat, func)    \
AT_DISPATCH_CASE(at::kHalf, func)     \
AT_DISPATCH_CASE(at::kBFloat16, func) \
AT_DISPATCH_CASE(at::kInt, func)      \
AT_DISPATCH_CASE(at::kShort, func)    \
AT_DISPATCH_CASE(at::kChar, func)     \
AT_DISPATCH_CASE(at::kByte, func))

template <typename scalar_t>
static void index_select_async_impl(
  uint64_t input_addr,
  uint64_t index_addr,
  uint64_t output_addr,
  uint32_t outer_size,
  uint32_t inner_size,
  int gather_num,
  int gathered_num
  )
{
  auto kernel = [&](TPUStream stream, tpuKernelModule_t ppl_module) -> int {
    if constexpr (std::is_same_v<scalar_t, float>) {
        return indexselect_fp32(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr, index_addr, input_addr,
            outer_size, inner_size, gather_num, gathered_num
            );
    } else if constexpr (std::is_same_v<scalar_t, at::Half>) {
        return indexselect_fp16(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr, index_addr, input_addr,
            outer_size, inner_size, gather_num, gathered_num
            );
    } else if constexpr (std::is_same_v<scalar_t, at::BFloat16>) {
        return indexselect_bf16(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr, index_addr, input_addr,
            outer_size, inner_size, gather_num, gathered_num
            );
    } else if constexpr (std::is_same_v<scalar_t, int32_t>) {
        return indexselect_int32(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr, index_addr, input_addr,
            outer_size, inner_size, gather_num, gathered_num
            );
    } else if constexpr (std::is_same_v<scalar_t, int16_t>) {
        return indexselect_int16(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr, index_addr, input_addr,
            outer_size, inner_size, gather_num, gathered_num
            );
    } else if constexpr (std::is_same_v<scalar_t, int8_t>) {
        return indexselect_int8(
            stream,
#ifndef BACKEND_SG2260
            ppl_module,
#endif
            output_addr, index_addr, input_addr,
            outer_size, inner_size, gather_num, gathered_num
            );
    }
    return -1;
  };

	auto stream = c10_tpu::getCurrentTPUStream();
	tpuKernelModule_t ppl_module = getPplModule();
    int ret = kernel(stream, ppl_module);
    if (ret == 0) {
        return;
    }
	TORCH_CHECK(false, "index select failed !");
}
#endif
namespace at
{
Tensor& index_select_out_tpu ( const Tensor & self, int64_t dim, const Tensor & index, Tensor & out)
{
  TIMING_START;
  if ( self.numel() == 0) {return out;}
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( index );
#ifdef USING_PPL
  if (usePPLKernels()) {
    uint32_t outer_size = 1;
    uint32_t gather_dim = self.size(dim);
    uint32_t inner_size = 1;

    for (int i = 0; i < dim; ++i) {
        outer_size *= self.size(i);
    }
    for (int i = dim + 1; i < self.dim(); ++i) {
        inner_size *= self.size(i);
    }
    auto index_flat = index.contiguous().view(-1);
    auto index_int32 = index_flat.to(torch::kInt32).contiguous();
    AT_DISPATCH_FLOAT_INT_TYPES( self.scalar_type(), "index_select", [&] {
            index_select_async_impl<scalar_t>(
                reinterpret_cast<uint64_t>(self.data_ptr()),
                reinterpret_cast<uint64_t>(index_flat.data_ptr()),
                reinterpret_cast<uint64_t>(out.data_ptr()),
                outer_size, // C
                inner_size, // W
                gather_dim,
                index_int32.size(0)
                );
        });
      } else
#endif
  {  auto stream = c10_tpu::getCurrentTPUStream();
    auto status = tpudnnIndexSelectAsync(
                  stream,
                  tpu::TPUGenerateTpudnnTensor(stream, self),
                  tpu::TPUGenerateTpudnnTensor(stream, index),
                  dim,
                  tpu::TPUGenerateTpudnnTensor(stream, out));
    TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
  }
  TIMING_END;
  SHOW_TENSOR_OP(self, index, out);
  return out;
}

Tensor index_select_tpu ( const Tensor & self, int64_t origin_dim, const Tensor & index )
{
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( index );
  int64_t dim = maybe_wrap_dim(origin_dim, self.dim());
#if 0
  auto out_cpu = index_select ( self.cpu(), dim, index.cpu() );
  auto out = TENSOR_TO_TPU ( out_cpu );
#else
  TensorOptions options = TensorOptions ( self.device() ).dtype ( self.dtype() );
  std::vector<int64_t> sizes_vec;
  for ( int i = 0; i < dim; i++ ) {
    sizes_vec.push_back ( self.size ( i ) );
  }
  for ( int i = 0; i <  index.dim(); i++ ) {
    sizes_vec.push_back ( index.size ( i ) );
  }
  for ( int i = dim + 1; i < self.dim(); i++ ) {
    sizes_vec.push_back ( self.size ( i ) );
  }
  IntArrayRef sizes ( sizes_vec.data(), sizes_vec.size() );
  auto out = torch::empty ( sizes, options );
  index_select_out_tpu(self, dim, index, out);
#endif
  SHOW_TENSOR_OP(self, index, out);
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "index_select.out", index_select_out_tpu );
  m.impl ( "index_select", index_select_tpu );
}

Tensor embedding_dense_backward_tpu ( const Tensor & grad_output, const Tensor & indices, int64_t num_weights, int64_t padding_idx, bool scale_grad_by_freq )
{
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE ( grad_output );
  CHECK_TENSOR_IN_DEVICE ( indices );
#if 0
  auto out_cpu = embedding_dense_backward ( grad_output.cpu(), indices.cpu(), num_weights, padding_idx, scale_grad_by_freq );
  auto out = TENSOR_TO_TPU ( out_cpu );
#else
  TensorOptions out_option = TensorOptions ( grad_output.device() ).dtype ( grad_output.dtype() );
  torch::Tensor out = torch::empty ( {num_weights, grad_output.size ( grad_output.dim() - 1 ) }, out_option );
  // indices should not be int64_t
  auto indices_int32 = indices.to ( torch::kInt32 );
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnEmbeddingBackwardAsync(
    stream,
    tpu::TPUGenerateTpudnnTensor(stream, grad_output),
    tpu::TPUGenerateTpudnnTensor(stream, indices_int32),
    tpu::TPUGenerateTpudnnTensor(stream, out));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
  TIMING_END;
  SHOW_TENSOR_OP(grad_output, indices, out);
  return out;
}
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "embedding_dense_backward", embedding_dense_backward_tpu );
}

Tensor masked_select_tpu(const at::Tensor & self, const at::Tensor & mask) {
  TIMING_START;
  CHECK_TENSOR_IN_DEVICE ( self );
  CHECK_TENSOR_IN_DEVICE ( mask );
#if 0
  LOG( WARNING ) << __func__ << " use cpu impl.";
  auto out_cpu = masked_select( self.cpu(), mask.cpu() );
  auto out = TENSOR_TO_TPU ( out_cpu );
#else
  int64_t num_nonzeros = torch::sum( mask ).item().toInt();
  std::vector<int64_t> out_sizes = {num_nonzeros};
  for (int i = mask.dim(); i < self.dim(); i++) { out_sizes.push_back( self.size(i) ); }
  auto out = torch::empty(out_sizes, self.options());
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnMaskedSelectAsync(
      stream,
      tpu::TPUGenerateTpudnnTensor(stream, self),
      tpu::TPUGenerateTpudnnTensor(stream, mask),
      tpu::TPUGenerateTpudnnTensor(stream, out));
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
#endif
  TIMING_END;
  return out;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
  m.impl ( "masked_select", masked_select_tpu );
}

} // namespace at
