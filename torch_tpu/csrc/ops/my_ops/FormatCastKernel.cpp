// FormatCast OP: modify storage Format
#include "torch_tpu/csrc/aten/TPUFormatCastHelper.h"
#include "torch_tpu/csrc/core/TPUStorageImpl.h"
#include "torch_tpu/csrc/core/TPULog.h"
#include "torch_tpu/csrc/core/TPUDeviceUtils.h"
#include "torch_tpu/csrc/core/TPUTorchUtils.h"
#include "common/config.h"
#include <functional>

using tensor_list = std::vector<at::Tensor>;

namespace at_tpu
{
namespace TPUNativeFunctions {

int64_t get_tpu_format(const at::Tensor & self){
  torch_tpu::utils::is_tpu(self);
  auto self_desc = torch_tpu::GetTpuStorageImpl(self)->tpu_desc_;
  int64_t self_format = self_desc.tpu_format_;
  return self_format;
}

int64_t get_origin_format(const at::Tensor & self){
  torch_tpu::utils::is_tpu(self);
  auto self_desc = torch_tpu::GetTpuStorageImpl(self)->tpu_desc_;
  int64_t self_format = self_desc.origin_format_;
  return self_format;
}
namespace {
  typedef enum {
    FORMATCAST_UNSUPPORT = -1,//
    Conv_W_ND_TO_32IC = 0,    // conv fp16 weight contiguous to 32IC
    Conv_W_32IC_TO_ND = 1,    // conv fp16 32IC weight to contiguous format(ND),
    Conv_W_ND_TO_32IC32OC = 2, // conv fp16 weight contiguous to 32IC32OC (train mode)
    Conv_W_32IC32OC_TO_ND = 3, // conv fp16 32IC32OC weight to contiguous format(ND)
    Conv_DW_32OC_TO_ND    = 4, // conv fp16 32OC weight'grad to contiguous format(ND)
    Conv_DW_ND_TO_32OC    = 5, // conv fp16 contiguous format(ND) weight'grad to 32OC
  }FORMATCAST_TYPE;
  std::map<FORMATCAST_TYPE, std::string> _FormatCast_Str = {
    {FORMATCAST_UNSUPPORT,  "unsupport formatcast"},
    {Conv_W_ND_TO_32IC,     "Conv_W_ND_TO_32IC"},
    {Conv_W_32IC_TO_ND,     "Conv_W_32IC_TO_ND"},
    {Conv_W_ND_TO_32IC32OC, "Conv_W_ND_TO_32IC32OC"},
    {Conv_W_32IC32OC_TO_ND, "Conv_W_32IC32OC_TO_ND"},
    {Conv_DW_32OC_TO_ND,    "Conv_DW_32OC_TO_ND"},
    {Conv_DW_ND_TO_32OC,    "Conv_DW_ND_TO_32OC"},
    };

  FORMATCAST_TYPE formatcast_type( const at::Tensor& src, int64_t dst_format )
  {
    int64_t src_format = get_tpu_format(src);
    caffe2::TypeMeta src_dtype = src.dtype();
    if ( (src_format == torch_tpu::TPU_DFORMAT_ND && dst_format == torch_tpu::TPU_DFORMAT_CONV_W_Infer)
        && (src_dtype == caffe2::TypeMeta::Make<at::Half>() || src_dtype == caffe2::TypeMeta::Make<at::BFloat16>()))
    {
      return Conv_W_ND_TO_32IC;
    }
    else if ( (src_format == torch_tpu::TPU_DFORMAT_CONV_W_Infer && dst_format == torch_tpu::TPU_DFORMAT_ND)
        && (src_dtype == caffe2::TypeMeta::Make<at::Half>() || src_dtype == caffe2::TypeMeta::Make<at::BFloat16>()))
    {
      return Conv_W_32IC_TO_ND;
    }
    else if ((src_format == torch_tpu::TPU_DFORMAT_ND && dst_format == torch_tpu::TPU_DFORMAT_CONV_W_Train)
        && (src_dtype == caffe2::TypeMeta::Make<at::Half>() || src_dtype == caffe2::TypeMeta::Make<at::BFloat16>()))
    {
      return Conv_W_ND_TO_32IC32OC;
    }
    else if ((src_format == torch_tpu::TPU_DFORMAT_CONV_W_Train && dst_format == torch_tpu::TPU_DFORMAT_ND)
        && (src_dtype == caffe2::TypeMeta::Make<at::Half>() || src_dtype == caffe2::TypeMeta::Make<at::BFloat16>()))
    {
      return Conv_W_32IC32OC_TO_ND;
    }
    else if ((src_format == torch_tpu::TPU_DFORMAT_ND && dst_format == torch_tpu::TPU_DFORMAT_CONV_DW)
        && (src_dtype == caffe2::TypeMeta::Make<at::Half>() || src_dtype == caffe2::TypeMeta::Make<at::BFloat16>()))
    {
      return Conv_DW_ND_TO_32OC;
    }
    else if ((src_format == torch_tpu::TPU_DFORMAT_CONV_DW && dst_format == torch_tpu::TPU_DFORMAT_ND)
        && (src_dtype == caffe2::TypeMeta::Make<at::Half>() || src_dtype == caffe2::TypeMeta::Make<at::BFloat16>()))
    {
      return Conv_DW_32OC_TO_ND;
    }
    else {
      return FORMATCAST_UNSUPPORT;
    }
  }
}

at::Tensor& format_cast_impl_out_tpu(at::Tensor& dst, const at::Tensor& src, FORMATCAST_TYPE cast_type) {
  auto handle = c10_tpu::getCurrentTPUStream();
  auto src_tensor = tpu::TPUGenerateTpudnnTensor(handle, src);
  auto dst_tensor = tpu::TPUGenerateTpudnnTensor(handle, dst);
  auto status = TPUDNN_STATUS_SUCCESS;
  if (cast_type == Conv_W_ND_TO_32IC || cast_type == Conv_W_32IC_TO_ND) {
    status = tpudnnReorderConv2dWeightAsync(handle, src_tensor, 0, dst_tensor);
  } else if (cast_type == Conv_W_ND_TO_32IC32OC ||
             cast_type == Conv_W_32IC32OC_TO_ND) {
    status = tpudnnReorderConv2dWeightAsync(handle, src_tensor, 2, dst_tensor);
  } else if (cast_type == Conv_DW_32OC_TO_ND ||
             cast_type == Conv_DW_ND_TO_32OC) {
    status = tpudnnReorderConv2dGradAsync(handle, src_tensor, dst_tensor);
  } else {
    TORCH_CHECK(0, "NO SUPPORT SUCH Format Cast.");
  }
  TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS, "FormatCast Failed.");
  return dst;
}

// conver self to tpu_format, write the result into new result tensor
at::Tensor tpu_format_cast_impl(
    const at::Tensor& src,
    int64_t tpu_format) {
  auto src_desc = torch_tpu::GetTpuStorageImpl(src)->tpu_desc_;
  if (src_desc.tpu_format_ == tpu_format) {
    LOG( INFO ) << "no need to do format cast, because of Same FormatCast\n";
    return src;
  }
  FORMATCAST_TYPE cast_type = formatcast_type(src, tpu_format);
  if (cast_type == FORMATCAST_UNSUPPORT) {
    LOG( INFO ) << "no need to do format cast, because of UnSupport FormatCast\n";
    return src;
  }

  at::Tensor dst = at_tpu::FormatCastPreparation::apply_tensor_with_format(
      src_desc.base_sizes_, src.options(), tpu_format);

  // calculate the output result of the TPU
  format_cast_impl_out_tpu(dst, src, cast_type);

  // format cast only change physical layout of base tensor and view tensor's
  // metadata remain unchanged
  dst.set_(dst.storage(), src.storage_offset(), src.sizes(), src.strides());
  return dst;
}

at::Tensor tpu_format_cast(const at::Tensor& self,
    int64_t tpu_format) {
  torch_tpu::utils::is_tpu(self);
  if (get_tpu_format(self) == tpu_format) {
    LOG( WARNING) << "no need to do format cast";
    return self;
  }
  return tpu_format_cast_impl(self, tpu_format);
}
// conver self to dst'format, write the result into new result tensor
at::Tensor tpu_format_cast(
    const at::Tensor& src,
    const at::Tensor& dst) {
  torch_tpu::utils::is_tpu(dst);
  auto dst_desc = torch_tpu::GetTpuStorageImpl(dst)->tpu_desc_;
  int64_t dst_format = dst_desc.tpu_format_;
  return tpu_format_cast(src, dst_format);
}

at::Tensor tpu_format_cast_back_to_origin(
    const at::Tensor& self) {
  torch_tpu::utils::is_tpu(self);
  int64_t tpu_format = get_tpu_format(self);
  int64_t ori_format = get_origin_format(self);
  if ( tpu_format == ori_format ) {
    LOG( WARNING) << "no need to do format cast";
    return self;
  }
  return tpu_format_cast_impl(self, ori_format);
}

} // namespace TPUNativeFunctions
} // namespace at_tpu
namespace at
{
void FormatCast(const at::Tensor &self, const at::Tensor &dst,
                int64_t tpu_format) {
  // TODO check something later
  auto format_type = at_tpu::TPUNativeFunctions::FORMATCAST_TYPE(tpu_format);
  at_tpu::TPUNativeFunctions::format_cast_impl_out_tpu(const_cast<at::Tensor&>(dst), self, format_type);
}
} // namespace at