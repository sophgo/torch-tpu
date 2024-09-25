#include "torch_tpu/csrc/core/TPUTorchUtils.h"
#include "torch_tpu/csrc/aten/TPUFormatCastHelper.h"
#include "torch_tpu/csrc/aten/TPUNativeFunctions.h"
#include <sgdnn_api.h>

namespace at_tpu
{

/**************************************************************
*****************   TPU Storage Desc Helper   *****************
***************************************************************/
namespace {
  constexpr int BLOCKSIZE = 16;
  // base format is ND
  FormatShape InferShapeConv_W_32IC(c10::IntArrayRef dims){ //  CONV2D 32IC
    AT_ASSERT(dims.size() == 4, "32IC FormatCast must with dims == 4, but got ", dims.size());
    std::vector<int> dims_vec(dims.begin(), dims.end()), res_vec(4,0);
    sgdnn32ICShape( dims_vec.data(), res_vec.data());
    FormatShape res(res_vec.begin(), res_vec.end());
    return res;
  }
  FormatShape InferShapeConv_W_32IC32OC(c10::IntArrayRef dims)
  {
    AT_ASSERT(dims.size() == 4, "32IC FormatCast must with dims == 4, but got ", dims.size());
    std::vector<int> dims_vec(dims.begin(), dims.end()), res_vec(8,0);
    sgdnn32ICShape( dims_vec.data(), res_vec.data());
    sgdnn32OCShape( dims_vec.data(), res_vec.data() + 4);
    FormatShape res(res_vec.begin(), res_vec.end());
    return res;
  }
  FormatShape InferShapeConv_DW_32OC(c10::IntArrayRef dims)
  {
    AT_ASSERT(dims.size() == 4, "32IC FormatCast must with dims == 4, but got ", dims.size());
    std::vector<int> dims_vec(dims.begin(), dims.end()), res_vec(4,0);
    sgdnn32OCShape( dims_vec.data(), res_vec.data());
    FormatShape res(res_vec.begin(), res_vec.end());
    return res;
  }
  FormatShape InferShapeofND(c10::IntArrayRef dims)
  {
    FormatShape res;
    res.resize(dims.size());
    for (size_t j = 0; j < dims.size(); j++)
    {
      res[j] = dims[j];
    }
    return res;
  }
} // StorageShapeInfer
std::unordered_map<tpuFormat, StorageDescHelper::FormatInfo> StorageDescHelper::info_ = {
  {TPU_DFORMAT_ND, (FormatInfo){TPU_DFORMAT_ND, TPU_DFORMAT_ND, InferShapeofND, "ND"}},
  {TPU_DFORMAT_CONV_W_Infer, (FormatInfo){TPU_DFORMAT_CONV_W_Infer, TPU_DFORMAT_ND, InferShapeConv_W_32IC, "CONV_W_Infer"}},
  {TPU_DFORMAT_CONV_W_Train, (FormatInfo){TPU_DFORMAT_CONV_W_Train, TPU_DFORMAT_ND, InferShapeConv_W_32IC32OC, "CONV_W_Train"}},
  {TPU_DFORMAT_CONV_DW, (FormatInfo){TPU_DFORMAT_CONV_DW, TPU_DFORMAT_ND, InferShapeConv_DW_32OC, "CONV_DW"}},
};


// ------------------------------------ Get
template <typename sizeType>
FormatShape StorageDescHelper::GetStorageSizes(tpuFormat format, sizeType ori_size)
{
  auto itr = info_.find(format);
  if (itr != info_.end())
  {
  if (itr->second.func)
  {
      return itr->second.func(ori_size);
  }
  }
  AT_ERROR("unsupport InferShape with format ", GetFormatName(format), "with shape", ori_size);
  return {};
}
FormatShape StorageDescHelper::GetStorageSizes(const torch_tpu::TPUStorageDesc &desc)
{
    auto ori_size = desc.base_sizes_;
    auto format = desc.tpu_format_;
    return GetStorageSizes(format, ori_size);
}
int64_t StorageDescHelper::GetMemorySize(const c10::IntArrayRef& size, tpuFormat format)
{
  const auto &physical_size = GetStorageSizes(format, size);
  if ( format == TPU_DFORMAT_CONV_W_Train) {
    int64_t size_32ic = 1, size_32oc = 1;
    for (size_t i = 0; i < 4; i ++ ) { size_32ic *= physical_size[i]; size_32oc *= physical_size[i + 4]; }
    return size_32ic + size_32oc;
  }
  else{
    return c10::multiply_integers(physical_size);
  }
}
int64_t StorageDescHelper::GetMemorySize(const TPUStorageDesc &desc)
{
  auto ori_sizes = desc.base_sizes_;
  auto format    = desc.tpu_format_; 
  return GetMemorySize(ori_sizes, format);
}
int64_t StorageDescHelper::GetMemorySize(const at::Tensor &dst)
{
  auto desc = GetTpuStorageImpl(dst)->tpu_desc_;
  return GetMemorySize(desc);
}

char *StorageDescHelper::GetFormatName(tpuFormat format)
{
    const auto& itr = info_.find(format);
    if (itr == info_.end())
    {
    AT_ERROR("unknown format type:", format);
    return nullptr;
    }
    return itr->second.formatName;
}
char *StorageDescHelper::GetFormatName(const at::Tensor &tensor)
{
    auto format = torch_tpu::GetTpuStorageImplDesc(tensor).tpu_format_;
    return GetFormatName(format);
}
tpuFormat StorageDescHelper::GetBaseFormat(tpuFormat format)
{
    const auto& itr = info_.find(format);
    if (itr == info_.end())
    {
    AT_ERROR("unknown format type:", format);
    return TPU_DFORMAT_ND;
    }
    return itr->second.baseFormat;
}
tpuFormat StorageDescHelper::GetBaseFormat(const at::Tensor &tensor)
{
    auto format = GetFormat(tensor);
    return GetBaseFormat(format);
}
tpuFormat StorageDescHelper::GetFormat(const at::Tensor &tensor)
{
    return torch_tpu::GetTpuStorageImplDesc(tensor).tpu_format_;
}

unsigned long long StorageDescHelper::GetDataPtrWithFormat(const at::Tensor& self)
{
    // TODO : to take backward grad into consideration
    return (unsigned long long) (self.data_ptr());
}


// ------------------------------------ Set
torch_tpu::TPUStorageDesc StorageDescHelper::SetDesc(const caffe2::TypeMeta &dtype)
{
  return SetDesc(dtype, {0}, {});
}
torch_tpu::TPUStorageDesc StorageDescHelper::SetDesc(const caffe2::TypeMeta &dtype, const c10::IntArrayRef& size,
                                                      const c10::IntArrayRef& strides)
{
  return SetDesc(dtype, size, strides, TPU_DFORMAT_ND);
}
torch_tpu::TPUStorageDesc StorageDescHelper::SetDesc(const caffe2::TypeMeta &dtype, const c10::IntArrayRef& size,
                                                      const c10::IntArrayRef& strides, tpuFormat tpu_format)
{
  struct torch_tpu::TPUStorageDesc tpu_desc;
  tpu_desc.data_type_ = dtype;
  tpu_desc.base_sizes_ = size;
  tpu_desc.base_strides_ = strides;
  tpu_desc.storage_sizes_ = GetStorageSizes(tpu_format, size);
  tpu_desc.origin_format_ = TPU_DFORMAT_ND;
  tpu_desc.tpu_format_ = tpu_format;
  return tpu_desc;
}

void StorageDescHelper::SetDesc(at::Tensor &dst)
{
  torch_tpu::GetTpuStorageImpl(dst)->tpu_desc_ = SetDesc(dst.dtype());
}

void StorageDescHelper::SetDesc(at::Tensor &dst, const c10::IntArrayRef& size, const c10::IntArrayRef& strides)
{
  torch_tpu::GetTpuStorageImpl(dst)->tpu_desc_ = SetDesc(dst.dtype(), size, strides);
}

void StorageDescHelper::SetDesc(at::Tensor &dst, const c10::IntArrayRef& size, const c10::IntArrayRef& strides, tpuFormat format)
{
  torch_tpu::GetTpuStorageImpl(dst)->tpu_desc_ = SetDesc(dst.dtype(), size, strides, format);
}

void StorageDescHelper::SetSgTensorAttributeWithFormat(const at::Tensor& self, SgdnnTensor_t& t)
{
    if (GetFormat(self) == TPU_DFORMAT_CONV_W_Infer){
        t.dim = self.dim();
        FormatShape f_shape = InferShapeConv_W_32IC(self.sizes());
        for (int i = 0; i < t.dim; i++) { t.shape[i] = f_shape[i]; }
        sgdnnContiguousStride(t.shape, 4, t.stride);
        t.dtype = tpu::TPUConvertDtype<decltype(t.dtype)> ( self.dtype() );
        t.format_casted = SGDNN_CONV_W_INFER_FORMAT;
    }
    else if (GetFormat(self) == TPU_DFORMAT_CONV_W_Train){
        FormatShape f_shape = InferShapeConv_W_32IC32OC(self.sizes());
        t.dim = self.dim() * 2; // double dim half for 32IC, half for 32OC
        for (int i = 0; i < t.dim; i++ ) { t.shape[i] = f_shape[i]; } // save 32ic shape and 32oc shape
        sgdnnContiguousStride(t.shape, 4, t.stride); // 32ic 's stride
        sgdnnContiguousStride(t.shape + 4, 4, t.stride + 4); // 32oc 's stride
        t.dtype = tpu::TPUConvertDtype<decltype(t.dtype)>( self.dtype() );
        t.format_casted = SGDNN_CONV_W_TRAIN_FORMAT;
    }
    else if (GetFormat(self) == TPU_DFORMAT_CONV_DW){
      t.dim = self.dim();
      FormatShape f_shape = InferShapeConv_DW_32OC(self.sizes());
      for (int i = 0; i < t.dim; i++) {t.shape[i] = f_shape[i];}
      sgdnnContiguousStride(t.shape, 4, t.stride);
      t.dtype = tpu::TPUConvertDtype<decltype(t.dtype)> ( self.dtype() );
      t.format_casted = SGDNN_CONV_DW_TRAIN_FORMAT;
    }
    else {
        AT_ERROR("unsupport InferShape with format ", GetFormatName(self));
        AT_ASSERT(false);
    }
}

void tpudnnContiguousStride ( const int * shape, int dim,  int * stride )
{
  int s = 1;
  for ( int i = dim - 1; i >= 0; --i )
  {
    stride[i] = s;
    s *= shape[i];
  }
}

void StorageDescHelper::SettpuTensorAttributeWithFormat(const at::Tensor& self, tpudnnTensor_t& t)
{
    if (GetFormat(self) == TPU_DFORMAT_CONV_W_Infer){
        t.dim = self.dim();
        FormatShape f_shape = InferShapeConv_W_32IC(self.sizes());
        for (int i = 0; i < t.dim; i++) { t.shape[i] = f_shape[i]; }
        tpudnnContiguousStride(t.shape, 4, t.stride);
        t.dtype = tpu::TPUConvertDtype<decltype(t.dtype)> ( self.dtype() );
        t.format_casted = TPUDNN_CONV_W_INFER_FORMAT;
    }
    else if (GetFormat(self) == TPU_DFORMAT_CONV_W_Train){
        FormatShape f_shape = InferShapeConv_W_32IC32OC(self.sizes());
        t.dim = self.dim() * 2; // double dim half for 32IC, half for 32OC
        for (int i = 0; i < t.dim; i++ ) { t.shape[i] = f_shape[i]; } // save 32ic shape and 32oc shape
        tpudnnContiguousStride(t.shape, 4, t.stride); // 32ic 's stride
        tpudnnContiguousStride(t.shape + 4, 4, t.stride + 4); // 32oc 's stride
        t.dtype = tpu::TPUConvertDtype<decltype(t.dtype)>( self.dtype() );
        t.format_casted = TPUDNN_CONV_W_TRAIN_FORMAT;
    }
    else if (GetFormat(self) == TPU_DFORMAT_CONV_DW){
      t.dim = self.dim();
      FormatShape f_shape = InferShapeConv_DW_32OC(self.sizes());
      for (int i = 0; i < t.dim; i++) {t.shape[i] = f_shape[i];}
      tpudnnContiguousStride(t.shape, 4, t.stride);
      t.dtype = tpu::TPUConvertDtype<decltype(t.dtype)> ( self.dtype() );
      t.format_casted = TPUDNN_CONV_DW_TRAIN_FORMAT;
    }
    else {
        AT_ERROR("unsupport InferShape with format ", GetFormatName(self));
        AT_ASSERT(false);
    }
}

// ------------------------------------ Check
bool StorageDescHelper::IsBaseFormatType(tpuFormat format)
{
    return GetBaseFormat(format) == format;
}

bool StorageDescHelper::IsBaseFormatType(const at::Tensor &tensor)
{
    if (tensor.device().type() == DeviceType::CPU) return true;
    auto format = torch_tpu::GetTpuStorageImplDesc(tensor).tpu_format_;
    return IsBaseFormatType(format);
}

bool StorageDescHelper::CheckDescInit(const c10::Storage &storage)
{
    // TOOD
    return true;
}

// ------------------------------------ Cast
at::Tensor& StorageDescHelper::unsafe_format_cast(at::Tensor& self, int64_t self_format, int64_t result_format) {
    torch_tpu::TPUStorageDesc &self_desc = torch_tpu::GetTpuStorageImpl(self)->tpu_desc_;
    if (self_format == TPU_DFORMAT_ND && result_format == TPU_DFORMAT_CONV_W_Infer) {   // normal -> 32ic w, infer mode
    self_desc.storage_sizes_ = InferShapeConv_W_32IC(self.sizes());
    self_desc.tpu_format_ = TPU_DFORMAT_CONV_W_Infer;
    } else if (self_format == TPU_DFORMAT_CONV_W_Infer && result_format == TPU_DFORMAT_ND) { 
    self_desc.storage_sizes_ = self_desc.base_sizes_;
    self_desc.tpu_format_ = TPU_DFORMAT_ND;
    } 
    else if (self_format == TPU_DFORMAT_ND && result_format == TPU_DFORMAT_CONV_W_Train) { // normal -> 32ic32oc
    self_desc.storage_sizes_ = InferShapeConv_W_32IC32OC(self.sizes());
    self_desc.tpu_format_ = TPU_DFORMAT_CONV_W_Train;
    } else if (self_format == TPU_DFORMAT_CONV_W_Train && result_format == TPU_DFORMAT_ND) {
    self_desc.storage_sizes_ = self_desc.base_sizes_;
    self_desc.tpu_format_ = TPU_DFORMAT_ND;
    }
    else if (self_format == TPU_DFORMAT_ND && result_format == TPU_DFORMAT_CONV_DW) {   // normal ->32oc
    self_desc.storage_sizes_ = InferShapeConv_DW_32OC(self.sizes());
    self_desc.tpu_format_ = TPU_DFORMAT_CONV_DW;
    } else if (self_format == TPU_DFORMAT_CONV_DW && result_format == TPU_DFORMAT_ND) {
    self_desc.storage_sizes_ = self_desc.base_sizes_;
    self_desc.tpu_format_ = TPU_DFORMAT_ND;
    }
    return self;
}

/***************************************************************************
 **********    Helper functions before FormatCast                  *********
***************************************************************************/
namespace FormatCastPreparation{
    at::Tensor apply_tensor_with_format(const at::Tensor &src, int64_t format, bool keep_format)
    {
        return apply_tensor_with_format(src, src.sizes(), format, keep_format);
    }

    at::Tensor apply_tensor_with_format(const at::Tensor &src, c10::IntArrayRef sizes, int64_t format,
                                                        bool keep_format)
    {
        return apply_tensor_with_format(sizes, src.options(), format, keep_format);
    }

    at::Tensor apply_tensor_with_format(c10::IntArrayRef sizes, const c10::TensorOptions &options,
                                                        int64_t format, bool keep_format)
    {
        TORCH_CHECK(options.device().type() == c10::DeviceType::PrivateUse1,
            "Expected all tensors to be on the same device. "
            "Expected TPU tensor, please check whether the input tensor device is correct.");
        return at_tpu::TPUNativeFunctions::unsafe_empty_with_format(
            sizes, optTypeMetaToScalarType(options.dtype_opt()), options.layout_opt(),
            options.device_opt(), options.pinned_memory_opt(), format, keep_format);
    }
} //namespace FormatCastPreparation
} //namespace at_tpu
