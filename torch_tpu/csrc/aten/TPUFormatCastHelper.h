#ifndef __TPU_FORMATCAST
#define __TPU_FORMATCAST

#include <ATen/ATen.h>
#include <unordered_map>

#include "torch_tpu/csrc/core/TPUStorageImpl.h"
#include <sgdnn_api.h>
using namespace torch_tpu;

namespace at_tpu
{
constexpr int MAX_FORMAT_SHAPE_SIZE = 8;
using FormatShape = c10::SmallVector<int64_t, MAX_FORMAT_SHAPE_SIZE>;
using baseFormatConverter = std::function<FormatShape(c10::IntArrayRef storage_dims, c10::IntArrayRef base_dims)>;

/**************************************************************
*****************   TPU Storage Desc Helper   *****************
***************************************************************/
class StorageDescHelper {
public:
  // Get Part
  // calculate storage size need by tpu memory
  static int64_t GetMemorySize(const at::Tensor &dst);
  static int64_t GetMemorySize(const TPUStorageDesc &desc);
  static int64_t GetMemorySize(const c10::IntArrayRef& size, tpuFormat format);
  template <typename sizeType>
  static FormatShape GetStorageSizes(tpuFormat format, sizeType ori_size);
  static FormatShape GetStorageSizes(const TPUStorageDesc &desc);
  // Get Attribute
  static char *GetFormatName(const at::Tensor &tensor);
  static char *GetFormatName(tpuFormat format);
  static tpuFormat GetBaseFormat(const at::Tensor &tensor);
  static tpuFormat GetBaseFormat(tpuFormat format);
  static tpuFormat GetFormat(const at::Tensor &tensor);
  static unsigned long long GetDataPtrWithFormat(const at::Tensor& self);

  // Set Part
  // StorageDesc Init/Set
  static void SetDesc(at::Tensor &dst);
  static void SetDesc(at::Tensor &dst, const c10::IntArrayRef& size, const c10::IntArrayRef& strides);
  static void SetDesc(at::Tensor &dst, const c10::IntArrayRef& size, const c10::IntArrayRef& strides, tpuFormat format);
  static TPUStorageDesc SetDesc(const caffe2::TypeMeta &dtype);
  static TPUStorageDesc SetDesc(const caffe2::TypeMeta &dtype, const c10::IntArrayRef& size,
                                            const c10::IntArrayRef& strides);
  static TPUStorageDesc SetDesc(const caffe2::TypeMeta &dtype, const c10::IntArrayRef& size,
                                            const c10::IntArrayRef& strides, tpuFormat tpu_format);
  static void SetSgTensorAttributeWithFormat(const at::Tensor& self, SgdnnTensor_t& t);

  // Check Part
  static bool IsBaseFormatType(tpuFormat format);
  static bool IsBaseFormatType(const at::Tensor &tensor);
  static bool CheckDescInit(const c10::Storage &storage);

  // Cast Part
  static at::Tensor& unsafe_format_cast(at::Tensor& self, int64_t self_format, int64_t result_format);

private:
  using shapeInfer = std::function<FormatShape(c10::IntArrayRef dims)>;
  typedef struct FormatInfo_
  {
    tpuFormat format = TPU_DFORMAT_ND;
    tpuFormat baseFormat = TPU_DFORMAT_ND;
    shapeInfer func = nullptr;
    char formatName[30] = {0};
  } FormatInfo;
  static std::unordered_map<tpuFormat, FormatInfo> info_;
}; // struct StorageDescHelper

/***************************************************************************
 **********    Helper functions before FormatCast                  *********
***************************************************************************/
namespace FormatCastPreparation {
  at::Tensor apply_tensor_with_format(const at::Tensor &src, int64_t format, bool keep_format = false);
  at::Tensor apply_tensor_with_format(const at::Tensor &src, c10::IntArrayRef sizes, int64_t format,
                                             bool keep_format = false);
  at::Tensor apply_tensor_with_format(c10::IntArrayRef sizes, const c10::TensorOptions &options, int64_t format,
                                             bool keep_format = false);
};  // namespace OpPreparation
}  // namespace:at_tpu
#endif