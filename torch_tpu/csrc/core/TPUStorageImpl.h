#pragma once

#include <ATen/Tensor.h>
#include <c10/core/StorageImpl.h>
#include <c10/core/Allocator.h>
#include <c10/core/ScalarType.h>
#include <c10/util/typeid.h>
#include <c10/util/order_preserving_flat_hash_map.h>

namespace torch_tpu {

/**************************************************************
*****************   TPU Storage Desc Define   *****************
***************************************************************/
typedef enum {
  TPU_DFORMAT_UNDEFINED = -1,
  TPU_DFORMAT_ND = 0,
  TPU_DFORMAT_CONV_W_Infer = 1,
  TPU_DFORMAT_CONV_W_Train = 2,
  TPU_DFORMAT_CONV_DW      = 3, // conv2d's weight's grad
} tpuFormat;

struct TPUStorageDesc {
public:
    struct use_byte_size_t {};

    c10::SmallVector<int64_t, 8> base_sizes_;
    c10::SmallVector<int64_t, 8> base_strides_;
    c10::SmallVector<int64_t, 8> storage_sizes_;
    int64_t base_offset_ = 0; // no use
    use_byte_size_t base_dtype_ = {}; // no use
    tpuFormat origin_format_ = TPU_DFORMAT_UNDEFINED;
    tpuFormat tpu_format_ = TPU_DFORMAT_ND;
    caffe2::TypeMeta data_type_;
};

/**************************************************************
*****************      TPU Storage Impl       *****************
***************************************************************/
struct TPUStorageImpl : public c10::StorageImpl {
  explicit TPUStorageImpl(use_byte_size_t use_byte_size,
      size_t size_bytes,
      at::DataPtr data_ptr,
      at::Allocator* allocator,
      bool resizable);
  ~TPUStorageImpl() override = default;

  void release_resources() override;

  // not private
  TPUStorageDesc tpu_desc_;

  TPUStorageDesc get_tpu_desc() const {
    return tpu_desc_;
  }
};

/**************************************************************
*****************      interface function     *****************
***************************************************************/
c10::intrusive_ptr<c10::StorageImpl> make_tpu_storage_impl(
    c10::StorageImpl::use_byte_size_t,
    c10::SymInt size_bytes,
    c10::Allocator* allocator,
    bool resizable);

TPUStorageImpl* GetTpuStorageImpl(c10::StorageImpl* storageImpl);
TPUStorageImpl* GetTpuStorageImpl(c10::Storage&& storage);
TPUStorageImpl* GetTpuStorageImpl(const at::Tensor& storageImpl);
TPUStorageDesc& GetTpuStorageImplDesc(const at::Tensor &tensor);
} //namespace torch_tpu