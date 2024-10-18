#pragma once

#include <ATen/Tensor.h>
#include <c10/core/TensorImpl.h>
#include "torch_tpu/csrc/core/TPUStorageImpl.h"

namespace torch_tpu {

// TPUTensorImpl class is derived from c10::TensorImpl, and it is only used to handle an TPU tensor.
// Its scope is just to handle an TPUTensor.
class TPUTensorImpl : public c10::TensorImpl {
public:
  explicit TPUTensorImpl(c10::Storage&& storage, const caffe2::TypeMeta& data_type);

  void shallow_copy_from(const c10::intrusive_ptr<TensorImpl>& impl) final;

  c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(
      const c10::VariableVersion& version_counter,
      bool allow_tensor_metadata_change) const final;
  /**
   * Return a TensorImpl that is a shallow-copy of this TensorImpl.
   *
   * For usage of `version_counter` and `allow_tensor_metadata_change`,
   * see NOTE [ TensorImpl Shallow-Copying ].
   */
  c10::intrusive_ptr<c10::TensorImpl> shallow_copy_and_detach(
      c10::VariableVersion&& version_counter,
      bool allow_tensor_metadata_change) const final;

public:
  TPUTensorImpl(const TPUTensorImpl&) = delete;
  TPUTensorImpl& operator=(const TPUTensorImpl&) = delete;
  TPUTensorImpl(TPUTensorImpl&&) = default;
  TPUTensorImpl& operator=(TPUTensorImpl&&) = default;
  ~TPUTensorImpl();
};

TPUTensorImpl* GetTpuTensorImpl(const at::Tensor &tensor);
}  // namespace torch_tpu
