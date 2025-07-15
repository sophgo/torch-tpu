#pragma once
// this file can generate by autogen by yaml file, todo.

#include <ATen/Tensor.h>
#include <ATen/ATen.h>

namespace at_tpu {
namespace TPUNativeFunctions {
/***************************************************************************
 **********              empty Tensor functions                    *********
***************************************************************************/
at::Tensor empty_with_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype={}, c10::optional<at::Layout> layout={}, c10::optional<at::Device> device={}, c10::optional<bool> pin_memory={}, int64_t tpu_format=0);
at::Tensor unsafe_empty_with_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype={}, c10::optional<at::Layout> layout={}, c10::optional<at::Device> device={}, c10::optional<bool> pin_memory={}, int64_t acl_format=2, bool keep_format=false);
at::Tensor make_tensor_from_ptr(void *ptr, std::vector<int64_t>& sizes, at::ScalarType dtype);
/***************************************************************************
 **********              format cast functions                     *********
***************************************************************************/
int64_t get_tpu_format(const at::Tensor & self);
// inplace is necessary?
//at::Tensor& tpu_format_cast(at::Tensor& self, int64_t tpu_format); // format cast inplace
at::Tensor tpu_format_cast(const at::Tensor& self, int64_t tpu_format);  // format cast generate new tensor
at::Tensor tpu_format_cast(const at::Tensor& self, const at::Tensor& dst);  // format cast self to a new tensor, with dst's format
at::Tensor tpu_format_cast_back_to_origin(const at::Tensor& self); // cast tpu_format to base_format, generate new tensor
} // namespace TPUNativeFunctions
}// namespace at_tpu