#include "TPUGuardImpl.h"
#include "TPUStorageImpl.h"

namespace c10_tpu
{
namespace impl
{
constexpr DeviceType TPUGuardImpl::static_type;
C10_REGISTER_GUARD_IMPL ( PrivateUse1, TPUGuardImpl );

#define REGISTER_PRIVATEUSE1_BACKEND(name)                                                      \
  int rename_privateuse1_backend() {                                                            \
    c10::register_privateuse1_backend(#name);                                                   \
    c10::SetStorageImplCreate(c10::DeviceType::PrivateUse1, &torch_tpu::make_tpu_storage_impl); \
    return 0;                                                                                   \
  }                                                                                             \
  static const int _temp_##name = rename_privateuse1_backend();

REGISTER_PRIVATEUSE1_BACKEND(tpu)

} // namespace impl
} // namespace c10_tpu
