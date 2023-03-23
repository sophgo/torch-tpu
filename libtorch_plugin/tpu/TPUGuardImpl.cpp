#include "TPUGuardImpl.h"

namespace c10
{
namespace tpu
{
namespace impl
{
constexpr DeviceType TPUGuardImpl::static_type;
C10_REGISTER_GUARD_IMPL ( TPU, TPUGuardImpl );
} // namespace impl
} // namespace tpu
} // namespace c10
