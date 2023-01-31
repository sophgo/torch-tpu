#include "TPUGuardImpl.h"

namespace c10
{
namespace tpu
{
namespace impl
{
constexpr DeviceType TPUGuardImpl::static_type;
C10_REGISTER_GUARD_IMPL ( PrivateUse1, TPUGuardImpl );
} // namespace impl
} // namespace tpu
} // namespace c10
