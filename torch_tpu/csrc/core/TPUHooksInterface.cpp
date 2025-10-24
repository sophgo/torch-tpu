#include <pybind11/chrono.h>
#include <torch/python.h>
#include "torch_tpu/csrc/core/TPUHooksInterface.h"


namespace c10_tpu {

TORCH_DECLARE_REGISTRY(PrivateUse1HooksRegistry, TPUHooksInterface, TPUHooksArgs);
#define REGISTER_PRIVATEUSE1_HOOKS(clsname) \
  C10_REGISTER_CLASS(PrivateUse1HooksRegistry, clsname, clsname)

C10_DEFINE_REGISTRY(PrivateUse1HooksRegistry, TPUHooksInterface, TPUHooksArgs)

void TPUHooksInterface::init() const
{
#ifndef BUILD_LIBTORCH
    torch_tpu::utils::tpu_lazy_init();
#endif
}

at::PrivateUse1HooksInterface* get_tpu_hooks()
{
    static at::PrivateUse1HooksInterface* tpu_hooks;
    static c10::once_flag once;
    c10::call_once(once, [] {
        tpu_hooks = new TPUHooksInterface();
    });
    return tpu_hooks;
}

}
