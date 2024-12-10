from ..utils import environ_flag
if not environ_flag('DISABLE_ADAPTOR', default_override="0") and environ_flag('ENABLE_TORCH_ADAPTOR'):
    from . import torch_adaptor
    
try:
    import deepspeed
    if not environ_flag('DISABLE_ADAPTOR', default_override="0") and environ_flag('ENABLE_DEEPSPEED_ADAPTOR'):
        from . import deepspeed_adaptor
        deepspeed_adaptor.accelerator_tpu_accelerator.set_tpu_accelerator()
except ImportError:
    print("Trying to import deepspeed adaptor, but deepspeed is not found")
    
try:
    import megatron
    try:
        from megatron import get_args # megatron-deepspeed
        if not environ_flag('DISABLE_ADAPTOR', default_override="0") and environ_flag('ENABLE_MEGATRON_DEEPSPEED_ADAPTOR'):
            from . import megatron_deepspeed_adaptor
    except ImportError: # megatron-lm
        if not environ_flag('DISABLE_ADAPTOR', default_override="0") and environ_flag('ENABLE_MEGATRON_LM_ADAPTOR'):
            from . import megatron_lm_adaptor
finally:
    pass
# except ImportError:
#     print("Trying to import megatron adaptor, but megatron is not found")