import sys
import types
from functools import wraps

import torch
import torch_tpu._C
import torch_tpu.tpu

from torch_tpu.utils import ( apply_module_patch, \
                             add_storage_methods, add_serialization_methods, apply_device_patch)

__all__ = []

def wrap_torch_error_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        raise RuntimeError(f"torch.{func.__name__} is deprecated and will be removed in future version. "
                           f"Use torch_tpu.{func.__name__} instead.")
    return wrapper

tpu_functions = {
    # "one_", "fast_gelu", "_amp_foreach_non_finite_check_", "empty_with_format", "unsafe_empty_with_format", 
    # "empty_with_format", "copy_memory_", "_dropout_with_byte_mask_backward", "dropout_with_byte_mask", 
    # "decode_jpeg", "crop_and_resize", "reverse", "image_normalize", "image_normalize_", "img_to_tensor", 
    # "_conv_depthwise2d_backward", "slow_conv_dilated2d_backward", "slow_conv_transpose2d_backward", 
    # "batch_norm_reduce", "batch_norm_gather_stats_update", "format_contiguous", "check_match", 
    # "check_memory_overlaps", "get_storage_size", "_dropout_with_byte_mask", "empty_with_format"
}


for name in dir(torch.ops.tpu):
    if name.startswith('__')  or name in ['_dir', 'name']:
        continue
    globals()[name] = getattr(torch.ops.tpu, name)
    if (name in tpu_functions):
        __all__.append(name)
    setattr(torch, name, wrap_torch_error_func(getattr(torch.ops.tpu, name)))

all_monkey_patches = [
    # ["nn.functional", tpu_functional],
    # ["nn", tpu_modules],
]

def _apply_patches(monkey_patches):
    
    def _getattr(module_list, root_module=torch):
        if len(module_list) <= 1:
            return root_module

        if hasattr(root_module, module_list[0]):
            return _getattr(module_list[1:], getattr(root_module, module_list[0]))
        else:
            empty_module_name = f'{root_module.__name__}.{module_list[0]}'
            sys.modules[empty_module_name] = types.ModuleType(empty_module_name)
            setattr(root_module, module_list[0], sys.modules.get(empty_module_name))
            return _getattr(module_list[1:], getattr(root_module, module_list[0]))

    for patch_pair in monkey_patches:
        dest, patch = patch_pair
        dest_module = _getattr(dest.split('.'), root_module=torch)
        last_module_level = dest.split(".")[-1]
        if not isinstance(patch, types.ModuleType):
            setattr(dest_module, last_module_level, patch)
            continue

        if not hasattr(dest_module, last_module_level) or not hasattr(patch, '__all__'):
            setattr(dest_module, last_module_level, patch)
            sys.modules[f'{dest_module.__name__}.{last_module_level}'] = patch
            continue

        if not hasattr(patch, '__all__'):
            raise NotImplementedError("Patch module must have __all__ definition.")
        dest_module = getattr(dest_module, last_module_level)
        for attr in patch.__all__:
            setattr(dest_module, attr, getattr(patch, attr))

# def _apply_distributed_patches():
#     torch.nn.parallel.DistributedDataParallel._ddp_init_helper = torch_npu.utils.module._ddp_init_helper
#     _apply_patches([["distributed", torch_npu.distributed]])

def apply_class_patches():
    add_storage_methods()
    add_serialization_methods()
    apply_device_patch()
    apply_module_patch()

torch.utils.rename_privateuse1_backend("tpu")
torch._register_device_module('tpu', torch_tpu.tpu)
unsupported_dtype = [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8, torch.int64]
torch.utils.generate_methods_for_privateuse1_backend(for_tensor=True, for_module=True, for_storage=True,
                                                     unsupported_dtype=unsupported_dtype)

# Apply monkey-patches.
_apply_patches(all_monkey_patches)
apply_class_patches()

# #### distributed
# # init and register hccl backend
# torch.distributed.is_sccl_available = lambda : True
# torch.distributed.Backend.register_backend("sccl", lambda store, group_rank, group_size, timeout:
#     torch_tpu._C._distributed_c10d.ProcessGroupSCCL(store, group_rank, group_size, timeout), devices=["tpu"])