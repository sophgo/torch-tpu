import sys
import types
from functools import wraps

import torch
import torch_tpu._C
import torch_tpu.tpu

from torch_tpu.utils import (add_torch_funcs, add_tensor_methods, apply_module_patch)

# tpu_TENSOR = set([
#     "FloatTensor", "IntTensor", "DoubleTensor",
#     "LongTensor", "ShortTensor", "CharTensor", "ByteTensor", "HalfTensor"])

# def _isinstance(obj, class_or_tuple):
#     try:
#         return builtin_isinstance(obj, class_or_tuple)
#     except TypeError as e:
#         class_tuple = (class_or_tuple, ) if type(class_or_tuple) != tuple else class_or_tuple
#         if torch._C.device in class_tuple or torch_tpu._C.device in class_tuple:
#             return builtin_isinstance(obj, class_tuple + (torch._C.device, torch_tpu._C.device))
#         raise e

# builtins.isinstance = _isinstance


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


for name in dir(torch_tpu._C._VariableFunctions):
    if name.startswith('__'):
        continue
    globals()[name] = getattr(torch_tpu._C._VariableFunctions, name)
    __all__.append(name)
    if (name in tpu_functions) or (name.find("tpu") != -1):
        setattr(torch, name, wrap_torch_error_func(getattr(torch_tpu._C._VariableFunctions, name)))
    else:
        setattr(torch, name, getattr(torch_tpu._C._VariableFunctions, name))

all_monkey_patches = [
    ["tpu", torch_tpu.tpu],
    ["tpu.amp", torch_tpu.tpu.amp],
    #["autograd.profiler", torch_tpu.tpu.profiler],
    #["distributed", torch_tpu.distributed],
    # ["nn.parallel.distributed._get_device_index", torch_tpu.tpu._get_device_index],
    # ["distributed.distributed_c10d", torch_tpu.distributed.distributed_c10d],
    # ["nn.parallel.distributed._get_default_group", torch_tpu.distributed.distributed_c10d._get_default_group],
    # ["nn.functional", tpu_functional],
    # ["nn", tpu_modules],
    ["device", torch_tpu._C.device],
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


def apply_class_patches():
    add_torch_funcs()
    add_tensor_methods()
    apply_module_patch()

# Apply monkey-patches.
_apply_patches(all_monkey_patches)
apply_class_patches()
# torch_tpu._C._initExtension()