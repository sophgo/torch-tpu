import sys
import types
from functools import wraps

import torch

import os

def symlink(src, dst):
    try:
        link_val = os.readlink(dst)
        if link_val == src:
            return
        else:
            os.unlink(dst)
    except FileNotFoundError:
        pass

    try:
        os.symlink(src, dst)
    except FileExistsError:
        pass

import pkgutil
pkg_path = os.path.dirname(pkgutil.get_loader('torch_tpu').get_filename())
lib_pwd = os.path.join(pkg_path, 'lib/')
arch = os.environ.get('CHIP_ARCH')
if arch or not os.path.exists(os.path.join(lib_pwd, 'libtorch_tpu.so')):
    if not arch:
        from ctypes import cdll
        try:
            cdll.LoadLibrary("libtpuv7_rt.so")
            arch = 'sg2260'
        except:
            arch = 'bm1684x'

    if arch == 'sg2260':
        symlink('libtorch_tpu.sg2260.so', os.path.join(lib_pwd, 'libtorch_tpu.so'))
        tpudnn = 'libtpudnn.sg2260.so'
    else:
        symlink('libtorch_tpu.bm1684x.so', os.path.join(lib_pwd, 'libtorch_tpu.so'))
        tpudnn = 'libtpudnn.bm1684x.so'

    if os.path.exists(os.path.join(lib_pwd, tpudnn)):
        symlink(tpudnn, os.path.join(lib_pwd, 'libtpudnn.so'))

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


###### CUSTOM OPs. TODO: change tpu to my_ops
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

##### register device
torch.utils.rename_privateuse1_backend("tpu")
torch._register_device_module('tpu', torch_tpu.tpu)
unsupported_dtype = [torch.quint8, torch.quint4x2, torch.quint2x4, torch.qint32, torch.qint8, torch.int64]
torch.utils.generate_methods_for_privateuse1_backend(for_tensor=True, for_module=True, for_storage=True,
                                                     unsupported_dtype=unsupported_dtype)
##### mokey-patches
def apply_class_patches():
    add_storage_methods()
    add_serialization_methods()
    apply_device_patch()
    apply_module_patch()

apply_class_patches()

## init TPU's Extension
# torch_tpu.torch_tpu._initExtension()

# #### distributed
# # init and register hccl backend
# torch.distributed.is_sccl_available = lambda : True
# torch.distributed.Backend.register_backend("sccl", lambda store, group_rank, group_size, timeout:
#     torch_tpu._C._distributed_c10d.ProcessGroupSCCL(store, group_rank, group_size, timeout), devices=["tpu"])
