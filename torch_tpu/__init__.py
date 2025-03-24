import sys
import types
from functools import wraps
import importlib
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

arch = arch_env = os.environ.get('CHIP_ARCH')

def make_symlinks(arch_sel):
    tpudnn = f'libtpudnn.{arch_sel}.so'
    sofn = f'libtorch_tpu.{arch_sel}.so'
    symlink(sofn, os.path.join(lib_pwd, 'libtorch_tpu.so'))
    symlink(tpudnn, os.path.join(lib_pwd, 'libtpudnn.so'))

if arch_env:
    # Env overrides everything
    make_symlinks(arch_env)

if not arch:
    # Try existing symlinks
    import re
    try:
        tpudnn = os.readlink(os.path.join(lib_pwd, 'libtpudnn.so'))
        m = re.match(r'^libtpudnn.(\w+).so$', tpudnn)
        if not m:
            re.match(r'.+TPU1686.+build_(\w+).+', tpudnn)
        arch = m.group(1) if m else 'sg2260' # default to sg2260
    except FileNotFoundError:
        pass

if not arch:
    # Select arch by checking if coresponding tpuDNN works
    from ctypes import cdll
    arch_list = ['sg2260', 'bm1684x']
    for arch_iter in arch_list:
        tpudnn = f'libtpudnn.{arch_iter}.so'
        try:
            cdll.LoadLibrary(os.path.join(lib_pwd, tpudnn))
        except:
            continue
        arch = arch_iter
        make_symlinks(arch_iter)
        break

if not os.environ.get('TPU_EMULATOR_PATH'):
    os.environ['TPU_EMULATOR_PATH'] = os.path.join(lib_pwd, f'{arch}_cmodel_firmware.so')

# OPEN INS-CACHE default
if os.environ.get('TPU_CACHE_BACKEND') is None:
    if not os.environ.get('DISABLE_CACHE'):
        os.environ['TPU_CACHE_BACKEND'] = os.environ['TPU_EMULATOR_PATH']

def modify_backend_py(arch:str):
    backend_f = os.path.join(pkg_path, "tpu/backend.py")
    with open(backend_f, 'w') as f:
        if arch == 'bm1684x':
            f.write("BACKEND='1684X'")
        elif arch == 'sg2260':
            f.write("BACKEND='SG2260'")
        else:
            raise RuntimeError("Failed to modify backend.py")
modify_backend_py(arch)

if arch == 'sg2260':
    try:
        _C_impl = importlib.import_module('torch_tpu.sg2260._C')
    except ImportError:
        raise ImportError("Failed to import BM1690 implement. "
                          "Please check if the torch-tpu is built with BM1690 backend.")
elif arch == 'bm1684x':
    try:
        _C_impl = importlib.import_module('torch_tpu.bm1684x._C')
    except ImportError:
        raise ImportError("Failed to import BM1684X implement. "
                          "Please check if the torch-tpu is built with BM1684X backend.")
else:
    raise RuntimeError("unsupported arch: {arch}")
sys.modules['torch_tpu._C'] = _C_impl
_C = _C_impl
import torch_tpu._C
import torch_tpu.tpu
from .tpu.jit import (jit, CallCppDynLib)
from torch_tpu.utils import ( apply_module_patch, \
                             add_storage_methods, add_serialization_methods, apply_device_patch)
if arch == 'sg2260':
    from torch_tpu._C import ProcessGroupSCCLOptions
    os.environ['CHIP'] = "bm1690"
elif arch == 'bm1684x':
    os.environ['CHIP'] = "bm1684x"

__all__ = []

def wrap_torch_error_func(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        raise RuntimeError(f"torch.{func.__name__} is deprecated and will be removed in future version. "
                           f"Use torch_tpu.{func.__name__} instead.")
    return wrapper


###### CUSTOM OPs. no use now
###### TODO: change namespace. torch_tpu:tpu/my_ops -> torch
# tpu_functions = {
#     "tpu_format_cast",
# }
# for name in dir(torch.ops.tpu):
#     if name.startswith('__')  or name in ['_dir', 'name']:
#         continue
#     globals()[name] = getattr(torch.ops.tpu, name)
#     if (name in tpu_functions):
#         __all__.append(name)
#     setattr(torch, name, wrap_torch_error_func(getattr(torch.ops.tpu, name)))

##### register device 
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
