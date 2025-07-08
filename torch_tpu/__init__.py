import os
import pkgutil

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

pkg_path = os.path.dirname(pkgutil.get_loader('torch_tpu').get_filename())
lib_pwd = os.path.join(pkg_path, 'lib/')

arch_list = ['sg2260', 'bm1684x']
arch = arch_env = os.environ.get('CHIP_ARCH')
def make_symlinks(arch_sel):
    tpudnn = f'libtpudnn.{arch_sel}.so'
    sofn   = f'libtorch_tpu.{arch_sel}.so'
    pysofn = f'libtorch_tpu_python.{arch_sel}.so'
    symlink(pysofn, os.path.join(lib_pwd, 'libtorch_tpu_python.so'))
    symlink(sofn,   os.path.join(lib_pwd, 'libtorch_tpu.so'))
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
        if arch in arch_list:
            os.environ['CHIP_ARCH'] = arch
    except FileNotFoundError:
        pass

if not arch:
    # Select arch by checking if coresponding tpuDNN works
    from ctypes import cdll
    for arch_iter in arch_list:
        tpudnn = f'libtpudnn.{arch_iter}.so'
        try:
            cdll.LoadLibrary(os.path.join(lib_pwd, tpudnn))
        except:
            print(f"ERROR: unable to load {os.path.join(lib_pwd, tpudnn)}")
            continue
        arch = arch_iter
        make_symlinks(arch_iter)
        os.environ['CHIP_ARCH'] = arch
        break

if not os.environ.get('TPU_EMULATOR_PATH'):
    os.environ['TPU_EMULATOR_PATH'] = os.path.join(lib_pwd, f'{arch}_cmodel_firmware.so')

# OPEN INS-CACHE default
if os.environ.get('TPU_CACHE_BACKEND') is None:
    if not os.environ.get('DISABLE_CACHE'):
        os.environ['TPU_CACHE_BACKEND'] = os.environ['TPU_EMULATOR_PATH']

#open kernel-module save default
os.environ['TorchTpuSaveKernelModule'] = '1'

### TORCH-CPP LOG LEVEL
if not os.environ.get('TORCH_TPU_CPP_LOG_LEVEL'):
    os.environ['TORCH_TPU_CPP_LOG_LEVEL'] = '2' # 0: INFO| 1: WARNING| 2: ERROR| 3:FATAL

import torch
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

## torch-tpu ENVS
def print_all_torch_tpu_envs():
    print(f"===================== CHIP-ARCH ===========================================")
    print(f"CHIP_ARCH                   = {os.environ.get('CHIP_ARCH')}, choose backend. value: 2260 | bm1684x | None(default). default will chose 2260.")
    print(f"CHIP                        = {os.environ.get('CHIP')}, same with`CHIP_ARCH`, to compatible with backend")
    print(f"TPU_EMULATOR_PATH           = {os.environ.get('TPU_EMULATOR_PATH')}, used for emulator version. value: the path of emulator lib.")

    print(f"===================== INST-CACHE ===========================================")
    print(f"DISABLE_CACHE              = {os.environ.get('DISABLE_CACHE')}, wheater use inst-cache. value: None | ON. default is None(use inst cache)")
    print(f"TPU_CACHE_BACKEND          = {os.environ.get('TPU_CACHE_BACKEND')}, the lib to generate tpu inst.")

    print(f"===================== ALLOCATOR-CONFIG ======================================")
    print(f"TPU_ALLOCATOR_FREE_DELAY_IN_MS = {os.environ.get('TPU_ALLOCATOR_FREE_DELAY_IN_MS')}, the time(ms) mem no use will free, default is 0.")

    print(f"===================== DYNAMO COMPILER=======================================")
    print(f"COMPILER_DTYPE             = {os.environ.get('COMPILER_DTYPE')}, compiler will use f32 dtype default, value: 0(f32): 1(f16): 2(bf16)")

    print(f"===================== BMODEL-RUNTIME =======================================")
    print(f"ModelRtRunWithTorchTpu     = {os.environ.get('ModelRtRunWithTorchTpu')}, model-rt will use torch-tpu's kernel-module(~/.torch_tpu_kernel_module)")
    print(f"TorchTpuSaveKernelModule   = {os.environ.get('TorchTpuSaveKernelModule')}, save kernel-module with ~/.torch_tpu_kernel_module. value: 1 | None(default), no save default")
    print(f"ModelRtWTorchDEBUG         = {os.environ.get('ModelRtWTorchDEBUG')}, save bmodel module's IO for debug. value: 1 | None(default), no save default")

    print(f"===================== DISTRIBUTED  =========================================")
    print(f"CHIP_MAP                   = {os.environ.get('CHIP_MAP')}, rank to physical-device.")

    print(f"===================== LOG-LEVEL ===========================================")
    print(f"TORCH_TPU_CPP_LOG_LEVEL    = {os.environ.get('TORCH_TPU_CPP_LOG_LEVEL')}, value: 0|1|2|3. default is 2(ERROR), only in debug version")
