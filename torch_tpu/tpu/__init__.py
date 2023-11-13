
__all__ = [
    "native_device", "tpu_device", "amp", "set_device",
    "is_initialized", "_lazy_call", "_lazy_init", "init",
    "device_count", "current_device", "is_available"
]

from typing import Tuple

import torch
import torch_tpu

from . import amp
from .device import __device__ as native_device
from .device import __tpu_device__ as tpu_device
from .utils import ( _lazy_call, _lazy_init, init,
                    device_count, current_device, set_device,is_initialized, is_available)

default_generators: Tuple[torch._C.Generator] = ()