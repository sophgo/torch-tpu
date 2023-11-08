import torch
import torch_tpu
from torch_tpu.utils.device_guard import torch_device_guard

@torch_device_guard
def _tpu(self, *args, **kwargs):
    return torch_tpu._C.tpu(self, *args, **kwargs)

@torch_device_guard
def _to(self, *args, **kwargs):
    return torch_tpu._C.to(self, *args, **kwargs)

@property
def _is_tpu(self):
    return torch_tpu._C.is_tpu(self)

def add_tensor_methods():
    torch.Tensor.tpu = _tpu
    torch.Tensor.to = _to
    torch.Tensor.is_tpu = _is_tpu