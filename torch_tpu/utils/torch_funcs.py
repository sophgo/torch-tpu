import torch
import torch_tpu
import warnings
from torch_tpu.utils.device_guard import torch_device_guard

@torch_device_guard
def _generator(*args, **kwargs):
  if 'privateuseone' in str(args) or 'privateuseone' in str(kwargs):
      warnings.warn(f"using torch_tpu._C.Generator for tpu device.")
      return torch_tpu._C.Generator(*args, **kwargs)
  return torch._C.Generator(*args, **kwargs)

@torch_device_guard
def _randint(*args, **kwargs):
    return torch_tpu.randint(*args, **kwargs)

@torch_device_guard
def _randint_like(*args, **kwargs):
    return torch_tpu.randint_like(*args, **kwargs)

@torch_device_guard
def _rand(*args, **kwargs):
    return torch_tpu.rand(*args, **kwargs)

@torch_device_guard
def _rand_like(*args, **kwargs):
    return torch_tpu.rand_like(*args, **kwargs)

@torch_device_guard
def _randn(*args, **kwargs):
    return torch_tpu.randn(*args, **kwargs)

@torch_device_guard
def _randn_like(*args, **kwargs):
    return torch_tpu.randn_like(*args, **kwargs)

@torch_device_guard
def _randperm(*args, **kwargs):
    return torch_tpu.randperm(*args, **kwargs)

def add_torch_funcs():
    torch.Generator = _generator
    torch.rand = _rand
    torch.rand_like = _rand_like
    torch.randint = _randint
    torch.randint_like = _randint_like
    torch.randn = _randn
    torch.randn_like = _randn_like
    torch.randperm = _randperm