import torch
import torch_tpu
import warnings
#from torch_tpu.utils.device_guard import torch_device_guard

def _generator(*args, **kwargs):
  if 'privateuseone' in str(args) or 'privateuseone' in str(kwargs):
      warnings.warn(f"using torch_tpu._C.Generator for tpu device.")
      return torch_tpu._C.Generator(*args, **kwargs)
  return torch._C.Generator(*args, **kwargs)

def add_torch_funcs():
    torch.Generator = _generator