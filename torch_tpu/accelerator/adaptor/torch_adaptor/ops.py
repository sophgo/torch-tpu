from functools import wraps
import torch
import torch_tpu

def fix_int64(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        dtype = kwargs.get('dtype')
        if dtype is not None:
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype, dtype)
            if dtype == torch.int64:
                kwargs['dtype'] = torch.int32
        return func(*args, **kwargs)
    return wrapper

torch.empty = fix_int64(torch.empty)
torch.arange = fix_int64(torch.arange)
