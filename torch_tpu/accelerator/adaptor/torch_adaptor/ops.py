from functools import wraps
import torch
import torch_tpu

def fix_int64(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        dtype = kwargs.get('dtype')
        if dtype is not None and not 'cpu' in str(kwargs.get('device')):
            if isinstance(dtype, str):
                dtype = getattr(torch, dtype, dtype)
            if dtype == torch.int64:
                kwargs['dtype'] = torch.int32
        return func(*args, **kwargs)
    return wrapper

torch.empty = fix_int64(torch.empty)
torch.arange = fix_int64(torch.arange)
torch.ones = fix_int64(torch.ones)
torch.zeros = fix_int64(torch.zeros)