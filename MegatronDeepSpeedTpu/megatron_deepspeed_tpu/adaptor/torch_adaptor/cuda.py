from functools import wraps
import torch
import torch_tpu

torch.cuda = torch.tpu

def get_tpu_device(device=None):
    if device is not None:
        if str(device) == 'cuda':
            device = f"tpu:{torch.tpu.current_device()}"
        elif str(device).startswith("cuda"):
            device = str(device).replace("cuda", "tpu")
        elif isinstance(device, int):
            device = f"tpu:{device}"
        if str(device) != "cpu":
            assert str(device) == f"tpu:{torch.tpu.current_device()}", f"Device {device} is not the current TPU device {torch.tpu.current_device()}"
    return device

def tpu(self, device=None, non_blocking=False):
    if device is None:
        device = f"tpu:{torch.tpu.current_device()}"
    return self.to(device=get_tpu_device(device), non_blocking=non_blocking)

torch.Tensor.cuda = tpu

def pin_memory(self, device=None):
    if device is None:
        device = self.device
    return self.to(device=get_tpu_device(device))

torch.Tensor.pin_memory = pin_memory

def fix_device(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if 'device' in kwargs:
            dvc = get_tpu_device(kwargs['device'])
            kwargs['device'] = dvc
        return func(*args, **kwargs)
    return wrapper

torch.tensor = fix_device(torch.tensor)
torch.empty = fix_device(torch.empty)
torch.zeros = fix_device(torch.zeros)
torch.ones = fix_device(torch.ones)
torch.full = fix_device(torch.full)
torch.rand = fix_device(torch.rand)
torch.randn = fix_device(torch.randn)
torch.randint = fix_device(torch.randint)
torch.eye = fix_device(torch.eye)
torch.arange = fix_device(torch.arange)
torch.linspace = fix_device(torch.linspace)
torch.logspace = fix_device(torch.logspace)
torch.meshgrid = fix_device(torch.meshgrid)

class BFloat16Tensor(torch.Tensor):
    def __new__(cls, *args, **kwargs):
        return torch.tensor(*args, dtype=torch.bfloat16, device=f"tpu:{torch.tpu.current_device()}", **kwargs)
    
class ByteTensor(torch.Tensor):
    def __new__(cls, *args, **kwargs):
        return torch.tensor(*args, dtype=torch.uint8, device=f"tpu:{torch.tpu.current_device()}", **kwargs)
    
class DoubleTensor(torch.Tensor):
    def __new__(cls, *args, **kwargs):
        return torch.tensor(*args, dtype=torch.float, device=f"tpu:{torch.tpu.current_device()}", **kwargs)
    
class FloatTensor(torch.Tensor):
    def __new__(cls, *args, **kwargs):
        return torch.tensor(*args, dtype=torch.float, device=f"tpu:{torch.tpu.current_device()}", **kwargs)
    
class HalfTensor(torch.Tensor):
    def __new__(cls, *args, **kwargs):
        return torch.tensor(*args, dtype=torch.half, device=f"tpu:{torch.tpu.current_device()}", **kwargs)
    
class IntTensor(torch.Tensor):
    def __new__(cls, *args, **kwargs):
        return torch.tensor(*args, dtype=torch.int, device=f"tpu:{torch.tpu.current_device()}", **kwargs)
    
class LongTensor(torch.Tensor):
    def __new__(cls, *args, **kwargs):
        return torch.tensor(*args, dtype=torch.int, device=f"tpu:{torch.tpu.current_device()}", **kwargs)

torch.tpu.BFloat16Tensor = BFloat16Tensor
torch.tpu.ByteTensor = ByteTensor
torch.tpu.DoubleTensor = DoubleTensor
torch.tpu.FloatTensor = FloatTensor
torch.tpu.HalfTensor = HalfTensor
torch.tpu.IntTensor = IntTensor
torch.tpu.LongTensor = LongTensor
torch.tpu._is_in_bad_fork = lambda: False
