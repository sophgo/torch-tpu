import torch
import types

def int64_to_int32_and_check_overflow(tensor):
    assert tensor.dtype == torch.int64
    tensor_value = tensor.cpu()
    if torch.any(tensor_value > 2147483647) or torch.any(tensor_value < -2147483648):
        raise ValueError("int32 out of range")
    return tensor.int()

def float64_to_float32_and_check_overflow(tensor):
    assert tensor.dtype == torch.float64
    tensor_value = tensor.cpu()
    if torch.any(tensor_value > 3.4028235e+38) or torch.any(tensor_value < -3.4028235e+38):
        raise ValueError("float32 out of range")
    return tensor.float()

def apply_recursively(reduce_func=None, destroy_dict=True):
    if reduce_func is None:
        _reduce_func = lambda x: x
    else:
        _reduce_func = lambda x: reduce_func(x.values()) if destroy_dict and isinstance(x, dict) else reduce_func(x)
    def wrapper(func):
        def inner_wrapper(object, *args, **kwargs):
            if isinstance(object, (list, tuple)):
                return _reduce_func(object.__class__(inner_wrapper(t, *args, **kwargs) for t in object))
            elif isinstance(object, types.GeneratorType):
                return _reduce_func(inner_wrapper(t, *args, **kwargs) for t in object)
            elif isinstance(object, dict):
                return _reduce_func({k: inner_wrapper(v, *args, **kwargs) for k, v in object.items()})
            else:
                return func(object, *args, **kwargs)
        return inner_wrapper
    return wrapper

def to_str(object):
    if isinstance(object, (list, tuple, types.GeneratorType)):
        return "(" + ", ".join([f"{t}" for t in object]) + ")"
    elif isinstance(object, dict):
        return "{" + ", ".join([f"{k}={v}" for k, v in object.items()]) +"}"
    else:
        return str(object)

@apply_recursively(to_str, False)
def get_tensor_info(tensor):
    if isinstance(tensor, torch.Tensor):
        return f"<tensor={list(tensor.shape)}, dtype={str(tensor.dtype).replace('torch.', '')}>"
    if isinstance(tensor, torch.nn.Module):
        return f"<module: {tensor.__class__.__name__}>"
    return tensor

def to_flat_dict(d, parent_key='', sep='_'):
    items = []
    if isinstance(d, (list, tuple, types.GeneratorType)):
        for i, v in enumerate(d):
            if v is not None:
                new_key = f"{parent_key}{sep}arg{i}" if parent_key else f"arg{i}"
                items.extend(to_flat_dict(v, new_key, sep=sep).items())
    elif isinstance(d, dict):
        for k, v in d.items():
            if v is not None:
                new_key = sep.join([i for i in [parent_key, k] if i])
                items.extend(to_flat_dict(v, new_key, sep=sep).items())
    else:
        if d is not None:
            items.append((parent_key, d))
    return dict(items)

@apply_recursively(destroy_dict=False)
def tensors_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.cpu().detach().numpy()
    return None

@apply_recursively(destroy_dict=False)
def tensor_grads_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.grad.cpu().detach().numpy() if tensor.grad is not None else None
    return None

prepare_tensor_for_save = lambda x: to_flat_dict(tensors_to_numpy(x))
prepare_tensor_grad_for_save = lambda x: to_flat_dict(tensor_grads_to_numpy(x))


@apply_recursively(any)
def has_64bit_tensor(tensor):
    return isinstance(tensor, torch.Tensor) and tensor.dtype in [torch.int64, torch.float64]

@apply_recursively(any)
def has_16bit_fp_tensor(tensor):
    return isinstance(tensor, torch.Tensor) and tensor.dtype in [torch.half, torch.bfloat16]

@apply_recursively(any)
def has_tpu_tensor(tensor):
    return isinstance(tensor, torch.Tensor) and str(tensor.device).startswith("tpu")

@apply_recursively(any)
def has_nan_inf_tensor(tensor):
    if isinstance(tensor, torch.Tensor):
        tensor_cpu = tensor.cpu()
        return torch.any(torch.isnan(tensor_cpu)) or torch.any(torch.isinf(tensor_cpu))
    return False

@apply_recursively()
def tensor_64bit_to_32bit(tensor, *args):
    if isinstance(tensor, torch.Tensor):
        if tensor.dtype == torch.int64:
            return int64_to_int32_and_check_overflow(tensor)
        elif tensor.dtype == torch.float64:
            return float64_to_float32_and_check_overflow(tensor)
    return tensor

@apply_recursively()
def tensor_tpu_to_cpu(tensor, *args):
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if str(tensor.device).startswith("tpu"):
        if tensor.dtype in [torch.half, torch.bfloat16]:
            return tensor.cpu().float()
        elif tensor.dtype == torch.int:
            return tensor.cpu().to(torch.int64)
        return tensor.cpu()
    else:
        raise ValueError(f"tensor is on {str(tensor.device)}, not on TPU")
    
@apply_recursively()
def tensor_cpu_to_tpu(tensor, device, dtype, *args):
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if str(device).startswith("tpu"):
        if tensor.dtype == torch.float and dtype in [torch.half, torch.bfloat16]:
            return tensor.to(dtype).to(device)
        elif tensor.dtype == torch.int64:
            return int64_to_int32_and_check_overflow(tensor).to(device)
        return tensor.to(device)
    else:
        raise ValueError("device is not TPU")

@apply_recursively()
def tensor_fp16_to_fp32(tensor, *args):
    return tensor.float() if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.half else tensor

@apply_recursively()
def tensor_fp32_to_fp16(tensor, device, dtype, *args):
    return tensor.half() if isinstance(tensor, torch.Tensor) and tensor.dtype == torch.float and dtype == torch.half else tensor

@apply_recursively()
def tensor_to_contiguous(tensor, *args):
    if isinstance(tensor, torch.Tensor) and not tensor.is_contiguous():
        return tensor.contiguous()
    return tensor

@apply_recursively()
def tensor_identity(tensor, *args):
    return tensor