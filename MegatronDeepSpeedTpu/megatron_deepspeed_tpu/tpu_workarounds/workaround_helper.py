from ..tpu_tensor_utils import *
from ..utils import print_log

def wrap_tpu_func_or_module(func, msg, input_fx, output_fx):
    def wrapper(*args, **kwargs):
        if len(args) > 0 and \
            (
                (isinstance(args[0], torch.Tensor) and has_tpu_tensor(args)) or
                (isinstance(args[0], torch.nn.Module) and has_tpu_tensor(args[0].parameters()))
            ):
            print_log(f"-{msg % func.__name__}, args: {get_tensor_info(args)}, kwargs: {get_tensor_info(kwargs)}")
            device = args[0].device
            dtype = args[0].dtype
            new_args = input_fx(args)
            new_kwargs = input_fx(kwargs)
            ans = output_fx(func(*new_args, **new_kwargs), device, dtype)
            if func.__name__.endswith("_") and not func.__name__.startswith("_"): # inplace
                args[0].copy_(ans)
                return args[0]
            return ans
        elif func.__name__ == "arange" and str(kwargs.get("device", "")).startswith("tpu"): # special case, creating tensor
            print_log(f"-{msg % func.__name__}, args: {get_tensor_info(args)}, kwargs: {get_tensor_info(kwargs)}")
            device = "tpu"
            dtype = torch.int32
            new_kwargs = {k: v for k, v in kwargs.items() if k != "device"}
            ans = output_fx(func(*args, **new_kwargs), device, dtype)
            return ans
        else:
            return func(*args, **kwargs)
    return wrapper

def wrap_64bit(func):
    return wrap_tpu_func_or_module(func, "TPU Unsupported 64bit op/module: %s, using 32bit",
                                   tensor_64bit_to_32bit, tensor_64bit_to_32bit)
 
def wrap_cpu(func):
    return wrap_tpu_func_or_module(func, "TPU Unsupported op: %s, falling back to CPU",
                                   tensor_tpu_to_cpu, tensor_cpu_to_tpu)

def wrap_contiguous(func):
    return wrap_tpu_func_or_module(func, "TPU Non-contiguous op: %s, using .contiguous()",
                                   tensor_to_contiguous, tensor_to_contiguous)

def wrap_fp32(func):
    _wrap_fp32 = lambda func: wrap_tpu_func_or_module(func, "TPU Unsupported fp16 op: %s, using fp32",
                                                      tensor_fp16_to_fp32, tensor_fp32_to_fp16)
    return wrap_if_arguments(has_16bit_fp_tensor, _wrap_fp32, func)
    
def wrap_if_arguments(condition, wrap, func, else_wrap=None):
    def wrapper(*args, **kwargs):
        if condition((args, kwargs)):
            return wrap(func)(*args, **kwargs)
        if else_wrap is not None:
            return else_wrap(func)(*args, **kwargs)
        return func(*args, **kwargs)
    return wrapper

def wrap_non_blocking(func):
    def wrapper(*args, **kwargs):
        if kwargs.get("non_blocking", False) == True:
            print_log(f"-TPU does not support non_blocking=True in tensor.{func.__name__}, using non_blocking=False")
            kwargs["non_blocking"] = False
        return func(*args, **kwargs)
    return wrapper