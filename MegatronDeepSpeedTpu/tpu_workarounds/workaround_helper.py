from functools import wraps
from nnmoduletools.module_debugger.tensor_utils import *
from nnmoduletools.module_debugger.utils import print_log

# wrapper definition
# func, [[args_condition0, input_fx0, msg0], [elif_args_condition1, input_fx1, msg1], ..., [None, else_input_fx, msgN]],
# [[output_condition0, output_fx0, msg0, dependency0(optional)], [elif_output_condition1, output_fx1, msg1, dependency0(optional], ..., [None, else_output_fx, msgN, dependency0(optional]]
# args_condition: f((args, kwargs)) -> bool, note that args is a tuple of args and kwargs
# output_condition: f(output) -> bool
# msg: str, with %s for func.__name__
# dependency: index(indices) of input_conditions that should be satisfied to run the output_fx

def wrap_tpu_func_or_module(func, input_conditions, output_conditions):
    @wraps(func)
    def wrapper(*args, **kwargs):
        hit = None
        if len(args) > 0 and \
            (
                has_tpu_tensor(args) or
                (isinstance(args[0], torch.nn.Module) and has_tpu_tensor(args[0].parameters()))
            ):
            device = get_tensor_flat_attrs(args, "device")[0]
            dtype = get_tensor_flat_attrs(args, "dtype")[0]
            for i, [condition, input_fx, msg] in enumerate(input_conditions):
                if condition is None or condition((args, kwargs)):
                    if msg: print_log(f"- {msg % func.__name__}, args: {get_tensor_info(args)}, kwargs: {get_tensor_info(kwargs)}")
                    ans = func(*input_fx(args), **input_fx(kwargs))
                    hit = i
                    break
            else:
                ans = func(*args, **kwargs)
        else:
            ans = func(*args, **kwargs)
            device = kwargs.get("device", get_tensor_flat_attrs(ans, "device")[0])
            dtype = kwargs.get("dtype", get_tensor_flat_attrs(ans, "dtype")[0])

        for condition, output_fx, msg, *dependency in output_conditions:
            if dependency and (hit is None or not hit in dependency): continue
            if condition is None or condition(ans):
                if msg: print_log(f"- {msg % func.__name__}, args: {get_tensor_info(args)}, kwargs: {get_tensor_info(kwargs)}, output: {get_tensor_info(ans)}")
                ans = output_fx(ans, device, dtype)
                break
        if func.__name__.endswith("_") and not func.__name__.startswith("_"): # inplace
            args[0].copy_(ans)
            return args[0]
        return ans
    return wrapper

def using_32bit_impl(func):
    return wrap_tpu_func_or_module(func, 
                                   [
                                       [has_64bit_tensor, tensor_64bit_to_32bit, "TPU op/module %s found 64bit in input, using 32bit"],
                                   ],
                                   [
                                       [has_64bit_tensor, tensor_64bit_to_32bit, "TPU op/module %s found 64bit in output, using 32bit"],
                                   ])
    
def using_cpu_impl(func, extra_input_condition=None, extra_output_condition=None):
    return wrap_tpu_func_or_module(func, 
                                   [
                                       [extra_input_condition, tensor_tpu_to_cpu, "TPU op %s using CPU impl"],
                                   ],
                                   [
                                       [extra_output_condition, tensor_cpu_to_tpu, None, 0],
                                   ])
    
def using_contiguous_impl(func):
    return wrap_tpu_func_or_module(func, 
                                   [
                                       [has_non_contiguous_tensor, tensor_to_contiguous, "TPU op %s found non-contiguous in input, using .contiguous()"],
                                   ],
                                   [
                                       [has_non_contiguous_tensor, tensor_to_contiguous, "TPU op %s found non-contiguous in output, using .contiguous()"],
                                   ])
    
def using_fp32_impl(func):
    return wrap_tpu_func_or_module(func, 
                                   [
                                       [has_16bit_fp_tensor, tensor_fp16_to_fp32, "TPU op %s fp16, using fp32 impl"],
                                   ],
                                   [
                                       [None, tensor_fp32_to_fp16, None, 0],
                                   ])

def using_fp16_impl(func):
    return wrap_tpu_func_or_module(func, 
                                   [
                                       [has_32bit_fp_tensor, tensor_fp32_to_fp16, "TPU op %s fp32, using fp16 impl"],
                                   ],
                                   [
                                       [None, tensor_fp16_to_fp32, None, 0],
                                   ])
    
def ignoring_empty_input(func):
    return wrap_tpu_func_or_module(func, 
                                   [
                                       [has_empty_tensor, tensor_nonempty, "TPU op %s found empty tensor, ignoring"],
                                   ],
                                   [
                                       
                                   ])
