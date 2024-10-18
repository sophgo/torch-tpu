from ..utils import *
from ..tpu_tensor_utils import *
import pdb
import numpy as np
import os

class TensorSaveHelper:
     _tensor_names_ = {}
     _input_list = []
     _result_list = []

def get_tensor_name(name):
    count = TensorSaveHelper._tensor_names_.get(name, 0)
    TensorSaveHelper._tensor_names_[name] = count + 1
    return f"{name}_{count}"

def cache_result(name):
    if "ward_input_" in name:
        TensorSaveHelper._input_list.append(name)
    elif "ward_output_" in name:
        TensorSaveHelper._result_list.append(name)

def combine_npz(step):
    for io, lst in (("input", TensorSaveHelper._input_list), ("output", TensorSaveHelper._result_list)):
        save_fn = get_log_file_path(f"results/rank_{read_rank()}_{io}_{step}.npz")
        to_save = {}
        for name in lst:
            to_save[name] = dict(np.load(get_log_file_path(f"results/rank_{read_rank()}_{name}.npz")).items())
            os.remove(get_log_file_path(f"results/rank_{read_rank()}_{name}.npz"))
        if to_save:
            np.savez(save_fn, **to_flat_dict(to_save))
            print_log(f"- saved combined {io} in {save_fn}", flush=True)
            del to_save
    TensorSaveHelper._input_list = []
    TensorSaveHelper._result_list = []

def check_nan_inf(tensor, name):
    if has_nan_inf_tensor(tensor):
        print_log(f"- nan or inf in {name}", flush=True)
        # pdb.set_trace()
            
def register_hook(module: torch.nn.Module):
    def pre_hook(module, args, kwargs):
        func_name = str(module._get_name())
        print_log(f"{func_name} forward start", flush=True)
        save_result_tensors((*args, kwargs), f"forward_input_{func_name}")
        check_nan_inf(args, "args")
        check_nan_inf(kwargs, "kwargs")
        check_nan_inf(module.parameters(), "parameters")
        increase_indent()
        
    def post_hook(module, input, output):
        decrease_indent()
        func_name = str(module._get_name())
        save_result_tensors(output, f"forward_output_{func_name}")
        check_nan_inf(output, "output")
        print_log(f"{func_name} forward end", flush=True)
        
    def pre_backward_hook(module, grad_output):
        func_name = str(module._get_name())
        print_log(f"{func_name} backward start", flush=True)
        save_result_tensors(grad_output, f"backward_input_{func_name}")
        check_nan_inf(grad_output, "grad_output")
        check_nan_inf(module.parameters(), "parameters")
        increase_indent()
        
    def backward_hook(module, grad_input, grad_output):
        decrease_indent()
        func_name = str(module._get_name())
        save_result_tensors(grad_input, f"backward_output_{func_name}")
        check_nan_inf(grad_input, "grad_input")
        print_log(f"{func_name} backward end", flush=True)
    
    if not_started():
        rank = int(os.environ.get("RANK", "0"))
        local_rank = int(os.environ.get("REAL_LOCAL_RANK", "0"))
        try:
            from deepspeed import get_accelerator
            device = get_accelerator().device_name()
        except ImportError:
            device = "unknown"

        record_device(device)
        record_rank(rank)
        record_local_rank(local_rank)
        mark_start()
    
    module.register_forward_pre_hook(pre_hook, with_kwargs=True)
    module.register_forward_hook(post_hook)
    module.register_full_backward_pre_hook(pre_backward_hook)
    module.register_full_backward_hook(backward_hook)
    
def save_tensors(tensors, name, dir=".", save_grad_instead=False):
    if os.environ.get("DBG_SAVE_ALL", "0") == "0":
        return
    fn = get_log_file_path(f"{dir}/rank_{read_rank()}_{name}.npz")
    cnt = 0
    while os.path.exists(fn):
        cnt += 1
        if cnt == 1:
            print_log(f"- Warning: {fn} already exists", flush=True)
        fn2 = get_log_file_path(f"{dir}/rank_{read_rank()}_{name}_{cnt}.npz")
        fn = fn2
    if cnt > 0: print_log(f"- renaming to{fn}", flush=True)
    if save_grad_instead:
        np.savez(fn, **prepare_tensor_grad_for_save(tensors))
    else:
        np.savez(fn, **prepare_tensor_for_save(tensors))
    print_log(f"- saved {'grad' if save_grad_instead else 'tensor'} {name.replace('_', ' ')} in {fn}", flush=True)

def save_result_tensors(tensors, name):
    if os.environ.get("DBG_SAVE_ALL", "0") == "0" and os.environ.get("DBG_SAVE_RESULTS", "0") == "0":
        return
    ts_name = get_tensor_name(name)
    cache_result(ts_name)
    save_tensors(tensors, ts_name, "results")

def save_model_params(module, step):
    if os.environ.get("DBG_SAVE_ALL", "0") == "0" and os.environ.get("DBG_SAVE_PARAMS", "0") == "0":
        return
    save_tensors(dict(module.named_parameters()), f"params_{step}", "params")
   
def save_model_grads(module, step):
    if os.environ.get("DBG_SAVE_ALL", "0") == "0" and os.environ.get("DBG_SAVE_GRADS", "0") == "0":
        return
    save_tensors(module.optimizer.bit16_groups, f"bit16_groups_{step}", "grads", save_grad_instead=True)
