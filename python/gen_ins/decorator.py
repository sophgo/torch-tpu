import collections
import os
from copy import deepcopy
import torch
from torch import nn
import numpy as np

level = 0

def write_log(msg):
    # return
    print(msg, flush=True)


def get_indent():
    global level
    return '|   ' * level

def get_tensor_info(tensor, depth=0):
    if isinstance(tensor, torch.Tensor):
        return str(list(tensor.shape))
    elif isinstance(tensor, (list, tuple)):
        if len(tensor) == 0:
            return ""
        depth += 1
        msg = ", ".join([get_tensor_info(e, depth=depth) for e in tensor])
        return f"({msg})" if depth > 1 else msg
    elif isinstance(tensor, dict):
        if not tensor:
            return ""
        depth += 1
        msg = ", ".join(
            [f"{k}: {get_tensor_info(v, depth=depth)}" for k, v in tensor.items()])
        return f"{{{msg}}}"
    else:
        return str(tensor)


def simulate_fp16(tensor):
    def _simulate_fp16(tensor):
        if isinstance(tensor, nn.Module):
            for p in tensor.parameters():
                p.data = p.half().float().data
            return tensor
        if tensor.dtype == torch.float32:
            tensor.data = tensor.half().float().data
        return tensor
    return _apply_to_tensor(tensor, _simulate_fp16)


def scale_tensor(tensor, scale):
    if scale == 1.0:
        return tensor
    return _apply_to_tensor(tensor, lambda t: t * scale if t.dtype in [torch.float32, torch.float16] else t)


def _apply_to_tensor(tensor, func):
    if isinstance(tensor, (torch.Tensor, nn.Module)):
        return func(tensor)
    elif isinstance(tensor, (list, tuple)):
        return tuple(_apply_to_tensor(e, func) for e in tensor)
    elif isinstance(tensor, dict):
        return {k: _apply_to_tensor(v, func) for k, v in tensor.items()}
    else:
        return tensor

_do_insert_input = False
_do_simulate_fp16 = False
_do_insert_output = False
_do_save = True

tmp_dir = "tmp/"
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

def do_insert_input(func_name):
    if not _do_insert_input:
        return False
    # if 'Softmax' in func_name: #and \
        # (not 'backward' in func_name and not 'grad_output' in func_name) :
        # return True
    # return False
    return True


def do_insert_output(func_name):
    if not _do_insert_output:
        return False
    # if 'LayerNorm' in func_name: # and \
        # (not 'backward' in func_name and not 'grad_input' in func_name):
        # return True
    # return False
    return True


def do_simulate_fp16(func_name):
    if not _do_simulate_fp16:
        return False
    if '_VocabParallelCrossEntropy' in func_name:
        return False
    return True

def do_save(func_name):
    if not _do_save:
        return False
    if 'ParallelTransformer' in func_name or \
        'params' in func_name or \
        'grads' in func_name:
        return True
    return False


def register_hook(module: nn.Module, device):
    def hook(module: nn.Module, input, output):
        global level
        level -= 1
        class_name = str(module._get_name())
        if device == "cpu" and do_simulate_fp16(class_name):
            output = simulate_fp16(output)
        save_result(class_name, None, output, device)
        write_log(f"{get_indent()}{class_name}.forward end [hook]")
        return output

    def backward_hook(module: nn.Module, grad_input, grad_output):
        global level
        level -= 1
        class_name = str(module._get_name())
        grad_input_save_name = f"{class_name}_grad_input"
        # grad_output_save_name =  f"{class_name}_grad_output"
        if hasattr(module, 'weight'):
            if module.weight.grad is not None:
                save_result( f"{class_name}_grad_weight", None, module.weight.grad, device)
            else:
                print("the weight is None")
        if hasattr(module, 'bias'):
            if module.bias.grad is not None:
                save_result( f"{class_name}_grad_bias", None, module.bias.grad, device)
            else:
                print("the bias is None")
        # if len(grad_input) > 1:
        #     save_result( f"{class_name}_grad_weight", None, grad_input[1], device)
        if device == "cpu" and do_simulate_fp16(class_name):
            grad_input = simulate_fp16(grad_input)
        grad_input_to_save = scale_tensor(grad_input, 1.0)
        # save_result(grad_output_save_name, None, grad_output, device)
        save_result(grad_input_save_name, None, grad_input_to_save, device)
        write_log(f"{get_indent()}{class_name}.backward end [hook]")
        return grad_input

    module.register_forward_hook(hook)
    module.register_full_backward_hook(backward_hook)

def add_tensor_to_dict(target_dict, name, tensor, save_non_tensor=False):
    device = None
    if isinstance(tensor, torch.Tensor):
        assert not name in target_dict
        target_dict[name] = tensor.cpu().clone().detach().numpy()
        device = str(tensor.device)
    elif isinstance(tensor, dict):
        for k, v in tensor.items():
            device = add_tensor_to_dict(
                target_dict, f"{name}_{k}", v, save_non_tensor) or device
    elif isinstance(tensor, (list, tuple)):
        for i, v in enumerate(tensor):
            device = add_tensor_to_dict(
                target_dict, f"{name}_{i}", v, save_non_tensor) or device
    elif save_non_tensor:
        assert not name in target_dict
        target_dict[name] = np.array(tensor)
    elif tensor is None:
        pass
    else:
        raise ValueError(f"Unknown type: {type(tensor)}")
    return device

def sanitize_filename(name):
    name = name.split("<function ")[-1]
    name = name.split(" at 0x")[0]
    return name

hash_map = [collections.defaultdict(lambda: collections.defaultdict(int)),
            collections.defaultdict(lambda: collections.defaultdict(int))]
save_cnt = [0, 0]



def save_result(name, self, result, device):
    print(name)
    # if not do_save(name):
    #     return
    if result is None:
        return
    name = sanitize_filename(name)
    if 'MakeViewlessTensor' in name or 'Float16Module.forward' in name:
        return
    if device == 'cpu':
        kn = f"{save_cnt[1]}_{name}_{hash_map[0][0][name]}"
        hash_map[0][0][name] += 1
    elif device == 'tpu':
        kn = f"{save_cnt[1]}_{name}_{hash_map[1][0][name]}"
        hash_map[1][0][name] += 1
    
    tensors = {}
    if isinstance(self, torch.Tensor):
        add_tensor_to_dict(tensors, "self", self)
    device = add_tensor_to_dict(tensors, "res", result)

    if tensors:
        if device is None:
            return
        if device.startswith("privateuseone"):
            device = "tpu"
        if device.startswith("cuda"):
            device = "cuda"
        filename = tmp_dir + f"{device}_results_{kn}"
        np.savez(filename, **tensors)
        write_log(f'{get_indent()}|-saved result in: {filename}.npz')
        filelist = open(tmp_dir + f"{device}_results.txt", "a")
        filelist.write(f"{filename}.npz\n")
        filelist.close()
