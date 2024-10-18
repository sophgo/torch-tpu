import torch
# torch.random.manual_seed(0)
from torch import nn
# import torch_tpu
import types
import copy
import sys
sys.path.append("/workspace/tpu-mlir/python/tools")
sys.path.append("~/workspace/tpu-mlir/python/tools")
import compare_visualizer as utils
from megatron_deepspeed_tpu.tpu_tensor_utils import *

@apply_recursively()
def new_tpu_tensor(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.new_tensor(tensor, device="tpu", requires_grad=tensor.requires_grad)
    return tensor

@apply_recursively()
def new_cpu_tensor(tensor):
    if isinstance(tensor, torch.Tensor): 
        if tensor.dtype in [torch.half, torch.bfloat16]:
            return tensor.new_tensor(tensor, dtype=torch.float, device="cpu", requires_grad=tensor.requires_grad)
        return tensor.new_tensor(tensor, dtype=tensor.dtype, device="cpu", requires_grad=tensor.requires_grad)
    return tensor

@apply_recursively()
def new_tpu_tensor(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.new_tensor(tensor, device="tpu", requires_grad=tensor.requires_grad)
    return tensor

@apply_recursively(torch.concat)
def get_1d_tensor(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.view(-1)
    return torch.tensor([])
    
@apply_recursively()
def get_tensor_grads(tensor):
    if isinstance(tensor, torch.Tensor) and tensor.grad is not None:
        return tensor.grad

def get_diff(target, ref):
    target_1d = get_1d_tensor(target).float()
    ref_1d = get_1d_tensor(ref).float()
    abs_diff = torch.abs(target_1d - ref_1d)
    rel_diff = abs_diff / (torch.abs(ref_1d) + 1e-10)
    abs_diff_max = torch.max(abs_diff)
    abs_diff_idx = torch.argmax(abs_diff)
    rel_diff_max = torch.max(rel_diff)
    rel_diff_idx = torch.argmax(rel_diff)
    similarities = [utils.calc_similarity(_target.detach().numpy(), _ref.detach().numpy()) for _target, _ref in zip(target, ref) if isinstance(_target, torch.Tensor) and isinstance(_ref, torch.Tensor)]
    return (abs_diff_max, target_1d[abs_diff_idx], ref_1d[abs_diff_idx]), (rel_diff_max, target_1d[rel_diff_idx], ref_1d[rel_diff_idx]), similarities
    
def random_tensor(*shape, dtype=torch.float32, min_val=-1, max_val=1, requires_grad=False):
    if len(shape) == 0:
        return random_tensor(1, dtype=dtype, min_val=min_val, max_val=max_val, requires_grad=requires_grad).reshape(tuple())
    if dtype in (torch.int, torch.int64):
        return torch.randint(int(min_val), int(max_val)+1, shape, dtype=dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, shape, dtype=dtype)
    return torch.rand(*shape, dtype=dtype, requires_grad=requires_grad) * (max_val - min_val) + min_val

def test_function_with_args(func, args=(), kwargs={}):
    print("=" * 50)
    print(f"Testing {func.__name__}: {get_tensor_info(args)} {get_tensor_info(kwargs)}")
    tpu_res = new_cpu_tensor(func(*new_tpu_tensor(args), **new_tpu_tensor(kwargs)))
    cpu_res = func(*new_cpu_tensor(args), **new_cpu_tensor(kwargs))
    abs_diff, rel_diff, similarity = get_diff((tpu_res, ), (cpu_res, ))
    print(f"TPU result: {tpu_res}")
    print(f"CPU result: {cpu_res}")
    print("Compare:", "\n".join(map(str, similarity)))
    print(f"Max abs diff: {abs_diff[0]}, target={abs_diff[1]}, ref={abs_diff[2]}")
    print(f"Max rel diff: {rel_diff[0]}, target={rel_diff[1]}, ref={rel_diff[2]}")
    
def test_module_with_args(module, args=(), kwargs={}, backward_inputs=None):
    print("=" * 50)
    print(f"Testing {module.__class__.__name__}: {get_tensor_info(args)} {get_tensor_info(kwargs)}")
    module_cpu = copy.deepcopy(module).float()
    module_tpu = module.to("tpu")
    # forward
    tpu_args, cpu_args = new_tpu_tensor(args), new_cpu_tensor(args)
    tpu_kwargs, cpu_kwargs = new_tpu_tensor(kwargs), new_cpu_tensor(kwargs)
    tpu_res = module_tpu(*tpu_args, **tpu_kwargs)
    cpu_res = module_cpu(*cpu_args, **cpu_kwargs)
    if isinstance(tpu_res, torch.Tensor):
        tpu_res = (tpu_res,)
    if isinstance(cpu_res, torch.Tensor):
        cpu_res = (cpu_res,)
    _tpu_res = new_cpu_tensor(tpu_res)
    res_abs_diff, res_rel_diff, res_similarity = get_diff(_tpu_res, cpu_res)
    # backward
    if backward_inputs is None:
        backward_input_cpu = random_tensor(*(cpu_res[0].shape), dtype=cpu_res[0].dtype, max_val=10.)
        backward_input_tpu = copy.deepcopy(backward_input_cpu).to(tpu_res[0].dtype).to("tpu")
    else:
        backward_input_cpu = new_cpu_tensor(backward_inputs)
        backward_input_tpu = new_tpu_tensor(backward_inputs)
        
    tpu_res[0].backward(backward_input_tpu)
    cpu_res[0].backward(backward_input_cpu)
    tpu_grads = new_cpu_tensor((*get_tensor_grads(module_tpu.parameters()), *get_tensor_grads(tpu_args), *get_tensor_grads(tpu_kwargs)))
    cpu_grads = (*get_tensor_grads(module_cpu.parameters()), *get_tensor_grads(cpu_args), *get_tensor_grads(cpu_kwargs))
    grad_abs_diff, grad_rel_diff, grads_similarity = get_diff(tpu_grads, cpu_grads)
    
    print(f"TPU result: {_tpu_res}")
    print(f"CPU result: {cpu_res}")
    print(f"TPU grads: {tpu_grads}")
    print(f"CPU grads: {cpu_grads}")
    print("Result compare:", "\n".join(map(str, res_similarity)))
    print(f"Res Max abs diff: {res_abs_diff[0]}, target={res_abs_diff[1]}, ref={res_abs_diff[2]}")
    print(f"Res Max rel diff: {res_rel_diff[0]}, target={res_rel_diff[1]}, ref={res_rel_diff[2]}")
    print("Grad compare:", "\n".join(map(str, grads_similarity)))
    print(f"Grad Max abs diff: {grad_abs_diff[0]}, target={grad_abs_diff[1]}, ref={grad_abs_diff[2]}")
    print(f"Grad Max rel diff: {grad_rel_diff[0]}, target={grad_rel_diff[1]}, ref={grad_rel_diff[2]}")
    
from megatron_deepspeed_tpu import tpu_workarounds # this line must before case_list if use workaround
tpu_workarounds.workaround_helper.print_log = lambda *args, **kwargs: print(*args, **kwargs)

case_list = [ # (func, args[, kwargs])
    # (torch.matmul, (random_tensor(2, 8, 21, 8, dtype=torch.half), random_tensor(2, 8, 8, 21, dtype=torch.half))),
    # (torch.add, (random_tensor(2, 8, 177, 177, dtype=torch.half), random_tensor(2, 1, 1, 177, dtype=torch.half))),
    # (torch.isnan, (random_tensor(50265, 192, dtype=torch.half), )),
    # (torch.isinf, (random_tensor(50265, 192, dtype=torch.half), )),
    # (torch.logical_or, (random_tensor(50265, 256, dtype=torch.bool), random_tensor(50265, 256, dtype=torch.bool))),
    # (torch.tanh, (random_tensor(2, 256, dtype=torch.half), )),
    # (torch.nonzero, (random_tensor(354, dtype=torch.bool), )),
    # (torch.index_select, (random_tensor(354, 256, dtype=torch.half), 0, random_tensor(9, dtype=torch.int32, min_val=0, max_val=353))),
    # (torch.nn.functional.cross_entropy, (random_tensor(2, 50265, dtype=torch.half), random_tensor(2, dtype=torch.int64, min_val=0, max_val=50264))),
    # (torch.max, (random_tensor(50265, 256, dtype=torch.float), )),
    # (torch.cumsum, (random_tensor(2, 177, dtype=torch.int32, min_val=0, max_val=1), ), {"dim": 1}),
    # (torch.arange, (10, ), {"device": "tpu"}),
]
    
module_list = [ # (module, args[, kwargs, backward_inputs])
    # (torch.nn.Embedding(50265, 192), (random_tensor(2, 230, dtype=torch.int32, min_val=0, max_val=50264), )),
    # (torch.nn.Embedding(514, 256), (random_tensor(2, 177, dtype=torch.int32, min_val=0, max_val=513), )),
    # (torch.nn.LayerNorm(256, dtype=torch.half), (random_tensor(2, 256, dtype=torch.half), )),
    # (torch.nn.Tanh(), (random_tensor(2, 256, dtype=torch.half, requires_grad=True), )),
]

if __name__ == "__main__":
    # for case in case_list:
    #     test_function_with_args(*case)
    # for case in module_list:
    #     test_module_with_args(*case)
    # exit()

    # import numpy as np
    # from pathlib import Path
    # input_npz = np.load(Path("/workspace/DeepSpeedExamples/training/HelloDeepSpeed/logs_2024.04.07.17.17.00") / "results/rank_0_input_1.npz")
    # linear_input = torch.tensor(input_npz['forward_input_Linear_4_arg0'], dtype=torch.half)
    # # embedding_input = torch.tensor(input_npz['forward_input_Embedding_0_arg0'], dtype=torch.int32)
    # # embedding_bw_input = torch.tensor(input_npz['backward_input_Embedding_0_arg0'], dtype=torch.half)
    # param_npz = np.load(Path("/workspace/DeepSpeedExamples/training/HelloDeepSpeed/logs_2024.04.07.17.17.00") / "params/rank_0_params_0.npz")
    # linear_weight = torch.tensor(param_npz['module.encoder.encoder.layer.4.intermediate.dense.weight'], dtype=torch.half)
    # linear_bias = torch.tensor(param_npz['module.encoder.encoder.layer.4.intermediate.dense.bias'], dtype=torch.half)
    
    class MyModule(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return x.sum()
        
    module = MyModule()
    test_module_with_args(module, (random_tensor(1, 256, dtype=torch.half, requires_grad=True) ))