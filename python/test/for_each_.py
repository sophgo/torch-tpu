import torch
import torch_tpu
import torch.nn as nn

def _foreach_add():
    print("==== start _foreach_add ====")
    params = []
    grads = []
    device_params = []
    device_grads = []
    N,C,H,W = 2, 3, 8, 8
    for i in range(3):
        param = nn.Parameter(torch.rand((N,C,H,W)))
        grad = torch.ones_like(param)
        params.append(param)
        grads.append(grad)
        device_params.append(param.tpu())
        device_grads.append(grad.tpu())
    results = None
    results_tpu = None
    with torch.no_grad():
        results = torch._foreach_add(params, grads, alpha=-1)
        results_tpu = torch._foreach_add(device_params, device_grads, alpha=-1)
    for i in range(len(results)):
        diff = abs(results[i] - results_tpu[i].cpu())
        print(f"idx: {i}, max diff:{torch.max(diff)}")
    print("==== end _foreach_add =====")

def _foreach_add_():
    print("==== start _foreach_add_ inplace ====")
    params = []
    grads = []
    device_params = []
    device_grads = []
    N,C,H,W = 2, 3, 8, 8
    for i in range(3):
        param = nn.Parameter(torch.rand((N,C,H,W)))
        grad = torch.ones_like(param)
        params.append(param)
        grads.append(grad)
        device_params.append(param.tpu())
        device_grads.append(grad.tpu())
    with torch.no_grad():
        torch._foreach_add_(params, grads, alpha=-1)
        torch._foreach_add_(device_params, device_grads, alpha=-1)
    for i in range(len(params)):
        diff = abs(params[i] - device_params[i].cpu())
        print(f"idx: {i}, max diff:{torch.max(diff)}")
    print("==== end _foreach_add_ inplace =====")


def _foreach_mul_():
    print("==== start _foreach_mul_ inplace ====")
    params = []
    grads = []
    device_params = []
    N,C,H,W = 2, 3, 8, 8
    for i in range(3):
        param = nn.Parameter(torch.rand((N,C,H,W)))
        params.append(param)
        device_params.append(param.tpu())
    with torch.no_grad():
        torch._foreach_mul_(params, 2.0)
        torch._foreach_mul_(device_params, 2.0)
    for i in range(len(params)):
        diff = abs(params[i] - device_params[i].cpu())
        print(f"idx: {i}, max diff:{torch.max(diff)}")
    print("==== end _foreach_mul_ inplace =====")

def _foreach_mul():
    print("==== start _foreach_mul ====")
    params = []
    device_params = []
    N,C,H,W = 2, 3, 8, 8
    for i in range(3):
        param = nn.Parameter(torch.rand((N,C,H,W)))
        params.append(param)
        device_params.append(param.tpu())
    with torch.no_grad():
        results = torch._foreach_mul(params, 2.0)
        results_tpu = torch._foreach_mul(device_params, 2.0)
    for i in range(len(results)):
        diff = abs(results[i] - results_tpu[i].cpu())
        print(f"idx: {i}, max diff:{torch.max(diff)}")
    print("==== end _foreach_mul =====")

def _foreach_neg():
    print("==== start _foreach_neg ====")
    params = []
    device_params = []
    N,C,H,W = 2, 3, 8, 8
    for i in range(3):
        param = torch.rand((N,C,H,W))
        params.append(param)
        device_params.append(param.tpu())
    with torch.no_grad():
        results = torch._foreach_neg(params)
        results_tpu = torch._foreach_neg(device_params)
    for i in range(len(results)):
        diff = abs(results[i] - results_tpu[i].cpu())
        print(f"idx: {i}, max diff:{torch.max(diff)}")
    print("==== end _foreach_neg =====")

def _foreach_neg_():
    print("==== start _foreach_neg_ inplace ====")
    params = []
    grads = []
    device_params = []
    N,C,H,W = 2, 3, 8, 8
    for i in range(3):
        param = torch.rand((N,C,H,W))
        params.append(param)
        device_params.append(param.tpu())
    with torch.no_grad():
        torch._foreach_neg_(params)
        torch._foreach_neg_(device_params)
    for i in range(len(params)):
        diff = abs(params[i] - device_params[i].cpu())
        print(f"idx: {i}, max diff:{torch.max(diff)}")
    print("==== end _foreach_neg_ inplace =====")

def  _foreach_sqrt():
    print("==== start _foreach_sqrt ====")
    params = []
    device_params = []
    N,C,H,W = 2, 3, 8, 8
    for i in range(3):
        param = torch.rand((N,C,H,W))
        params.append(param)
        device_params.append(param.tpu())
    with torch.no_grad():
        results = torch._foreach_sqrt(params)
        results_tpu = torch._foreach_sqrt(device_params)
    for i in range(len(results)):
        diff = abs(results[i] - results_tpu[i].cpu())
        print(f"idx: {i}, max diff:{torch.max(diff)}")
    print("==== end _foreach_sqrt =====")

def  _foreach_addcmul_():
    print("==== start _foreach_addcmul_ inplace ====")
    params = []
    grads = []
    device_params = []
    device_grads = []
    N,C,H,W = 2, 3, 8, 8
    for i in range(3):
        param = torch.rand((N,C,H,W))
        grad = torch.ones_like(param)
        params.append(param)
        grads.append(grad)
        device_params.append(param.tpu())
        device_grads.append(grad.tpu())
    with torch.no_grad():
        torch._foreach_addcmul_(params, grads, grads, 0.1)
        torch._foreach_addcmul_(device_params, device_grads, device_grads, 0.1)
    for i in range(len(params)):
        diff = abs(params[i] - device_params[i].cpu())
        print(f"idx: {i}, max diff:{torch.max(diff)}")
    print("==== end _foreach_addcmul_ inplace =====")

def  _foreach_addcdiv_():
    print("==== start _foreach_addcdiv_ inplace ====")
    params = []
    grads = []
    device_params = []
    device_grads = []
    N,C,H,W = 2, 3, 8, 8
    for i in range(3):
        param = torch.rand((N,C,H,W))
        grad = torch.ones_like(param)
        params.append(param)
        grads.append(grad)
        device_params.append(param.tpu())
        device_grads.append(grad.tpu())
    with torch.no_grad():
        torch._foreach_addcdiv_(params, grads, grads, 0.1)
        torch._foreach_addcdiv_(device_params, device_grads, device_grads, 0.1)
    for i in range(len(params)):
        diff = abs(params[i] - device_params[i].cpu())
        print(f"idx: {i}, max diff:{torch.max(diff)}")
    print("==== end _foreach_addcdiv_ inplace =====")

def  _foreach_lerp_():
    print("==== start _foreach_lerp_ inplace ====")
    params = []
    grads = []
    device_params = []
    device_grads = []
    N,C,H,W = 2, 3, 8, 8
    for i in range(3):
        param = torch.rand((N,C,H,W))
        grad = torch.ones_like(param)
        params.append(param)
        grads.append(grad)
        device_params.append(param.tpu())
        device_grads.append(grad.tpu())
    with torch.no_grad():
        torch._foreach_lerp_(params, grads, 0.1)
        torch._foreach_lerp_(device_params, device_grads, 0.1)
    for i in range(len(params)):
        diff = abs(params[i] - device_params[i].cpu())
        print(f"idx: {i}, max diff:{torch.max(diff)}")
    print("==== end _foreach_lerp_ inplace =====")

def _foreach_div_():
    print("==== start _foreach_div_ inplace ====")
    params = []
    grads = []
    device_params = []
    device_grads = []
    N,C,H,W = 2, 3, 8, 8
    for i in range(3):
        param = torch.rand((N,C,H,W))
        grad = 1
        params.append(param)
        grads.append(grad)
        device_params.append(param.tpu())
        device_grads.append(1)
    with torch.no_grad():
        torch._foreach_div_(params, grads)
        torch._foreach_div_(device_params, device_grads)
    for i in range(len(params)):
        diff = abs(params[i] - device_params[i].cpu())
        print(f"idx: {i}, max diff:{torch.max(diff)}")
    print("==== end _foreach_div_ inplace =====")

if __name__ == "__main__":
    _foreach_add()
    _foreach_add_()
    _foreach_mul()
    _foreach_mul_()
    _foreach_neg()
    _foreach_neg_()

    _foreach_sqrt()
    _foreach_addcmul_()
    _foreach_addcdiv_()
    _foreach_lerp_()
    _foreach_div_()