import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():
    found_inf = torch.tensor([0.0], device=device)
    inv_scale = torch.tensor([2.0]).to(device) 
    grads = [torch.ones([320, 240]).to(device).half(), torch.ones([320, 240]).to(device).half()* 70000 ]
    torch._amp_foreach_non_finite_check_and_unscale_(
            grads,
            found_inf,
            inv_scale)
    torch_tpu.tpu.synchronize()
    print("found_inf: ", found_inf.cpu())
    print("inv_scale: ", inv_scale.cpu())
    for i in range(len(grads)):
       print("grad: ", grads[i].cpu()[:10])

    
    grad = torch.tensor([float('nan'), 3.0, 4.0, 5.0], dtype=torch.float16, device=device)
    grad1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float16, device=device)
    grad2 = torch.tensor([1.0, 2.0, float('inf'), 6.0], dtype=torch.float16, device=device)
    found_inf = torch.tensor([0], dtype=torch.float).to(device)
    inv_scale = torch.tensor([2.0], dtype=torch.float).to(device)
    torch._amp_foreach_non_finite_check_and_unscale_([grad1, grad1, grad1, grad2, grad], found_inf, inv_scale)
    torch.tpu.synchronize()
    cpu_nan1 = torch.isnan(grad.cpu()).sum() + torch.isnan(grad1.cpu()).sum() + torch.isnan(grad2.cpu()).sum()
    cpu_inf1 = torch.isinf(grad.cpu()).sum() + torch.isinf(grad1.cpu()).sum() + torch.isinf(grad2.cpu()).sum()
    print(f"cpu found_inf: {cpu_nan1} {cpu_inf1}")
    print("found_inf: ", found_inf.item())
    print(f"grad: {grad.cpu()}")
    print(f"grad1: {grad1.cpu()}")
    print(f"grad2: {grad2.cpu()}")
if __name__ == "__main__":
    case1()