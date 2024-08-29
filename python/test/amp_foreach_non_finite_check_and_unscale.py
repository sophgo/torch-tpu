import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():
    found_inf = torch.zeros([1]).to(device).half()
    inv_scale = torch.ones([1]).to(device) * 70000
    grads = [torch.ones([320, 240]).to(device).half(), torch.ones([320, 240]).to(device).half()* 70000 ]
    torch._amp_foreach_non_finite_check_and_unscale_(
            grads,
            found_inf,
            inv_scale)
    print("found_inf: ", found_inf.cpu())
    print("inv_scale: ", inv_scale.cpu())
    for i in range(len(grads)):
       print("grad: ", grads[i].cpu()[:10])
    tensors = [torch.rand(3, 3, 3, 3).to(device) for i in range(15)]
    found_inf = torch.tensor(0, device=device, dtype=torch.float)
    inv_scale = torch.tensor(2.0, device=device, dtype=torch.float)
    torch._amp_foreach_non_finite_check_and_unscale_(tensors, found_inf, inv_scale)
    print("found_inf: ", found_inf.cpu())
    print("inv_scale: ", inv_scale.cpu())
    for i in range(len(grads)):
       print("grad: ", grads[i].cpu()[:10])
    grad = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 65536]).half().to(device)
    grad1 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).half().to(device)
    grad2 = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).half().to(device)
    found_inf = torch.tensor([0], dtype=torch.float).to(device)
    inv_scale = torch.tensor([1.0], dtype=torch.float).to(device)
    torch._amp_foreach_non_finite_check_and_unscale_([grad, grad1, grad2], found_inf, inv_scale)
    print(found_inf.cpu())
    
if __name__ == "__main__":
    case1()