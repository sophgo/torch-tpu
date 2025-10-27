import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
from python.utest_ops.top_utest import TensorComparator
import sys

torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def test():
    num_step = 15
    params = [torch.randn((48, 48, 94, 32), requires_grad=True).to(device) for _ in range(num_step)]
    params_grad = [torch.randn_like(params[i]).to(device) for i in range(num_step)]  


    lr = 0.5
    beta1 = 0.9
    beta2 = 0.999
    weight_decay = 0.01
    eps = 1e-8
    amsgrad = False
    maximize = False

    def standard_adam_update(a, b, lr, beta1, beta2, weight_decay, eps, amsgrad=False, maximize=False):
        optimizer = torch.optim.Adam(a, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay, eps=eps, amsgrad=amsgrad)
        optimizer.zero_grad()
        for i in range(len(a)):
            a[i].grad = b[i]
        optimizer.step()

        return a


    import copy
    params_standard = [copy.deepcopy(params[i].detach().cpu()).requires_grad_(True) for i in range(num_step)]
    params_grad_cpu = [params_grad[i].clone().to('cpu') for i in range(num_step)]

    params_standard_updated = standard_adam_update(params_standard, params_grad_cpu, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize)

    params_fused = [params[i].clone().detach().requires_grad_().to(device) for i in range(num_step)]
    def tpu_adam(a, b, lr, beta1, beta2, weight_decay, eps, amsgrad=False, maximize=False):
        optimizer = torch.optim.Adam(a, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay, eps=eps, amsgrad=amsgrad, fused=True)
        optimizer.zero_grad()
        for i in range(len(a)):
            a[i].grad = b[i]
        optimizer.step()
        return a
    params_fused_update = tpu_adam(params_fused, params_grad, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize)


    comparator = TensorComparator()
    status_v = [comparator.cmp_result(params_standard_updated[i].detach().cpu(), params_fused_update[i].detach().cpu()) for i in range(len(params_standard))]
    status = True
    for i in range(len(status_v)):
        status = status and status_v[i]
    
    if not status:
        print("Optimizer Adam is wrong!")
        sys.exit(255)
    else:
        print("Optimizer Adam is correct!")

if __name__ == "__main__":
    import os
    if os.environ['CHIP_ARCH'] in ['bm1684x']:
        print(f'Skip test for this arch')
        sys.exit(0)

    test()
