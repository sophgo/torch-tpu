import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from utils import compare_model_grad

import torch_tpu
torch.manual_seed(1000)

cases = \
"""
8	32	4	4
64	64	160	160
64	32	160	160
64	32	160	160
64	32	160	160
64	32	160	160
64	64	160	160
64	128	80	80
64	64	80	80
64	64	80	80
64	64	80	80
64	64	80	80
64	64	80	80
64	64	80	80
64	128	80	80
64	256	40	40
64	128	40	40
64	128	40	40
64	128	40	40
64	128	40	40
64	128	40	40
64	128	40	40
64	128	40	40
64	128	40	40
64	256	40	40
64	512	20	20
64	256	20	20
64	256	20	20
64	256	20	20
64	256	20	20
64	512	20	20
64	256	20	20
64	512	20	20
64	256	20	20
64	128	40	40
64	128	40	40
64	128	40	40
64	128	40	40
64	256	40	40
64	128	40	40
64	64	80	80
64	64	80	80
64	64	80	80
64	64	80	80
64	128	80	80
64	128	40	40
64	128	40	40
64	128	40	40
64	128	40	40
64	128	40	40
64	256	40	40
64	256	20	20
64	256	20	20
64	256	20	20
64	256	20	20
64	256	20	20
64	512	20	20
"""

def case1(use_fp16=False):
    """
    conv backward 
    """
    device = "tpu"
    cases_int = cases.split("\n")
    for case_str in cases_int:
        print("="*50)
        print(case_str)
        case_s = case_str.split('\t')
        if len(case_s) != 4 :
            continue
        B, C, H, W = int(case_s[0]), int(case_s[1]), int(case_s[2]), int(case_s[3])
        inp_cpu = torch.range(1, B*C*H*W).view((B,C,H,W))
        inp_tpu = copy.deepcopy(inp_cpu)
        
        inp_cpu.requires_grad = True
        inp_tpu = inp_tpu.to(device)
        if use_fp16: inp_tpu = inp_tpu.half()
        inp_tpu.requires_grad = True

        net_cpu = nn.BatchNorm2d(C)
        net_tpu = copy.deepcopy(net_cpu)
        net_tpu = net_tpu.to(device)
        if use_fp16: net_tpu = net_tpu.half()
        print( "tpu:", net_tpu.state_dict()['running_mean'].cpu())
        print( "cpu:", net_cpu.state_dict()['running_mean'].cpu())

        import pdb; pdb.set_trace()

        out_cpu = net_cpu(inp_cpu)
        out_tpu = net_tpu(inp_tpu)
        diff = inp_cpu - inp_tpu.cpu()
        print(torch.max(abs(diff)))
        print( "tpu:", net_tpu.state_dict()['running_mean'].cpu())
        print( "cpu:", net_cpu.state_dict()['running_mean'].cpu())

        import pdb; pdb.set_trace()

        grad_o = torch.ones(out_cpu.shape)
        grad_o_tpu = grad_o.to(device)
        if use_fp16: grad_o_tpu = grad_o_tpu.half()

        out_cpu.backward(grad_o)
        out_tpu.backward(grad_o_tpu)

        diff = inp_cpu.grad - inp_tpu.grad.cpu()
        print(torch.max(abs(diff)))
        compare_model_grad(net_cpu, net_tpu)
        print( "tpu:", net_tpu.state_dict()['running_mean'].cpu())
        print( "cpu:", net_cpu.state_dict()['running_mean'].cpu())

        import pdb; pdb.set_trace()

if __name__ == "__main__":
    case1(use_fp16 = False)