import copy
import torch
import torch.nn as nn
from utils import ForwardHack, BackwardHack

def dump_forward():
    """
    gelu 
    """
    device = "privateuseone"
    batch = 8
    sequence = 2
    hidden_size = 8

    inp_cpu = torch.rand(batch, sequence, hidden_size)
    inp_tpu = copy.deepcopy(inp_cpu).to(device)
    inp_tpu.requires_grad = True

    grad_cpu = torch.rand(batch, sequence, hidden_size)
    grad_tpu = grad_cpu.to(device)

    net_cpu = nn.GELU()
    net_tpu = copy.deepcopy(net_cpu).to(device)

    inp_tpu = ForwardHack.apply("gelu", inp_tpu)
    out_tpu = net_tpu(inp_tpu)
    out_tpu.backward(grad_tpu)

def dump_backward():
    """
    gelu 
    """
    device = "privateuseone"
    batch = 8
    sequence = 2
    hidden_size = 8

    inp_cpu = torch.rand(batch, sequence, hidden_size)
    inp_tpu = copy.deepcopy(inp_cpu).to(device)
    inp_tpu.requires_grad = True

    grad_cpu = torch.rand(batch, sequence, hidden_size)
    grad_tpu = grad_cpu.to(device)

    net_cpu = nn.GELU()
    net_tpu = copy.deepcopy(net_cpu).to(device)

    inp_tpu = BackwardHack.apply("gelu_back", inp_tpu)
    out_tpu = net_tpu(inp_tpu)
    out_tpu.backward(grad_tpu)

if __name__ == "__main__":
    dump_forward()