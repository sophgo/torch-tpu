import torch
import torch.nn as nn
import sys
import copy
import transformers.models
from top_utest import TensorComparator
from torch_tpu.tpu.custom_op.rmsnorm import RMSNormFunc

import torch_tpu
torch.manual_seed(10086)
torch.set_printoptions(precision=6)
device = "tpu:0"

class RMSNorm(nn.Module):
    def __init__(self, d=0, axis=-1., eps=1e-8, with_scale=False, with_bias=False):
        super(RMSNorm, self).__init__()
        self.d = d
        self.axis = axis
        self.eps = eps
        self.with_bias = with_bias
        self.with_scale = with_scale
        if self.with_scale:
            self.scale = nn.Parameter(torch.rand(self.d),requires_grad=True)
        if self.with_bias:
            self.bias = nn.Parameter(torch.rand(self.d),requires_grad=True)

    def forward(self, x):
        ms = torch.mean(torch.square(x), dim=self.axis, keepdim=True)
        rms = torch.sqrt(ms + self.eps)
        y = x / rms
        if self.with_scale:
            y *= self.scale
        if self.with_bias:
            y += self.bias
        return y

class RMSNormBlock(nn.Module):
    def __init__(self, scale=None, bias=None, axis=-1, eps=1e-8):
        super().__init__()
        self.scale = scale
        self.bias = bias
        self.axis = axis
        self.eps = eps

    def forward(self, x):
        return RMSNormFunc.apply(x, self.scale, self.bias, self.axis, self.eps)


def check_rmsnorm():
    batch = 16
    hidden_size = 8192
    axis = 3
    eps = 1e-5

    net_cpu = RMSNorm(d=hidden_size, axis=axis, eps=eps, with_bias=True, with_scale=True)

    x = torch.randn((batch, 1, 1, hidden_size), requires_grad=True)
    x_tpu = copy.deepcopy(x).to(device).half()
    x_tpu.retain_grad()

    out_cpu = net_cpu(x)

    scale = None
    bias = None
    if 'scale' in net_cpu.state_dict():
        scale = net_cpu.state_dict()['scale'].clone().detach().contiguous().requires_grad_(True).to(device).half()
    if 'bias' in net_cpu.state_dict():
        bias = net_cpu.state_dict()['bias'].clone().detach().contiguous().requires_grad_(True).to(device).half()
    bias.retain_grad()
    scale.retain_grad()

    net_tpu = RMSNormBlock(axis=axis, eps=eps, scale=scale, bias=bias)
    out_tpu = net_tpu(x_tpu)
    out_tpu = out_tpu.float()

    vocab_size = 32000
    label_cpu = torch.randint(0,vocab_size,(batch,hidden_size))
    label_tpu = copy.deepcopy(label_cpu).to(device)
    classifier_cpu = nn.Linear(hidden_size, hidden_size)
    classifier_tpu = copy.deepcopy(classifier_cpu).to(device)
    logits_cpu = classifier_cpu(out_cpu.view(batch, hidden_size)) 
    loss_cpu = (logits_cpu - label_cpu).sum()
    loss_cpu.backward()

    logits_tpu = classifier_tpu(out_tpu.view(batch, hidden_size))
    loss_tpu = (logits_tpu - label_tpu).sum()
    loss_tpu.backward()
    
    comparator = TensorComparator()
    status_forward = comparator.cmp_result(logits_cpu.detach(), logits_tpu.cpu().detach().float())
    status_backward_x = comparator.cmp_result(x.grad, x_tpu.grad.cpu().float())
    status_backward_scale = comparator.cmp_result(net_cpu.scale.grad.detach(), net_tpu.scale.grad.cpu().detach().float())
    status_backward_bias = comparator.cmp_result(net_cpu.bias.grad.detach(), net_tpu.bias.grad.cpu().detach().float())
    status_backward = status_backward_bias and status_backward_scale and status_backward_x
    return status_forward,status_backward

if __name__=="__main__":
    status_forward, status_backward = check_rmsnorm()
    print("================== Test rmsnorm forward ======================")
    if status_forward == False:
        print(f"[Failed] llama_rmsnorm forward compare failed!")
        sys.exit(255)
    else:
        print(f"[Success] llama_rmsnorm forward compare succeed!")
    print("================== Test rmsnorm backward ======================")
    if status_backward == False:
        print(f"[Failed] llama_rmsnorm backward ompare failed!")
        sys.exit(255)
    else:
        print(f"[Success] llama_rmsnorm backward compare succeed!")
