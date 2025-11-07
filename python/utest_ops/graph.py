import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import copy

torch.manual_seed(1000)
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
            self.scale = nn.Parameter(torch.rand(self.d))
        if self.with_bias:
            self.bias = nn.Parameter(torch.rand(self.d))

    def forward(self, x):
        ms = torch.mean(torch.square(x), dim=self.axis, keepdim=True)
        rms = torch.sqrt(ms + self.eps)
        y = x / rms
        if self.with_scale:
            y *= self.scale
        if self.with_bias:
            y += self.bias
        return y

def check_rmsnorm():
    import torch_tpu

    class RMSNormGraph:
        def __init__(self, input, output, scale=None, bias=None, axis=-1, eps=1e-8):
            self.static_scale = scale.to(device)
            self.static_bias = bias.to(device)
            self.static_input = input.to(device)
            self.static_output = output.to(device)
            self.graph = None
            self.axis = axis
            self.eps = eps

        def capture(self):
            self.graph = torch_tpu.tpu.TPUGraph(True)

            with torch_tpu.tpu.graph(self.graph):
                torch.ops.my_ops.rmsnorm_forward(self.static_input,
                                                 self.static_scale,
                                                 self.static_bias,
                                                 self.static_output,
                                                 self.axis,
                                                 self.eps)

        def replay(self):
            self.graph.replay()
            return self.static_output.cpu()

        def update_input(self, input):
            self.static_input.copy_(input)

    batch = 16
    hidden_size = 8192
    axis = 3
    eps = 1e-5

    net_cpu = RMSNorm(d=hidden_size, axis=axis, eps=eps, with_bias=True, with_scale=True)
    x1 = torch.randn((batch, 1, 1, hidden_size), requires_grad=False)
    x2 = torch.randn((batch, 1, 1, hidden_size), requires_grad=False)
    out1_cpu = net_cpu(x1)
    out2_cpu = net_cpu(x2)

    scale = None
    bias = None
    if 'scale' in net_cpu.state_dict():
        scale = net_cpu.state_dict()['scale'].clone().detach().contiguous().requires_grad_(False)
    if 'bias' in net_cpu.state_dict():
        bias = net_cpu.state_dict()['bias'].clone().detach().contiguous().requires_grad_(False)

    # construct graph
    tmp = torch.empty_like(out1_cpu)
    rms_graph = RMSNormGraph(x1, tmp, scale=scale, bias=bias, axis=axis, eps=eps)
    rms_graph.capture()

    # first call
    out1_tpu = rms_graph.replay()
    # second call
    rms_graph.update_input(x2)
    out2_tpu = rms_graph.replay()
    # 3th call
    rms_graph.update_input(x1)
    out3_tpu = rms_graph.replay()

    # print res
    out1_diff = out1_cpu - out1_tpu
    #print(out1_cpu[0][0][0][:50])
    #print(out1_tpu[0][0][0][:50])
    print(torch.max(abs(out1_diff)))
    assert (torch.max(abs(out1_diff)) < 1e-5)
    out2_diff = out2_cpu - out2_tpu
    #print(out2_cpu[0][0][0][:50])
    #print(out2_tpu[0][0][0][:50])
    print(torch.max(abs(out2_diff)))
    assert (torch.max(abs(out2_diff)) < 1e-5)
    out3_diff = out1_cpu - out3_tpu
    #print(out1_cpu[0][0][0][:50])
    #print(out3_tpu[0][0][0][:50])
    assert (torch.max(abs(out3_diff)) < 1e-5)
    print (torch.max(abs(out3_diff)))

if __name__=="__main__":
    import os
    if os.environ['CHIP_ARCH'] in ['bm1684x']:
        print(f'Skip test for this arch')
        sys.exit(0)

    if 'DISABLE_CACHE' in os.environ:
        del os.environ['DISABLE_CACHE']
    os.environ['PYTORCH_TPU_ALLOCATOR'] = 'caching'
    check_rmsnorm()
