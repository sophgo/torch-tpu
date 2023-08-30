import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

batch_size = 4
length = 10
embed_dim = 32
intermediate_size = 128
eps = 1e-5
input_dim = 3


class GPT2AddlnMatmul(nn.Module):
    def __init__(self, embed_dim, intermediate_size, layer_norm_epsilon):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=layer_norm_epsilon)
        self.fc = nn.Linear(embed_dim, intermediate_size)

    def forward(self, x1, x2):
        x_ = torch.add(x1, x2)
        x = self.ln(x_)
        x = self.fc(x)
        return x_, x


class AddlnMatmulFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, w, b, gamma, beta):
        D = w.shape[1]

        if x1.dim() == 3:
            B, M, N = x1.shape
            mean = torch.empty((B, M), dtype = x1.dtype, device = x1.device)
            rstd = torch.empty((B, M), dtype = x1.dtype, device = x1.device)
            out = torch.empty((B, M, D), dtype = x1.dtype, device = x1.device)
        else:
            M, N = x1.shape
            mean = torch.empty((M,), dtype = x1.dtype, device = x1.device)
            rstd = torch.empty((M,), dtype = x1.dtype, device = x1.device)
            out = torch.empty((M, D), dtype = x1.dtype, device = x1.device)
        out_add = torch.empty(x1.shape, dtype = x1.dtype, device = x1.device)

        assert x1.shape == x2.shape
        assert w.shape == (N, D)
        assert gamma.shape == (N,)
        assert beta.shape == (N,)
        if b != None:
            assert b.shape == (D,)

        torch.ops.my_ops.add_ln_mm_forward(x1,
                                     x2,
                                     w,
                                     b,
                                     gamma,
                                     beta,
                                     eps,
                                     out_add,
                                     mean,
                                     rstd,
                                     out)

        ctx.save_for_backward(x1, x2, w, mean, rstd)

        return out_add, mean, rstd, out 


class AddlnMatmulBlock(nn.Module):
    def __init__(self, w, b, gamma, beta):
        super().__init__()
        self.w = w
        self.b = b
        self.gamma = gamma
        self.beta = beta

    def forward(self, x1, x2):
        return AddlnMatmulFunc.apply(x1, x2, self.w, self.b, self.gamma, self.beta)
    


def check_add_ln_matmul():
    net_cpu = GPT2AddlnMatmul(embed_dim, intermediate_size, eps)

    gamma = net_cpu.state_dict()['ln.weight'].clone().detach().contiguous().requires_grad_(True).to(device).half()
    beta = None
    if 'ln.bias' in net_cpu.state_dict().keys():
        beta = net_cpu.state_dict()['ln.bias'].clone().detach().requires_grad_(True).to(device).half()
   
    w = net_cpu.state_dict()['fc.weight'].clone().detach().transpose(0,1).contiguous().requires_grad_(True).to(device).half()
    b = None
    if 'fc.bias' in net_cpu.state_dict().keys():
        b = net_cpu.state_dict()['fc.bias'].clone().detach().requires_grad_(True).to(device).half()
    
    net_tpu = AddlnMatmulBlock(w, b, gamma, beta)

    print("=====forward======")
    if input_dim == 3:
        x1 = torch.randn(batch_size, length, embed_dim, requires_grad=True)
        x2 = torch.randn(batch_size, length, embed_dim, requires_grad=True)
    else:
        x1 = torch.randn(length, embed_dim, requires_grad=True)
        x2 = torch.randn(length, embed_dim, requires_grad=True)
    x1_tpu = x1.to(device).half()
    x2_tpu = x2.to(device).half()
    out_add_tpu, mean_tpu, rstd_tpu, out_tpu = net_tpu(x1_tpu, x2_tpu)
    out_add_cpu, out_cpu = net_cpu(x1, x2)

    mean_cpu = out_add_cpu.mean(-1)
    rstd_cpu = 1/(out_add_cpu.var(-1) + eps).sqrt()

    add_diff = out_add_cpu - out_add_tpu.float().to("cpu")
    mean_diff = mean_cpu - mean_tpu.float().to("cpu")
    rstd_diff = rstd_cpu - rstd_tpu.float().to("cpu")
    out_diff = out_cpu - out_tpu.float().to("cpu")

    import pdb;pdb.set_trace()
    print("add_diff:", torch.max(abs(add_diff)))
    print("mean_diff:", torch.max(abs(mean_diff)))
    print("rstd_diff:", torch.max(abs(rstd_diff)))
    print("out_diff:", torch.max(abs(out_diff)))

    return



if __name__ == "__main__":
    check_add_ln_matmul()

