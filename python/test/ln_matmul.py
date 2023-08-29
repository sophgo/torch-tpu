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


class GPT2lnMatmul(nn.Module):
    def __init__(self, embed_dim, intermediate_size, layer_norm_epsilon):
        super().__init__()
        self.ln = nn.LayerNorm(embed_dim, eps=layer_norm_epsilon)
        self.fc = nn.Linear(embed_dim, intermediate_size)

    def forward(self, x):
        x = self.ln(x)
        x = self.fc(x)
        return x


class lnMatmulFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, b, gamma, beta):
        D = w.shape[1]
        if x.dim() == 3:
            B, M, N = x.shape
            
            mean = torch.empty((B, M), dtype = x.dtype, device = x.device)
            rstd = torch.empty((B, M), dtype = x.dtype, device = x.device)
            out = torch.empty((B, M, D), dtype = x.dtype, device = x.device)
        else:
            M, N = x.shape
            
            mean = torch.empty((M,), dtype = x.dtype, device = x.device)
            rstd = torch.empty((M,), dtype = x.dtype, device = x.device)
            out = torch.empty((M, D), dtype = x.dtype, device = x.device)

        assert w.shape == (N, D)
        assert gamma.shape == (N,)
        assert beta.shape == (N,)
        if b != None:
            assert b.shape == (D,)

        torch.ops.my_ops.ln_mm_forward(x,
                                     w,
                                     b,
                                     gamma,
                                     beta,
                                     eps, 
                                     mean,
                                     rstd,
                                     out)

        ctx.save_for_backward(x, w, mean, rstd)

        return mean, rstd, out 


class lnMatmulBlock(nn.Module):
    def __init__(self, w, b, gamma, beta):
        super().__init__()
        self.w = w
        self.b = b
        self.gamma = gamma
        self.beta = beta

    def forward(self, x):
        return lnMatmulFunc.apply(x, self.w, self.b, self.gamma, self.beta)
    


def check_ln_matmul():
    net_cpu = GPT2lnMatmul(embed_dim, intermediate_size, eps)

    gamma = net_cpu.state_dict()['ln.weight'].clone().detach().contiguous().requires_grad_(True).to(device).half()
    beta = None
    if 'ln.bias' in net_cpu.state_dict().keys():
        beta = net_cpu.state_dict()['ln.bias'].clone().detach().requires_grad_(True).to(device).half()
   
    w = net_cpu.state_dict()['fc.weight'].clone().detach().transpose(0,1).contiguous().requires_grad_(True).to(device).half()
    b = None
    if 'fc.bias' in net_cpu.state_dict().keys():
        b = net_cpu.state_dict()['fc.bias'].clone().detach().requires_grad_(True).to(device).half()
    
    net_tpu = lnMatmulBlock(w, b, gamma, beta)

    print("=====forward======")
    if input_dim == 3:
        x = torch.randn(batch_size, length, embed_dim, requires_grad=True)
    else:
        x = torch.randn(length, embed_dim, requires_grad=True)
    x_tpu = x.to(device).half()
    mean_tpu, rstd_tpu, out_tpu = net_tpu(x_tpu)
    out_cpu = net_cpu(x)

    mean_cpu = x.mean(-1)
    rstd_cpu = x.std(-1)

    mean_diff = mean_cpu - mean_tpu.float().to("cpu")
    rstd_diff = rstd_cpu - rstd_tpu.float().to("cpu")
    out_diff = out_cpu - out_tpu.float().to("cpu")

    import pdb;pdb.set_trace()
    print("mean_diff:", torch.max(abs(mean_diff)))
    print("rstd_diff:", torch.max(abs(rstd_diff)))
    print("out_diff:", torch.max(abs(out_diff)))

    return



if __name__ == "__main__":
    check_ln_matmul()

