import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

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

        ctx.save_for_backward(out_add, w, mean, rstd, gamma, beta)

        return out_add, out
    
    @staticmethod
    def backward(ctx, grad_add, grad_output):
        out_add, w, mean, rstd, gamma, beta = ctx.saved_tensors

        D = w.shape[1]
        if out_add.dim() == 3:
            B, M, N = out_add.shape
            out_ln_cpu = ((out_add.cpu() - mean.unsqueeze(-1).cpu()) * rstd.unsqueeze(-1).cpu()) * gamma.unsqueeze(0).unsqueeze(1).cpu() + beta.unsqueeze(0).unsqueeze(1).cpu()
            grad_out_ln = torch.matmul(grad_output, w.unsqueeze(0).transpose(-1,-2))
        else:
            M, N = out_add.shape
            out_ln_cpu = ((out_add.cpu() - mean.unsqueeze(-1).cpu()) * rstd.unsqueeze(-1).cpu()) * gamma.unsqueeze(0).cpu() + beta.unsqueeze(0).cpu()
            grad_out_ln = torch.matmul(grad_output, w.transpose(-1,-2))

        out_ln = out_ln_cpu.to(grad_output.device)

        grad_out_add = torch.ones(out_add.shape, dtype = out_add.dtype, device = grad_output.device)
        grad_gamma = torch.ones((N,), dtype = out_add.dtype, device = grad_output.device)
        grad_beta = torch.ones((N,), dtype = out_add.dtype, device = grad_output.device)

        grad_w = torch.matmul(out_ln.transpose(-1,-2), grad_output)
        grad_b = grad_output.reshape(-1, D).sum(0)
        
        torch.ops.my_ops.add_ln_mm_backward(grad_out_ln,
                                        out_add,
                                        mean.unsqueeze(-1),
                                        rstd.unsqueeze(-1),
                                        gamma,
                                        grad_out_add,
                                        grad_gamma,
                                        grad_beta)
        
        return grad_out_add, grad_out_add, grad_w, grad_b, grad_gamma, grad_beta



class AddlnMatmulBlock(nn.Module):
    def __init__(self, w, b, gamma, beta):
        super().__init__()
        self.w = w
        self.b = b
        self.gamma = gamma
        self.beta = beta

        self.w.retain_grad()
        self.b.retain_grad()
        self.gamma.retain_grad()
        self.beta.retain_grad()

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

    print("===============forward===============")
    if input_dim == 3:
        x1 = torch.randn(batch_size, length, embed_dim)
        x2 = torch.randn(batch_size, length, embed_dim)
    else:
        x1 = torch.randn(length, embed_dim)
        x2 = torch.randn(length, embed_dim)
    x1_tpu = x1.to(device).half()
    x2_tpu = x2.to(device).half()

    x1.requires_grad = True
    x2.requires_grad = True
    x1_tpu.requires_grad = True    # HACK
    x2_tpu.requires_grad = True

    out_add_tpu, out_tpu = net_tpu(x1_tpu, x2_tpu)
    out_add_cpu, out_cpu = net_cpu(x1, x2)

    # mean_cpu = out_add_cpu.mean(-1)
    # rstd_cpu = 1 / (out_add_cpu.var(-1, unbiased=False) + eps).sqrt()

    add_diff = out_add_cpu - out_add_tpu.float().to("cpu")
    out_diff = out_cpu - out_tpu.float().to("cpu")


    print("===============backward================")
    if input_dim == 3:
        ref_cpu = torch.ones(batch_size, length, b.shape[0])
    else:
        ref_cpu = torch.ones(length, b.shape[0])
    ref_tpu = ref_cpu.to(device)

    out_cpu.backward(ref_cpu)
    out_tpu.backward(ref_tpu)

    grad_x1_diff = x1.grad - x1_tpu.grad.float().to("cpu")
    grad_x2_diff = x2.grad - x2_tpu.grad.float().to("cpu")
    grad_w_diff = net_cpu.fc.weight.grad.transpose(-1,-2) - net_tpu.w.grad.float().to("cpu")
    grad_b_diff = net_cpu.fc.bias.grad - net_tpu.b.grad.float().to("cpu")
    grad_gamma_diff = net_cpu.ln.weight.grad - net_tpu.gamma.grad.float().to("cpu")
    grad_beta_diff = net_cpu.ln.bias.grad - net_tpu.beta.grad.float().to("cpu")

    import pdb;pdb.set_trace()
    print("add_diff:", torch.max(abs(add_diff)))
    print("out_diff:", torch.max(abs(out_diff)))
    print("grad_x1_diff:", torch.max(abs(grad_x1_diff)))
    print("grad_x2_diff:", torch.max(abs(grad_x2_diff)))
    print("grad_w_diff:", torch.max(abs(grad_w_diff)))
    print("grad_b_diff:", torch.max(abs(grad_b_diff)))
    print("grad_gamma_diff:", torch.max(abs(grad_gamma_diff)))
    print("grad_beta_diff:", torch.max(abs(grad_beta_diff)))

    return



if __name__ == "__main__":
    check_add_ln_matmul()

