import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import copy


torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
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

        ctx.save_for_backward(x, w, mean, rstd, gamma, beta)

        return out 

    @staticmethod
    def backward(ctx, grad_output):
        x, w, mean, rstd, gamma, beta = ctx.saved_tensors
        D = w.shape[1]
        if x.dim() == 3:
            B, M, N = x.shape
            out_ln_cpu = ((x.cpu() - mean.unsqueeze(-1).cpu()) * rstd.unsqueeze(-1).cpu()) * gamma.unsqueeze(0).unsqueeze(1).cpu() + beta.unsqueeze(0).unsqueeze(1).cpu()
            grad_out_ln = torch.matmul(grad_output, w.unsqueeze(0).transpose(-1,-2))
        else:
            M, N = x.shape
            out_ln_cpu = ((x.cpu() - mean.unsqueeze(-1).cpu()) * rstd.unsqueeze(-1).cpu()) * gamma.unsqueeze(0).cpu() + beta.unsqueeze(0).cpu()
            grad_out_ln = torch.matmul(grad_output, w.transpose(-1,-2))

        out_ln = out_ln_cpu.to(grad_output.device)

        grad_x = torch.ones(x.shape, dtype = x.dtype, device = grad_output.device)
        grad_gamma = torch.ones((N,), dtype = x.dtype, device = grad_output.device)
        grad_beta = torch.ones((N,), dtype = x.dtype, device = grad_output.device)

        grad_w = torch.matmul(out_ln.transpose(-1,-2), grad_output)
        grad_b = grad_output.reshape(-1, D).sum(0)
        
        torch.ops.my_ops.ln_mm_backward(grad_out_ln,
                                        x,
                                        mean.unsqueeze(-1),
                                        rstd.unsqueeze(-1),
                                        gamma,
                                        grad_x,
                                        grad_gamma,
                                        grad_beta)
        
        return grad_x, grad_w, grad_b, grad_gamma, grad_beta


class lnMatmulBlock(nn.Module):
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
    # net_tpu_old = copy.deepcopy(net_cpu).to(device).half()

    print("===============forward================")
    if input_dim == 3:
        x = torch.randn(batch_size, length, embed_dim)
    else:
        x = torch.randn(length, embed_dim)
    x_tpu = x.to(device).half()
    # x_tpu_old = copy.deepcopy(x).to(device).half()

    x.requires_grad = True    # HACK
    x_tpu.requires_grad = True
    # x_tpu_old.requires_grad = True

    out_cpu = net_cpu(x)
    t1_forward = time.time()
    out_tpu = net_tpu(x_tpu)
    t2_forward = time.time()

    # t1_forward_old = time.time()
    # out_tpu_old = net_tpu_old(x_tpu_old)
    # t2_forward_old = time.time()

    out_diff = out_cpu - out_tpu.float().to("cpu")

    print("===============backward================")
    if input_dim == 3:
        ref_cpu = torch.ones(batch_size, length, b.shape[0])
    else:
        ref_cpu = torch.ones(length, b.shape[0])
    ref_tpu = ref_cpu.to(device)
    # ref_tpu_old = copy.deepcopy(ref_cpu).to(device)

    out_cpu.backward(ref_cpu)

    t1_backward = time.time()
    out_tpu.backward(ref_tpu)
    t2_backward = time.time()

    # t1_backward_old = time.time()
    # out_tpu_old.backward(ref_tpu_old)
    # t2_backward_old = time.time()

    grad_x_diff = x.grad - x_tpu.grad.float().to("cpu")
    grad_w_diff = net_cpu.fc.weight.grad.transpose(-1,-2) - net_tpu.w.grad.float().to("cpu")
    grad_b_diff = net_cpu.fc.bias.grad - net_tpu.b.grad.float().to("cpu")
    grad_gamma_diff = net_cpu.ln.weight.grad - net_tpu.gamma.grad.float().to("cpu")
    grad_beta_diff = net_cpu.ln.bias.grad - net_tpu.beta.grad.float().to("cpu")

    import pdb;pdb.set_trace()
    print("out_diff:", torch.max(abs(out_diff)))
    print("grad_x_diff:", torch.max(abs(grad_x_diff)))
    print("grad_w_diff:", torch.max(abs(grad_w_diff)))
    print("grad_b_diff:", torch.max(abs(grad_b_diff)))
    print("grad_gamma_diff:", torch.max(abs(grad_gamma_diff)))
    print("grad_beta_diff:", torch.max(abs(grad_beta_diff)))
    # print("forward time:", t2_forward - t1_forward)
    # print("forward time old:", t2_forward_old - t1_forward_old)
    # print("backward time:", t2_backward - t1_backward)
    # print("backward time old:", t2_backward_old - t1_backward_old)

    return



if __name__ == "__main__":
    check_ln_matmul()

