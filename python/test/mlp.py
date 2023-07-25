import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import copy


torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

class Conv1D(nn.Module):
    """
    Basically works like a linear layer

    Args:
        nf (:obj:`int`): The number of output features.
        nx (:obj:`int`): The number of input features.
    """

    def __init__(self, nx, nf):
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x
    

class GPT2Mlp(nn.Module):
    def __init__(self, embed_dim, intermediate_size):
        super().__init__()
        self.c_fc = nn.Linear(embed_dim, intermediate_size)
        self.c_proj = nn.Linear(intermediate_size, embed_dim)
        self.act = F.gelu

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        return x
    

class MlpFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2, b1, b2):
        B, M, N = x.shape
        D1 = w1.shape[1]
        D2 = w2.shape[1]
        assert w1.shape == (N, D1)
        assert w2.shape == (D1, D2)
        assert b1.shape == (D1,)
        assert b2.shape == (D2,)
        out1 = torch.empty(B, M, D1).type_as(x).to(device)
        p = torch.empty(B, M, D1).type_as(x).to(device)
        out2 = torch.empty(B, M, D2).type_as(x).to(device)

        torch.ops.my_ops.mlp_forward(x,
                                     w1,
                                     w2,
                                     b1,
                                     b2,
                                     out1,
                                     p,
                                     out2)

        ctx.save_for_backward(x, w1, w2, out1, p)

        return out2

    @staticmethod
    def backward(ctx, grad_output):
        x, w1, w2, out1, p = ctx.saved_tensors

        B, M, N = x.shape
        D1 = w1.shape[1]
        D2 = w2.shape[1]
        grad_output.to(device)
        grad_x = torch.ones(x.shape, dtype = x.dtype).to(device)
        grad_w1 = torch.ones(w1.shape, dtype = x.dtype).to(device)
        grad_w2 = torch.ones(w2.shape, dtype = x.dtype).to(device)

        grad_b1 = torch.ones((D1,), dtype = x.dtype).to(device)
        grad_b2 = torch.ones((D2,), dtype = x.dtype).to(device)

        torch.ops.my_ops.mlp_backward(grad_output,
                                        x,
                                        w1,
                                        w2,
                                        out1,
                                        p,
                                        grad_x,
                                        grad_w1,
                                        grad_w2,
                                        grad_b1,
                                        grad_b2)

        return grad_x, grad_w1, grad_w2, grad_b1, grad_b2


class MlpBlock(nn.Module):
    def __init__(self, w1, w2, b1, b2):
        super().__init__()
        self.w1 = w1
        self.w2 = w2
        self.b1 = b1
        self.b2 = b2

    def forward(self, x):
        return MlpFunc.apply(x, self.w1, self.w2, self.b1, self.b2)


def check_mlp():
    batch_size = 4
    length = 10
    embed_dim = 32
    intermediate_size = 128
    net_cpu = GPT2Mlp(embed_dim, intermediate_size)

    # w1 = torch.tensor(copy.deepcopy(net_cpu.state_dict()['c_fc.weight']), requires_grad=True).to(device)
    w1 = net_cpu.state_dict()['c_fc.weight'].clone().detach().transpose(0,1).contiguous().requires_grad_(True).to(device).half()   # TODO
    b1 = net_cpu.state_dict()['c_fc.bias'].clone().detach().requires_grad_(True).to(device).half()
    w2 = net_cpu.state_dict()['c_proj.weight'].clone().detach().transpose(0,1).contiguous().requires_grad_(True).to(device).half()
    b2 = net_cpu.state_dict()['c_proj.bias'].clone().detach().requires_grad_(True).to(device).half()

    net_tpu = MlpBlock(w1, w2, b1, b2)

    print("=====forward======")
    x = torch.randn(batch_size, length, embed_dim, requires_grad=True)
    x_tpu = x.to(device).half()
    out_tpu = net_tpu(x_tpu)
    out_cpu = net_cpu(x)
    out_diff = out_cpu - out_tpu.float().to("cpu")
    print (torch.max(abs(out_diff)))
    # import pdb;pdb.set_trace()

    print("=====backward======")
    ref_tpu = torch.ones(batch_size, length, b2.shape[0]).to(device)
    ref_cpu = ref_tpu.cpu()
    out_tpu.backward(ref_tpu)
    out_cpu.backward(ref_cpu)

    compare_mlp_grad(net_cpu, net_tpu)
    compare_mlp_weight(net_cpu, net_tpu)
    return


def compare_mlp_grad(net_cpu, net_tpu):
    w1_cpu = net_cpu.c_fc.weight.grad.numpy().transpose()
    w2_cpu = net_cpu.c_proj.weight.grad.numpy().transpose()
    b1_cpu = net_cpu.c_fc.bias.grad.numpy()
    b2_cpu = net_cpu.c_proj.bias.grad.numpy()

    w1_tpu = net_tpu.w1.grad.numpy()
    w2_tpu = net_tpu.w2.grad.numpy()
    b1_tpu = net_tpu.b1.grad.numpy()
    b2_tpu = net_tpu.b2.grad.numpy()

    grad_name = ['w1', 'w2', 'b1', 'b2']
    cpu_grad = {
        'w1':w1_cpu,
        'w2':w2_cpu,
        'b1':b1_cpu,
        'b2':b2_cpu,
    }
    tpu_grad = {
        'w1':w1_tpu,
        'w2':w2_tpu,
        'b1':b1_tpu,
        'b2':b2_tpu,
    }

    for k in grad_name:
        c_g = cpu_grad[k]
        t_g = tpu_grad[k]
        diff = abs(c_g - t_g)
        # index_abs = diff.argmax()
        related_diff = abs(diff/c_g)
        # index_related = related_diff.argmax()
        print(k, 
                ",max abs diff: ", np.max(diff),
                ",max rel diff: ", np.max(related_diff)
            )

    return


def compare_mlp_weight(net_cpu, net_tpu):
    w1_cpu = net_cpu.c_fc.weight.numpy().transpose()
    w2_cpu = net_cpu.c_proj.weight.numpy().transpose()
    b1_cpu = net_cpu.c_fc.bias.numpy()
    b2_cpu = net_cpu.c_proj.bias.numpy()

    w1_tpu = net_tpu.w1.numpy()
    w2_tpu = net_tpu.w2.numpy()
    b1_tpu = net_tpu.b1.numpy()
    b2_tpu = net_tpu.b2.numpy()

    grad_name = ['w1', 'w2', 'b1', 'b2']
    cpu_weight = {
        'w1':w1_cpu,
        'w2':w2_cpu,
        'b1':b1_cpu,
        'b2':b2_cpu,
    }
    tpu_weight = {
        'w1':w1_tpu,
        'w2':w2_tpu,
        'b1':b1_tpu,
        'b2':b2_tpu,
    }

    for k in grad_name:
        c_g = cpu_weight[k]
        t_g = tpu_weight[k]
        diff = abs(c_g - t_g)
        # index_abs = diff.argmax()
        related_diff = abs(diff/c_g)
        # index_related = related_diff.argmax()
        print(k, 
                ",max abs diff: ", np.max(diff),
                ",max rel diff: ", np.max(related_diff)
            )
    return


# def test():
#     batch_size = 4
#     length = 10
#     embed_dim = 32
#     intermediate_size = 128
#     net_cpu = GPT2Mlp(embed_dim, intermediate_size)
#     w1 = net_cpu.state_dict()['c_fc.weight'].clone().detach()
#     b2 = net_cpu.state_dict()['c_proj.bias'].clone().detach()
#     x = torch.randn(batch_size, length, embed_dim, requires_grad=True)
#     out_cpu = net_cpu(x)

#     ref_cpu = torch.ones(batch_size, length, b2.shape[0])
#     out_cpu.backward(ref_cpu)

#     return
    


if __name__ == "__main__":
    check_mlp()
    # test()
