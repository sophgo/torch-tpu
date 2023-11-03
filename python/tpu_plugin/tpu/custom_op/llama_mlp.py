import os
import torch
import torch.nn as nn

torch.ops.load_library("../../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

class LLamaMlpFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w0, w1, w2):
        output = torch.empty(x.shape, dtype = x.dtype, device = x.device)

        torch.ops.my_ops.llama_mlp_forward(x,
                                     w0,
                                     w1,
                                     w2,
                                     output)
        return output

class LLamaMlpBlock(nn.Module):
    def __init__(self, embed_dim, intermediate_size):
        super().__init__()
        w0 = torch.empty(embed_dim, intermediate_size)
        w1 = torch.empty(embed_dim, intermediate_size)
        w2 = torch.empty(intermediate_size, embed_dim)
        nn.init.normal_(w0, std=0.02)
        nn.init.normal_(w1, std=0.02)
        nn.init.normal_(w2, std=0.02)
        self.w0 = nn.Parameter(w0)
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)

    def forward(self, x):
        return LLamaMlpFunc.apply(x, self.w0, self.w1, self.w2)
