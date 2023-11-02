import os
import torch
import torch.nn as nn

torch.ops.load_library("../../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

class LLamaAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, Kcache, Vcache, w1, w2, w3, C):
        Y = torch.empty(Q.shape, dtype = Q.dtype, device = Q.device)

        torch.ops.my_ops.llama_attention(Q,
                                     K,
                                     V,
                                     Kcache,
                                     Vcache,
                                     w1,
                                     w2,
                                     w3,
                                     Y,
                                     C)
        return Y

class LLamaAttentionBlock(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, embeddings, C):
        super().__init__()
        d = int(hidden_size / num_attention_heads)
        w1 = torch.empty(d)
        w2 = torch.empty(d)
        w3 = torch.empty(embeddings)
        nn.init.normal_(w1, std=0.02)
        nn.init.normal_(w2, std=0.02)
        nn.init.normal_(w3, std=0.02)
        self.w1 = nn.Parameter(w1)
        self.w2 = nn.Parameter(w2)
        self.w3 = nn.Parameter(w3)
        self.C = C

    def forward(self, Q, K, V, Kcache, Vcache):
        return LLamaAttentionFunc.apply(Q, K, V, Kcache, Vcache, self.w1, self.w2, self.w3, self.C)
