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

class LLamaAttention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, embeddings, C):
        super().__init__()
        self.C = C
        self.num_attention_heads = num_attention_heads
        self.embeddings = embeddings
        self.d = int(hidden_size / num_attention_heads)
        self.kv_heads = int(num_attention_heads / 8)
        self.w1 = nn.Parameter(torch.rand(self.d))
        self.w2 = nn.Parameter(torch.rand(self.d))
        self.w3 = nn.Parameter(torch.rand(self.embeddings))

    def forward(self, Q, K, V, Kcache, Vcache):
        mask_coeff = torch.ones(self.d)
        mask_coeff[64:] *= -1.
        K = K * self.w1 * mask_coeff + K * self.w2
        Q = Q * self.w1 * mask_coeff + Q * self.w2
        KConcat = torch.concat([torch.permute(Kcache, (0,1,3,2)),torch.permute(K, (0,1,3,2))], dim=3)
        VConcat = torch.concat([Vcache, V], dim=2)
        KConcat = KConcat.unsqueeze(2)
        VConcat = VConcat.unsqueeze(2)
        KConcat = KConcat.repeat((1, 1, int(self.num_attention_heads / self.kv_heads), 1, 1))
        VConcat = VConcat.repeat((1, 1, int(self.num_attention_heads / self.kv_heads), 1, 1))
        KConcat = torch.flatten(KConcat, start_dim=1, end_dim=2)
        VConcat = torch.flatten(VConcat, start_dim=1, end_dim=2)
        res_qk = torch.matmul(Q, KConcat) * self.C + self.w3
        res_qk = F.softmax(res_qk, dim=3)
        Y = torch.matmul(res_qk, VConcat)
        return Y


class LLamaAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, Kcache, Vcache, w1, w2, w3, C):
        output = torch.empty(Q.shape, dtype = Q.dtype, device = Q.device)

        torch.ops.my_ops.llama_attention(Q,
                                    K,
                                    V,
                                    Kcache,
                                    Vcache,
                                    w1,
                                    w2,
                                    w3,
                                    output,
                                    C)
        return output

class LLamaAttentionBlock(nn.Module):
    def __init__(self, w0, w1, w2, C):
        super().__init__()
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.C = C

    def forward(self, Q, K, V, Kcache, Vcache):
        return LLamaAttentionFunc.apply(Q, K, V, Kcache, Vcache, self.w0, self.w1, self.w2, self.C)


def check_mlp():
    #  TP8: attention_heads 8, hidden_size 1024
    #  TP4: attention_heads 16, hidden_size 2048
    #  TP8: attention_heads 64, hidden_size 8192

    batch = 16
    # # TP8
    attention_heads = 8
    hidden_size = 1024
    # TP4
    # attention_heads = 16
    # hidden_size = 2048
    # # TP1
    # attention_heads = 64
    # hidden_size = 8192

    embeddings = 4096
    C = 1. / np.sqrt(128)  # sqrt(128)
    d = int(hidden_size/ attention_heads)  # 128
    assert d == 128
    k_v_heads = int(attention_heads / 8)
    net_cpu = LLamaAttention(hidden_size, attention_heads, embeddings, C)

    w0 = net_cpu.state_dict()['w1'].clone().detach().contiguous().requires_grad_(False).to(device).half()
    w1 = net_cpu.state_dict()['w2'].clone().detach().contiguous().requires_grad_(False).to(device).half()
    w2 = net_cpu.state_dict()['w3'].clone().detach().contiguous().requires_grad_(False).to(device).half()

    net_tpu = LLamaAttentionBlock(w0, w1, w2, C)

    print("=====forward======")
    Q = torch.randn((batch, attention_heads, 1, d), requires_grad=False)
    K = torch.randn((batch, k_v_heads, 1, d), requires_grad=False)
    V = torch.randn((batch, k_v_heads, 1, d), requires_grad=False)
    Kcache = torch.randn((batch, k_v_heads, embeddings - 1, d), requires_grad=False)
    Vcache = torch.randn((batch, k_v_heads, embeddings - 1, d), requires_grad=False)
    Q_tpu = Q.to(device).half()
    K_tpu = K.to(device).half()
    V_tpu = V.to(device).half()
    Kcache_tpu = Kcache.to(device).half()
    Vcache_tpu = Vcache.to(device).half()

    out_tpu = net_tpu(Q_tpu, K_tpu, V_tpu, Kcache_tpu, Vcache_tpu)
    out_tpu = out_tpu.float().to("cpu")
    out_cpu = net_cpu(Q, K, V, Kcache, Vcache)
    out_cpu_f = out_cpu.flatten()
    out_tpu_f = out_tpu.flatten()
    out_diff = out_cpu_f - out_tpu_f
    print('-'*40 + " onnx_out " + '-'*40)
    print(out_cpu_f[128:128+50])
    print('-'*40 + " tpu_out " + '-'*40)
    print(out_tpu_f[128:128+50])
    # print(out_cpu.shape)
    print('-'*40 + " max diff " + '-'*40)
    print (torch.max(abs(out_diff)))
    print('-'*40 + " diff > 0.01 position " + '-'*40)
    print(torch.where(out_diff > 0.01))
    return


if __name__ == "__main__":
    check_mlp()
