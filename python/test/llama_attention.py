import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import copy


torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"
tpu_device = "privateuseone:0"
torch.set_printoptions(profile="full")

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
    
    def embedding(self, x, cos, sin, mask_coeff):
        x_temp = torch.concat((x[..., 64:], x[..., :64]), dim=-1)
        x = x * cos + x_temp * mask_coeff * sin
        return x

    def forward(self, Q, K, V, Kcache, Vcache):
        mask_coeff = torch.ones(128)
        mask_coeff[:64] *= -1.
        K = self.embedding(K, self.w1, self.w2, mask_coeff)
        Q = self.embedding(Q, self.w1, self.w2, mask_coeff)
        # K = K * self.w1 * mask_coeff + K * self.w2
        # Q = Q * self.w1 * mask_coeff + Q * self.w2
        KConcat = torch.concat([torch.permute(Kcache, (0,1,3,2)),torch.permute(K, (0,1,3,2))], dim=3)
        VConcat = torch.concat([Vcache, V], dim=2)
        KConcat = KConcat.unsqueeze(2)
        VConcat = VConcat.unsqueeze(2)
        KConcat = KConcat.repeat((1, 1, int(self.num_attention_heads / self.kv_heads), 1, 1))
        VConcat = VConcat.repeat((1, 1, int(self.num_attention_heads / self.kv_heads), 1, 1))
        KConcat = torch.flatten(KConcat, start_dim=1, end_dim=2)
        VConcat = torch.flatten(VConcat, start_dim=1, end_dim=2)
        print(f"CPU SHAPE: {Q.shape, KConcat.shape, self.w3.shape}")
        res_qk = torch.matmul(Q, KConcat) * self.C + self.w3
        res_qk = F.softmax(res_qk, dim=3)
        Y = torch.matmul(res_qk, VConcat)
        return Y, KConcat[:, 0].transpose(1, 2), VConcat[:, 0].transpose(1, 2)


class LLamaAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, Kcache, Vcache, w1, w2, w3, embeddings, C):
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
                                    embeddings,
                                    C)
        return output, Kcache, Vcache

class LLamaAttentionBlock(nn.Module):
    def __init__(self, w0, w1, w2, C):
        super().__init__()
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.C = C

    def forward(self, Q, K, V, Kcache, Vcache, embeddings):
        print(f"shape: {Q.shape, K.shape, V.shape}")
        print(f"{Kcache.shape, Vcache.shape}")
        return LLamaAttentionFunc.apply(Q, K, V, Kcache, Vcache, self.w0, self.w1, self.w2, embeddings, self.C)


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
    Q = torch.randn((batch, attention_heads, 1, d), requires_grad=False, dtype=torch.float16)
    K = torch.randn((batch, k_v_heads, 1, d), requires_grad=False, dtype=torch.float16)
    V = torch.randn((batch, k_v_heads, 1, d), requires_grad=False, dtype=torch.float16)
    Kcache = torch.randn((batch, k_v_heads, embeddings-1, d), requires_grad=False, dtype=torch.float16)
    Vcache = torch.randn((batch, k_v_heads, embeddings-1, d), requires_grad=False, dtype=torch.float16)

    out_cpu, Kcache, Vcache = net_cpu(Q, K, V, Kcache, Vcache)

    Q_tpu = Q.to(device).half()
    K_tpu = K.to(device).half()
    V_tpu = V.to(device).half()

    Kcache_tpu = torch.empty((batch, k_v_heads, embeddings,  d), dtype=torch.float16)
    Vcache_tpu = torch.empty((batch, k_v_heads, embeddings,  d), dtype=torch.float16)
    Kcache_tpu[:, :, :-1] = Kcache
    Vcache_tpu[:, :, :-1] = Vcache
    Kcache_tpu = Kcache_tpu.to(device)
    Vcache_tpu = Vcache_tpu.to(device)

    out_tpu, Kcache_tpu, Vcache_tpu = net_tpu(Q_tpu, K_tpu, V_tpu, Kcache_tpu, Vcache_tpu, embeddings)
    out_tpu = out_tpu.float().to("cpu")
    Kcache_tpu = Kcache_tpu.squeeze().cpu()
    Vcache_tpu = Vcache_tpu.squeeze().cpu()

    
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
    print(torch.where(abs(out_diff) > 0.01))

    print("kv cache")
    k_cache_diff = abs(Kcache - Kcache_tpu)
    print(torch.where(k_cache_diff > 0.01))
    print(f"k cache diff max: {torch.max(k_cache_diff)}")


    return



def debug(q, k_cache, v_cache, softmax_scale, slots):
    # import pdb; pdb.set_trace()
    res_qk_71 = torch.matmul(q[:, 6].to(torch.float32), k_cache[:, 0, :, :slots+1].to(torch.float32)) * softmax_scale
    res_qk_71 = torch.nn.functional.softmax(res_qk_71, dim=2)
    out_711 = torch.matmul(res_qk_71, v_cache[:, 0, :slots+1].to(torch.float32))
    out_712 = torch.matmul(res_qk_71, v_cache[:, 1, :slots+1].to(torch.float32))
    out_713 = torch.matmul(res_qk_71, v_cache[:, 2, :slots+1].to(torch.float32))

    res_qk_72 = torch.matmul(q[:, 6].to(torch.float32), k_cache[:, 1, :, :slots+1].to(torch.float32)) * softmax_scale
    res_qk_72 = torch.nn.functional.softmax(res_qk_72, dim=2)
    out_721 = torch.matmul(res_qk_72, v_cache[:, 0, :slots+1].to(torch.float32))
    out_722 = torch.matmul(res_qk_72, v_cache[:, 1, :slots+1].to(torch.float32))
    out_723 = torch.matmul(res_qk_72, v_cache[:, 2, :slots+1].to(torch.float32))

def CPULlamaAttentionFun(q, k, v, k_cache, v_cache, cos, sin, mask, softmax_scale, slots):
        def rotary_embedding(x, cos, sin, mask_coeff):
            x_temp = torch.concat((x[..., 64:], x[..., :64]), dim=-1)
            x = x * cos + x_temp * mask_coeff * sin
            return x
        
        mask_coeff = torch.ones(128)
        mask_coeff[:64] *= -1.
        k = rotary_embedding(k, cos, sin, mask_coeff)
        q = rotary_embedding(q, cos, sin, mask_coeff)
        k_cache[:, :, slots] = k[:, :, 0]
        v_cache[:, :, slots] = v[:, :, 0]

        # save_dict = {"k_cache_cat":k_cache, "v_cache_cat":v_cache}
        # import os
        # path = "/usr/src/server/attention/"
        # os.makedirs(path, exist_ok=True)
        # torch.save(save_dict, path + f"kv_cache_cat.pth")

        k_cache = torch.permute(k_cache, (0, 1, 3, 2))

        q_head = q.size(1)
        kv_head = k.size(1)

        # debug(q, k_cache, v_cache, softmax_scale, slots)

        k_cache = k_cache.unsqueeze(2)
        v_cache = v_cache.unsqueeze(2)
        k_cache = k_cache.repeat((1, 1, int(q_head / kv_head), 1, 1))
        v_cache = v_cache.repeat((1, 1, int(q_head / kv_head), 1, 1))
        k_cache = torch.flatten(k_cache, start_dim=1, end_dim=2)
        v_cache = torch.flatten(v_cache, start_dim=1, end_dim=2)
        # import pdb; pdb.set_trace()

        # mask = torch.zeros_like(mask, dtype=torch.float32)
        # mask[slots+1:] = torch.tensor(-float('inf'), dtype=torch.float32)
        # res_qk = torch.matmul(q.to(torch.float32), k_cache.to(torch.float32))* softmax_scale + mask
        # res_qk[..., slots+1:] = torch.tensor(-float('inf'))
        # Y = torch.matmul(res_qk, v_cache.to(torch.float32))

        res_qk = torch.matmul(q.to(torch.float32), k_cache[..., :slots+1].to(torch.float32)) * softmax_scale
        res_qk = torch.nn.functional.softmax(res_qk, dim=3)
        Y = torch.matmul(res_qk, v_cache[:, :, :slots+1].to(torch.float32))
        return Y.squeeze_(2)


def case3():
    parameter = torch.load("/workspace/model.layers.0.self_attn.pth")
    q = parameter['q']
    k = parameter['k']
    v = parameter['v']
    k_cache_soph = parameter['k_cache']
    v_cache_soph = parameter['v_cache']
    cos = parameter['cos']
    sin = parameter['sin']
    mask = parameter['mask']
    embedding = parameter['embedding']
    softmax_scale = parameter['softmax_scale']
    embedding = 8

    k_cache_cpu = k_cache_soph.clone().detach()
    v_cache_cpu = v_cache_soph.clone().detach()

    attn_output = torch.empty_like(q, dtype=torch.float16).to(tpu_device)
    k_cache_tpu = k_cache_soph[:, :, :embedding].clone().detach().to(tpu_device)
    v_cache_tpu = v_cache_soph[:, :, :embedding].clone().detach().to(tpu_device)
    cos_tpu = cos.to(tpu_device)
    sin_tpu = sin.to(tpu_device)
    q_tpu = q.to(tpu_device)
    k_tpu = k.to(tpu_device)
    v_tpu = v.to(tpu_device)
    mask_tpu = torch.zeros((embedding,), dtype=torch.float16).to(tpu_device)

    # save_dict = {"q":q, "k":k, "v":v, "k_cache":k_cache_cpu, "v_cache":v_cache_cpu, "cos":cos, "sin":sin, "embedding":embedding, "softmax_scale":softmax_scale}
    # attn_output_cpu = CPULlamaAttentionFun(q, k, v, k_cache_cpu, v_cache_cpu, cos, sin, mask, softmax_scale, embedding-1)
    # save_dict['attn_output'] = attn_output_cpu

    # import os
    # # path = "/worksapce/tpu-train/"
    # # os.makedirs(path, exist_ok=True)
    # torch.save(save_dict, "cpu_sw.pth")
    # print("save dict")

    # import pdb; pdb.set_trace()

    Ycpu = np.loadtxt("/workspace/Ytpu_teng.txt")
    Ycpu_tensor = torch.from_numpy(Ycpu).reshape((1, 64, 128))
    # print(f"{Ycpu_tensor.shape, attn_output_cpu.shape}")
    # print(f"cpu diff: {torch.max(abs(Ycpu_tensor - attn_output_cpu))}")
    # print(f"diff where: {torch.where(abs(Ycpu_tensor - attn_output_cpu) > 1)[1]}")

    # import pdb; pdb.set_trace()

    # print(f"q : {torch.max(abs(q - q_tpu.cpu()))}")
    # print(f"k : {torch.max(abs(k - k_tpu.cpu()))}")
    # print(f"v : {torch.max(abs(v - v_tpu.cpu()))}")
    # # print(f"k_cache_soph : {torch.max(abs(k_cache_soph - k_cache_tpu.cpu()))}")
    # # print(f"v_cache_soph : {torch.max(abs(v_cache_soph - v_cache_tpu.cpu()))}")
    # print(f"cos : {torch.max(abs(cos - cos_tpu.cpu()))}")
    # print(f"sin : {torch.max(abs(sin - sin_tpu.cpu()))}")
    # print(f"shape: {k_cache_tpu.shape, v_cache_tpu.shape}")

    torch.ops.my_ops.llama_attention(q_tpu, k_tpu, v_tpu, 
                                    k_cache_tpu, v_cache_tpu, 
                                    cos_tpu, sin_tpu, mask_tpu, 
                                    attn_output, embedding, 0, softmax_scale)

    print(attn_output.shape, attn_output.device)
    attn_output = attn_output.cpu().squeeze_(2)

    # print(k_cache_tpu.shape)
    # print(attn_output[0, :, :2])
    # print(attn_output_cpu[0, :, :2])
    # print(f"cpu tpu diff: {torch.max(abs(attn_output_cpu - attn_output.cpu()))}")
    # print(f"diff where: {torch.where(abs(attn_output_cpu - attn_output.cpu()) > .1)}")
    print(f"cpu tpu diff: {torch.max(abs(Ycpu_tensor - attn_output.cpu()))}")
    print(f"diff where: {torch.where(abs(Ycpu_tensor - attn_output.cpu()) > .1)}")

def add_one(x):
    x = x + torch.ones(1).to(tpu_device)
    print(x.cpu())

def test_temp():
    x = torch.zeros((3, 4)).to(tpu_device)
    print(x.cpu())
    add_one(x)
    print(x.cpu())


if __name__ == "__main__":
    # check_mlp()
    case2()
    # case3()
    # test_temp()
