import random
import torch
import torch.nn as nn
from typing import Optional, Callable
import torch.nn.functional as F
import torch_tpu

MOE_CONFIG_16B = {
    "dim": 2048,
    "moe_inter_dim": 1408,
    "n_routed_experts": 64,
    "n_activated_experts": 6,
}

MOE_CONFIG_236B = {
    "dim": 5120,
    "moe_inter_dim": 1536,
    "n_routed_experts": 160,
    "n_activated_experts": 6,
}

MOE_CONFIG_671B = {
    "dim": 7168,
    "moe_inter_dim": 2048,
    "n_routed_experts": 256,
    "n_activated_experts": 8,
}

SELECTED_EXPERTS_B8_E256 = torch.tensor([
    [206,  15, 213,   8, 163, 254, 242, 192],
    [236, 160,  46, 132,  43, 175, 147,  61],
    [ 35,  86,  60, 169,  56,  69,  36, 107],
    [ 90, 178,  37, 123, 170, 112,  73,  61],
    [ 52, 107, 102, 208,  42, 201, 203,  57],
    [203, 100,  45,  84,  42, 119, 194, 204],
    [107,  26,   2, 124, 109,  68,  86, 140],
    [140, 160, 216, 208,  90, 147, 171, 223]], dtype=torch.int32)

SELECTED_EXPERTS_B22_E256 = torch.tensor([
    [160,  30, 242,  13,  57, 236,  75,   7],
    [ 83, 160, 242, 216, 115,   7,  57, 223],
    [213, 238,  57, 208, 104,   7, 240, 227],
    [242, 208, 131,  21, 160, 240,  57, 113],
    [208,  31, 221,   7, 160, 211, 209, 230],
    [ 41,  30, 110,  31, 221,  53, 167, 109],
    [242, 131,  48,  57, 154, 208,  21, 247],
    [ 48, 221, 154, 208,  31,   7, 160,  13],
    [187,  61, 168, 155, 223,  82, 125,  57],
    [123, 233, 228, 140,  49, 154, 120, 209],
    [ 83, 242,   7, 223, 115,  13, 206, 158],
    [158, 206, 134, 242, 223, 157, 133, 243],
    [177, 109, 133, 119, 155, 253, 186,  75],
    [146,  88, 160, 223, 109, 249, 128, 236],
    [ 69,  86, 169, 252, 107,  56,   7,  41],
    [178,  37, 170,  61,  46, 112,  53, 252],
    [107, 208,  42, 216,  92, 102, 201,  57],
    [100, 203,  84, 194,  42, 119,  90, 204],
    [ 26,   2, 124, 107,  68, 169, 109,  69],
    [ 16, 174,  22, 192, 128, 130,  13,  11],
    [253, 139,  35,   8,  80, 125,  36,  42],
    [  4,  45, 109,  56, 146, 177,   7,  90]], dtype=torch.int32)

class QwenMlp(nn.Module):
    def __init__(self, w0, w1, w2):
        super().__init__()
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2

    def forward(self, x):
        # self.act_fn(self.gate_proj(x))
        out00 = F.linear(x, self.w0)
        out0 = F.silu(out00)
        # self.up_proj(x)
        out1 = F.linear(x, self.w1)
        # self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        out2 = out0 * out1
        # self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        out3 = F.linear(out2, self.w2)

        return out3
    
class QwenMoE(nn.Module):
    def __init__(self, num_experts, num_experts_per_tok, w0, w1, w2):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.experts = nn.ModuleList([QwenMlp(w0[i], w1[i], w2[i]) for i in range(num_experts)])

    def forward(self, x, selected_experts):
        out = torch.zeros(x.shape[0], self.num_experts_per_tok, x.shape[1]).to(x.dtype).to(x.device)
        for i in range(selected_experts.shape[0]):
            for j in range(selected_experts.shape[1]):
                out[i, j, :] = self.experts[selected_experts[i, j]](x[i])
        return out

class QwenMoEFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w0, w1, w2, 
                num_experts, num_experts_per_tok, blocksize, selected_experts, routing_weights):
        output_sample = None
        input_sample = None
        num_select_experts = None
        select_experts_middle = None
        routing_weights_middle = None
        use_grouped_topk = False
        num_expert_group = 1
        topk_group = 1
        # gathered_experts_out_buf is of shape (seq x k x hs)
        gathered_experts_out_buf = torch.empty(x.shape[0], num_experts_per_tok, x.shape[1], dtype = x.dtype, device = x.device)
        #output = torch.empty(x.shape, dtype = x.dtype, device = x.device)
        print(f"selected_experts shape: {selected_experts.shape}, selected_experts: {selected_experts}")
        torch.ops.my_ops.fused_moe_fused_experts(gathered_experts_out_buf, x,
                                                 output_sample, input_sample,
                                                 w0, w1, w2,
                                                 None, None, None,
                                                 selected_experts, routing_weights,
                                                 num_select_experts,
                                                 select_experts_middle, routing_weights_middle,
                                                 blocksize, num_experts, num_experts_per_tok,
                                                 use_grouped_topk, num_expert_group, topk_group,
                                                 None, None, None, False)
        return gathered_experts_out_buf

class QwenMoEBlock(nn.Module):
    def __init__(self, num_experts, num_experts_per_tok, w0, w1, w2, blocksize):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.blocksize = blocksize

    def forward(self, x, selected_experts, routing_weights):
        return QwenMoEFunc.apply(x, self.w0, self.w1, self.w2,
                                     self.num_experts, self.num_experts_per_tok,
                                     self.blocksize, selected_experts, routing_weights)

def cal_diff(x: torch.Tensor, y: torch.Tensor, name: str) -> None:
    x, y = x.double(), y.double()
    RMSE = ((x - y) * (x - y)).mean().sqrt().item()
    cos_diff = 1 - 2 * (x * y).sum().item() / max((x * x + y * y).sum().item(), 1e-12)
    amax_diff = (x - y).abs().max().item()
    #print(f"{name}: {cos_diff=}, {RMSE=}, {amax_diff=}")
    #assert cos_diff < 1e-5
    return cos_diff, RMSE, amax_diff

def test_Qwen_moe(batch_size, input_w, middle_w, num_experts, num_experts_per_tok):
    assert num_experts_per_tok <= num_experts
    blocksize = 128
    selected_experts_list = []
    for i in range(batch_size):
        selected_experts_list.append(random.sample(range(num_experts), num_experts_per_tok))
    selected_experts = torch.tensor(selected_experts_list, dtype=torch.int32).reshape(batch_size, num_experts_per_tok)

    if num_experts == 256 and batch_size == 8:
        selected_experts = SELECTED_EXPERTS_B8_E256
    elif num_experts == 256 and batch_size == 22:
        selected_experts = SELECTED_EXPERTS_B22_E256

    routing_weights = torch.randn(batch_size, num_experts_per_tok, dtype=torch.bfloat16)

    w0 = torch.randn(num_experts, middle_w, input_w).to(torch.bfloat16)
    w1 = torch.randn(num_experts, middle_w, input_w).to(torch.bfloat16)
    w2 = torch.randn(num_experts, input_w, middle_w).to(torch.bfloat16)

    net_cpu = QwenMoE(num_experts, num_experts_per_tok, w0, w1, w2)
    x = torch.randn(batch_size, input_w)
    out_cpu = net_cpu(x, selected_experts)

    device = "tpu:0"
    x_tpu = x.to(device)
    w0_tpu = w0.to(device)
    w1_tpu = w1.to(device)
    w2_tpu = w2.transpose(1, 2).to(device) # transpose for TPU

    selected_experts_tpu = selected_experts.to(device)
    routing_weights_tpu = routing_weights.to(device)
    
    net_tpu = QwenMoEBlock(num_experts, num_experts_per_tok, w0_tpu, w1_tpu, w2_tpu, blocksize)
    out_tpu = net_tpu(x_tpu, selected_experts_tpu, routing_weights_tpu)

    out_cpu = out_cpu.float().flatten()
    out_tpu = out_tpu.float().to("cpu").flatten()

    cos_diff, RMSE, amax_diff = cal_diff(out_cpu, out_tpu, "fused_experts")

    if cos_diff > 1e-4:
        print(f"batch_size: {batch_size}, input_w: {input_w}, middle_w: {middle_w}, num_experts: {num_experts}, num_experts_per_tok: {num_experts_per_tok} fail")
        print(f"  out_tpu shape: {out_tpu.shape}, out_tpu: {out_tpu.cpu().flatten()[0:8]}")
        print(f"  out_cpu shape: {out_cpu.shape}, out_cpu: {out_cpu.flatten()[0:8]}")
        return False
    else:
        return True

if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.manual_seed(1000)
    random.seed(1000)

    ret = []
    ret.append(test_Qwen_moe(batch_size=1, input_w=7168, middle_w=256, num_experts=1, num_experts_per_tok=1))
    ret.append(test_Qwen_moe(batch_size=1, input_w=7168, middle_w=256, num_experts=16, num_experts_per_tok=8))
    ret.append(test_Qwen_moe(batch_size=1, input_w=7168, middle_w=256, num_experts=256, num_experts_per_tok=8))
    ret.append(test_Qwen_moe(batch_size=2, input_w=4096, middle_w=384, num_experts=2, num_experts_per_tok=2))
    ret.append(test_Qwen_moe(batch_size=8, input_w=4096, middle_w=384, num_experts=128, num_experts_per_tok=8))
    ret.append(test_Qwen_moe(batch_size=22, input_w=4096, middle_w=384, num_experts=128, num_experts_per_tok=8))
    if all(ret):
        print("all pass")
    else:
        print(f"some fail, ret: {ret}")
