import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pytest
import torch_tpu
from torch_tpu.utils.compare import cos_sim, cal_diff
import random

def weight_dequant(
    x: torch.Tensor, s: torch.Tensor, block_size: int = 128
) -> torch.Tensor:
    """
    使用提供的缩放张量对给定的权重张量进行反量化

    Args:
        x (torch.Tensor): 形状为 (M, N) 的量化权重张量
        s (torch.Tensor): 形状为 (M, N) 的缩放张量
        block_size (int, optional): 分块大小。默认为 128

    Returns:
        torch.Tensor: 与 `x` 形状相同的反量化权重张量
    """
    assert x.is_contiguous() and s.is_contiguous(), "Input tensors must be contiguous"
    assert x.dim() == 2 and s.dim() == 2, "Input tensors must have 2 dimensions"

    M, N = x.size()
    # 第一步：创建输出张量，使用默认数据类型
    y = torch.empty_like(x, dtype=torch.get_default_dtype())

    # 第二步：模拟 Triton 的分块计算逻辑，遍历每个块
    for pid_m in range((M + block_size - 1) // block_size):
        for pid_n in range((N + block_size - 1) // block_size):
            # 第三步：计算当前块的起始和结束索引
            offs_m_start = pid_m * block_size
            offs_m_end = min(offs_m_start + block_size, M)
            offs_n_start = pid_n * block_size
            offs_n_end = min(offs_n_start + block_size, N)

            # 第四步：提取当前块的量化权重和缩放因子
            x_block = x[offs_m_start:offs_m_end, offs_n_start:offs_n_end]
            s_block = s[pid_m, pid_n]  # 假设 s 的每个块对应一个缩放因子

            # 第五步：反量化逻辑，将量化权重乘以缩放因子
            y_block = x_block.to(torch.float32) * s_block

            # 第六步：将反量化结果写回输出张量
            y[offs_m_start:offs_m_end, offs_n_start:offs_n_end] = y_block

    return y

class DeepseekMlp(nn.Module):
    def __init__(self, w0, w1, w2, s0, s1, s2, blocksize):
        super().__init__()
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        self.blocksize = blocksize

    def forward(self, x):
        # dequantized_weight0 = weight_dequant_vector(self.w0, self.s0, self.blocksize)
        dequantized_weight0 = weight_dequant(self.w0, self.s0, self.blocksize)
        out00 = F.linear(x, dequantized_weight0)

        dw0 = dequantized_weight0.to(torch.float32)
        x0 = x.to(torch.float32)
        out1 = F.linear(x0, dw0)
        out0 = F.silu(out00)

        # dequantized_weight1 = weight_dequant_vector(self.w1, self.s1, self.blocksize)
        dequantized_weight1 = weight_dequant(self.w1, self.s1, self.blocksize)
        out1 = F.linear(x, dequantized_weight1)
        out2 = out0 * out1
        # import pdb
        # pdb.set_trace()

        # dequantized_weight2 = weight_dequant_vector(self.w2, self.s2, self.blocksize)
        dequantized_weight2 = weight_dequant(self.w2, self.s2, self.blocksize)
        out3 = F.linear(out2, dequantized_weight2)

        return out3

class DeepseekMoE(nn.Module):
    def __init__(self, num_experts, num_experts_per_tok, w0, w1, w2, s0, s1, s2, blocksize):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.experts = nn.ModuleList([DeepseekMlp(w0[i], w1[i], w2[i], s0[i], s1[i], s2[i], blocksize) for i in range(num_experts)])

    def forward(self, x, selected_experts, out):
        # out = torch.zeros(x.shape[0], self.num_experts_per_tok, x.shape[1]).to(x.dtype).to(x.device)
        for i in range(selected_experts.shape[0]):
            for j in range(selected_experts.shape[1]):
                out[i, j, :] = self.experts[selected_experts[i, j]](x[i])
        return out

class DeepseekMoEFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w0, w1, w2, s0, s1, s2,
                num_experts, num_experts_per_tok, blocksize, selected_experts, routing_weights, 
                gathered_experts_out_buf, select_experts_middle, routing_weights_middle, gather_buffer, scatter_buffer, use_ppl):
        output_sample = None
        input_sample = None
        num_select_experts = None
        # select_experts_middle = None
        # routing_weights_middle = None
        # select_experts_middle = torch.empty(num_experts, x.shape[0], dtype = torch.int32, device = x.device)
        # routing_weights_middle = torch.empty(num_experts, x.shape[0], dtype = torch.int32, device = x.device)
        use_grouped_topk = True
        num_expert_group = 8
        topk_group = 4
        # gathered_experts_out_buf is of shape (seq x k x hs)
        # gathered_experts_out_buf = torch.empty(x.shape[0], num_experts_per_tok, x.shape[1], dtype = x.dtype, device = x.device)
        # print(f"selected_experts shape: {selected_experts.shape}, selected_experts: {selected_experts}")
        if use_ppl:
            torch.ops.my_ops.fused_moe_fused_experts_v2(gathered_experts_out_buf, x,
                                            output_sample, input_sample,
                                            w0, w1, w2,
                                            s0, s1, s2,
                                            selected_experts, routing_weights,
                                            num_select_experts,
                                            select_experts_middle, routing_weights_middle,
                                            gather_buffer,scatter_buffer,
                                            blocksize, num_experts, num_experts_per_tok,
                                            use_grouped_topk, num_expert_group, topk_group,
                                            None, None, None, False)
        else:
            torch.ops.my_ops.fused_moe_fused_experts(gathered_experts_out_buf, x,
                                                    output_sample, input_sample,
                                                    w0, w1, w2,
                                                    s0, s1, s2,
                                                    selected_experts, routing_weights,
                                                    num_select_experts,
                                                    select_experts_middle, routing_weights_middle,
                                                    blocksize, num_experts, num_experts_per_tok,
                                                    use_grouped_topk, num_expert_group, topk_group,
                                                    None, None, None, False)
        return gathered_experts_out_buf

class DeepseekMoEBlock(nn.Module):
    def __init__(self, num_experts, num_experts_per_tok, w0, w1, w2, s0, s1, s2, blocksize):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.w0 = w0
        self.w1 = w1
        self.w2 = w2
        self.s0 = s0
        self.s1 = s1
        self.s2 = s2
        self.blocksize = blocksize

    def forward(self, x, selected_experts, routing_weights, gathered_experts_out_buf, select_experts_middle, routing_weights_middle, gather_buffer, scatter_buffer, use_ppl):
        return DeepseekMoEFunc.apply(x, self.w0, self.w1, self.w2, self.s0, self.s1, self.s2,
                                     self.num_experts, self.num_experts_per_tok,
                                     self.blocksize, selected_experts, routing_weights, 
                                     gathered_experts_out_buf, select_experts_middle, routing_weights_middle, gather_buffer, scatter_buffer, use_ppl)

def deepseek_moe_cpu(x, selected_experts, out, num_experts, num_experts_per_tok, w0, w1, w2, s0, s1, s2, blocksize):
    """CPU 版本的 Deepseek MOE 函数式实现"""
    net_cpu = DeepseekMoE(num_experts, num_experts_per_tok, w0, w1, w2, s0, s1, s2, blocksize)
    out_cpu = net_cpu(x, selected_experts, out)
    return out_cpu


def deepseek_moe_tpu(x_tpu, selected_experts_tpu, routing_weights_tpu, select_experts_middle, routing_weights_middle, gathered_experts_out_buf, 
                    gather_buffer, scatter_buffer,num_experts, num_experts_per_tok, w0_tpu, w1_tpu, w2_tpu, s0_tpu, 
                    s1_tpu, s2_tpu, blocksize, use_ppl):
    """使用自定义 TPU 算子的 Deepseek MOE 函数式实现"""
    net_tpu = DeepseekMoEBlock(num_experts, num_experts_per_tok, w0_tpu, w1_tpu, w2_tpu, s0_tpu, s1_tpu, s2_tpu, blocksize)
    out_tpu = net_tpu(x_tpu, selected_experts_tpu, routing_weights_tpu, 
                gathered_experts_out_buf, select_experts_middle, routing_weights_middle, gather_buffer, scatter_buffer, use_ppl)

    return gathered_experts_out_buf


def cmp_deepseek_mm(out_cpu, out_tpu, tolerance=0.1):
    """
    比较 CPU 和 TPU 输出结果的函数

    Args:
        out_cpu: CPU 输出结果
        out_tpu: TPU 输出结果
        tolerance: 容差值

    Returns:
        dict: 包含各种比较指标的字典
    """
    # 第一步：将输出转换为相同格式进行比较
    out_cpu_flat = out_cpu.float().flatten()
    out_tpu_flat = out_tpu.float().cpu().flatten()

    # 第二步：计算差异
    out_diff = out_cpu_flat - out_tpu_flat
    ratio = abs(out_diff / torch.max(abs(out_cpu_flat), abs(out_tpu_flat)))

    # 第三步：找到最大差异的位置
    max_ratio_idx = torch.argmax(ratio)

    # 第四步：计算余弦相似度
    cosine_sim = cos_sim(out_cpu_flat.numpy(), out_tpu_flat.numpy())

    # 第五步：计算平均差异
    mean_diff = torch.mean(abs(out_diff)).item()

    return {
        "mean_diff": mean_diff,
        "max_ratio": ratio[max_ratio_idx].item(),
        "cosine_sim": cosine_sim,
        "max_ratio_cpu_val": out_cpu_flat[max_ratio_idx].item(),
        "max_ratio_tpu_val": out_tpu_flat[max_ratio_idx].item(),
    }


@pytest.mark.parametrize(
    "batch_size, input_w, middle_w, num_experts, num_experts_per_tok",
    [
        (1, 7168, 256, 1, 1),
        (1, 7168, 256, 2, 2),
        (1, 7168, 256, 16, 8),
        (1, 7168, 256, 256, 8),
        (8, 7168, 256, 256, 8),
    ],
)
@pytest.mark.priority_high
def test_deepseek_moe(
    batch_size,
    input_w,
    middle_w,
    num_experts,
    num_experts_per_tok,
    device,
    setup_random_seed
):
    """
    测试 Deepseek MOE 的 TPU 实现与 CPU 参考实现的一致性

    Args:
        batch_size: 批次大小
        input_w: 输入特征维度
        middle_w: 输出特征维度
        block_size: 分块大小
        device: 测试设备（由 fixture 提供）
        setup_random_seed: 随机种子设置（由 fixture 提供）
    """
    block_size = 128
    use_ppl = 0
    # 第一步：计算缩放因子的维度
    scale_m = (middle_w + block_size - 1) // block_size
    scale_n = (input_w + block_size - 1) // block_size

    selected_experts_list = []
    for i in range(batch_size):
        selected_experts_list.append(random.sample(range(num_experts), num_experts_per_tok))
    selected_experts = torch.tensor(selected_experts_list, dtype=torch.int32).reshape(batch_size, num_experts_per_tok)

    # 第二步：设置默认数据类型为 bfloat16
    torch.set_default_dtype(torch.bfloat16)

    routing_weights = torch.randn(batch_size, num_experts_per_tok, dtype=torch.bfloat16)

    w0 = torch.randn(num_experts, middle_w, input_w, )*0.1
    w1 = torch.randn(num_experts, middle_w, input_w)*0.1
    w2 = torch.randn(num_experts, input_w, middle_w)*0.1
    s0 = torch.randn(num_experts, scale_m, scale_n)*0.1
    s1 = torch.randn(num_experts, scale_m, scale_n)*0.1
    s2 = torch.randn(num_experts, scale_n, scale_m)*0.1 # N, M for cpu
    w0 = w0.to(torch.float8_e4m3fn)
    w1 = w1.to(torch.float8_e4m3fn)
    w2 = w2.to(torch.float8_e4m3fn)
    x = torch.randn(batch_size, input_w)

    # 第六步：运行 CPU 参考实现
    out = torch.zeros(x.shape[0], num_experts_per_tok, x.shape[1]).to(x.dtype).to(x.device)
    out_cpu = deepseek_moe_cpu(x, selected_experts, out, num_experts, num_experts_per_tok, w0, w1, w2, s0, s1, s2, block_size)

    # 第七步：运行 TPU 实现
    device = "tpu:0"
    x_tpu = x.to(device)
    w0_tpu = w0.to(device)
    w1_tpu = w1.to(device)
    w2_tpu = w2.transpose(1, 2).to(device) # transpose for TPU
    s0_tpu = s0.to(device)
    s1_tpu = s1.to(device)
    s2_tpu = s2.transpose(1, 2).to(device) # transpose for TPU
    selected_experts_tpu = selected_experts.to(device)
    routing_weights_tpu = routing_weights.to(device)
    select_experts_middle = torch.empty(num_experts, x.shape[0], dtype = torch.int32).to(device)
    routing_weights_middle = torch.empty(num_experts, x.shape[0], dtype = torch.int32).to(device)
    gathered_experts_out_buf = torch.empty(x.shape[0], num_experts_per_tok, x.shape[1], dtype = x.dtype).to(device)
    gather_buffer = torch.empty(num_experts, x.shape[0], dtype = torch.int32).to(device)
    scatter_buffer = torch.empty(num_experts, x.shape[0], num_experts_per_tok, dtype = torch.int32).to(device)
    out_tpu = deepseek_moe_tpu(x_tpu, selected_experts_tpu, routing_weights_tpu, select_experts_middle, routing_weights_middle, gathered_experts_out_buf, 
                    gather_buffer, scatter_buffer,num_experts, num_experts_per_tok, w0_tpu, w1_tpu, w2_tpu, s0_tpu, 
                    s1_tpu, s2_tpu, block_size, use_ppl)

    # 第八步：使用比较函数进行结果验证
    metrics = cmp_deepseek_mm(out_cpu, out_tpu, tolerance=0.1)
    out_cpu = out_cpu.float().flatten()
    out_tpu = out_tpu.to("cpu").float().flatten()
    cosm = cos_sim(out_cpu.numpy(), out_tpu.numpy())
    cos_diff, RMSE, amax_diff = cal_diff(out_cpu, out_tpu, "fused_experts")

    # 第九步：断言验证结果在容差范围内
    assert cosm > 0.99 and cos_diff < 1e-4

@pytest.mark.parametrize(
    "batch_size, input_w, middle_w, num_experts, num_experts_per_tok",
    [
        # (16384, 7168, 128, 256, 8),
        (256, 7168, 256, 256, 8),
    ],
)
@pytest.mark.priority_medium
def test_deepseek_moe_large_batch(
    batch_size,
    input_w,
    middle_w,
    num_experts,
    num_experts_per_tok,
    device,
    setup_random_seed,
):
    """
    测试大批次大小的 Deepseek MM 实现
    """
    test_deepseek_moe(
        batch_size,
        input_w,
        middle_w,
        num_experts,
        num_experts_per_tok,
        device,
        setup_random_seed,
    )

