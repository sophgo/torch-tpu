import pytest
import torch
import torch_tpu

# scatter_add_ 用于LLM服务中进行频率惩罚相关操作，注意为原地操作
# ground truth 实现：用 CPU 的 torch.Tensor.scatter_add_ 作为参考
# 第一步 在cpu上和tpu上准备输入数据
# 第二步：输入数据在 CPU和TPU 上分别调用 torch.Tensor.scatter_add_ ，得到原地scatter_add_后的结果
# 第三步：TPU上的结果搬回CPU，和CPU上计算的结果进行比较

# 参数化测试frequency_penalty里的scatter_add_
# 用 pytest.mark.parametrize 参数化 batch_size, vocab_size, seq_len
# 第一步：生成随机输入
# 第二步：分别用 cpu/tpu 跑torch.Tensor.scatter_add_，注意scatter_add_ 在TPU上只支持最后一个维度进行scatter_add_
# 第三步：用 torch.allclose 对比
@pytest.mark.parametrize("batch_size", [1, 10, 20, 30, 40, 50, 60,])
@pytest.mark.parametrize("vocab_size", [32000, 129280]) # llama-2-7b-chat-hf: 32000, DeepSeek-R1: 129280
@pytest.mark.parametrize("seq_len", [128, 512, 1024, 2048, 4096, 8192, 16384, 32768, ])
def test_frequency_penalty(batch_size, vocab_size, seq_len, device="tpu"):
    token_freq_cpu = torch.zeros((batch_size, vocab_size), device="cpu", dtype=torch.float32)
    token_freq_tpu = token_freq_cpu.to(device)
    input_ids_cpu = torch.randint(0, vocab_size, (batch_size, seq_len), device="cpu", dtype=torch.int64)
    input_ids_tpu = input_ids_cpu.to(torch.int32).to(device)
    src_cpu = torch.ones_like(input_ids_cpu, device="cpu", dtype=torch.float32)
    src_tpu = src_cpu.to(device)

    # tpu目前只支持dim()-1，即最后一个维度做scatter_add_
    token_freq_cpu.scatter_add_(token_freq_cpu.dim() - 1, input_ids_cpu, src_cpu)
    token_freq_tpu.scatter_add_(token_freq_tpu.dim() - 1, input_ids_tpu, src_tpu)
    assert torch.allclose(token_freq_cpu.float(), token_freq_tpu.cpu().float())