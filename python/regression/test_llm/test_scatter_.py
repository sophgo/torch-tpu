import pytest
import torch
import torch_tpu

# scatter_ 用于LLM服务中进行存在惩罚，top_p，typical_p 相关操作，注意为原地操作
# ground truth 实现：用 CPU 的 torch.Tensor.scatter_ 作为参考
# 第一步 在cpu上和tpu上准备输入数据
# 第二步：输入数据在 CPU和TPU 上分别调用 torch.Tensor.scatter_ ，得到原地scatter_后的结果
# 第三步：TPU上的结果搬回CPU，和CPU上计算的结果进行比较

# 参数化测试presence_penalty里的scatter_
# 用 pytest.mark.parametrize 参数化 batch_size, vocab_size, seq_len
# 第一步：生成随机输入
# 第二步：分别用 cpu/tpu 实现跑torch.Tensor.scatter_，注意scatter_ 在TPU上只支持最后一个维度进行scatter_
# 第三步：用 torch.allclose 对比
@pytest.mark.parametrize("batch_size", [1, 10, 20, 30, 40, 50, 60,])
@pytest.mark.parametrize("vocab_size", [32000, 129280]) # llama-2-7b-chat-hf: 32000, DeepSeek-R1: 129280
@pytest.mark.parametrize("seq_len", [128, 512, 1024, 2048, 4096, 8192, 16384, 32768, ])
def test_presence_penalty(batch_size, vocab_size, seq_len, device="tpu"):
    logits_cpu = torch.randn((batch_size, vocab_size), device="cpu", dtype=torch.float32)
    scores_cpu = torch.softmax(logits_cpu, dim=logits_cpu.dim()-1).to(torch.float16)
    scores_tpu = scores_cpu.to(device)
    input_ids_cpu = torch.randint(0, vocab_size, (batch_size, seq_len), device="cpu", dtype=torch.int64)
    input_ids_tpu = input_ids_cpu.to(torch.int32).to(device)
    score_cpu = torch.rand_like(input_ids_cpu, device="cpu", dtype=torch.float16)
    score_tpu = score_cpu.to(device)

    # tpu目前只支持dim()-1，即最后一个维度做scatter_
    scores_cpu.scatter_(scores_cpu.dim() - 1, input_ids_cpu, score_cpu)
    scores_tpu.scatter_(scores_tpu.dim() - 1, input_ids_tpu, score_tpu)
    assert torch.allclose(scores_cpu.float(), scores_tpu.cpu().float())