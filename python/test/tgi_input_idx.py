import torch
import torch_tpu
import numpy as np
import random

def debug_tensor_info(tensor, name):
    print(f"{name}: device={tensor.device}, dtype={tensor.dtype}, shape={tensor.shape}, is_contiguous={tensor.is_contiguous()}")

def cpu_reference_impl(all_input_ids, next_ids, input_lengths, n_accept=1):
    cpu_input = all_input_ids.clone()
    cpu_lengths = input_lengths.copy()

    for b in range(len(input_lengths)):
        current_len = input_lengths[b]
        cpu_input[b, current_len:current_len+n_accept] = next_ids[b, :n_accept]
        cpu_lengths[b] += n_accept

    return cpu_input, cpu_lengths

def generate_test_data(batch_size=2, seq_length=10):
    """生成测试数据（支持任意 batch_size）"""
    device_id = 0
    torch.tpu.set_device(device_id)

    # 初始化TPU张量 (形状 [batch_size, seq_length])
    all_input_ids = torch.zeros((batch_size, seq_length),
                              dtype=torch.int32,
                              device=f"tpu:{device_id}")

    # 随机生成每个batch的初始长度（1 <= length < seq_length）
    input_lengths = [random.randint(1, seq_length - 1) for _ in range(batch_size)]

    # 填充初始数据
    for b in range(batch_size):
        all_input_ids[b, :input_lengths[b]] = torch.randint(1, 10, (input_lengths[b],))

    # 新生成的token (形状 [batch_size, 1])
    next_ids = torch.randint(10, 100, (batch_size, 1),
                            dtype=torch.int32,
                            device=f"tpu:{device_id}")

    return all_input_ids, next_ids, input_lengths

def run_batch_test(batch_size):
    tpu_input, tpu_next_ids, tpu_lengths = generate_test_data(batch_size)
    cpu_input = tpu_input.cpu().clone()
    cpu_next_ids = tpu_next_ids.cpu().clone()
    cpu_lengths = tpu_lengths.copy()

    print(f"\nBatch大小: {batch_size}, 序列长度: {tpu_input.shape[1]}")
    print("TPU初始数据:\n", tpu_input.cpu().numpy())
    print("TPU新token:\n", tpu_next_ids.cpu().numpy())
    print("各序列长度:", tpu_lengths)

    cpu_result, _ = cpu_reference_impl(cpu_input, cpu_next_ids, cpu_lengths)

    torch.ops.my_ops.tgi_input_ids_update_decode_phase(
        tpu_input,
        tpu_next_ids,
        tpu_lengths,
        1  # n_accept=1
    )

    tpu_result = tpu_input.cpu()
    print("\n===== 结果对比 =====")
    print("CPU结果:\n", cpu_result.numpy())
    print("TPU结果:\n", tpu_result.numpy())

    is_match = torch.allclose(cpu_result, tpu_result)
    print("\n验证结果:", "✅ 匹配" if is_match else "❌ 不匹配")
    return is_match

if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    for batch_size in [16]:
        success = run_batch_test(batch_size)
        print(f"\n测试 batch_size={batch_size}", "通过" if success else "失败")
