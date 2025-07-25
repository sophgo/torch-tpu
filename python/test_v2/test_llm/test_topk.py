import torch
import pytest

# ground truth 实现：用 CPU 的 torch.topk 作为参考
# 第一步：输入数据在 CPU 上调用 torch.topk，得到 value 和 index
# 返回 value 和 index


def topk_cpu(x, k):
    # 第一步：直接在 CPU 上调用 torch.topk
    values, indices = torch.topk(x, k, dim=-1)
    return values.cpu().float(), indices.cpu().float()


# tpu 实现：用 tpu 上的 torch.topk
# 第一步：将输入搬到 tpu
# 第二步：在 tpu 上调用 torch.topk
# 第三步：结果搬回 cpu


def topk_tpu(x, k, device):
    # 第一步：将输入搬到 tpu
    x_tpu = x.to(device)
    # 第二步：在 tpu 上调用 torch.topk
    values_tpu, indices_tpu = torch.topk(x_tpu, k, dim=-1)
    # 第三步：结果搬回 cpu
    return values_tpu.cpu().float(), indices_tpu.cpu().float()


def cmp_topk(values_cpu, indices_cpu, values_tpu, indices_tpu):
    # 第一步：判断 value 是否一致
    assert torch.allclose(
        values_cpu, values_tpu, rtol=1e-5, atol=1e-6
    ), "values not close"
    # 第二步：判断 index 是否一致
    assert torch.equal(indices_cpu, indices_tpu), "indices not equal"


# 参数化测试函数
# 用 pytest.mark.parametrize 参数化 shape, k, dim
# 第一步：生成随机输入
# 第二步：分别用 cpu/tpu 实现跑
# 第三步：用 cmp_topk 对比


@pytest.mark.parametrize(
    "rand_batch,repeat_batch,n,k",
    [
        (1, 1, 256, 8),
        (8, 1, 256, 8),
        (16, 1, 256, 8),
        (64, 1, 256, 8),
        (128, 1, 256, 8),
    ],
)
def test_moe_topk(rand_batch, repeat_batch, n, k, device="tpu"):
    # 第一步：生成随机输入，float32，范围[-1000, 1000]
    x = torch.stack(
        [torch.randperm(n, dtype=torch.bfloat16) for _ in range(rand_batch)]
    ).repeat(repeat_batch, 1)

    # 第二步：分别用 cpu/tpu 实现跑
    values_cpu, indices_cpu = topk_cpu(x, k)
    values_tpu, indices_tpu = topk_tpu(x, k, device)
    # 第三步：用 cmp_topk 对比
    cmp_topk(values_cpu, indices_cpu, values_tpu, indices_tpu)


@pytest.mark.parametrize(
    "rand_batch,repeat_batch,n,k",
    [
        (1, 1, 256, 8),
        (8, 1, 256, 8),
        (16, 1, 256, 8),
        (64, 1, 256, 8),
        (128, 1, 256, 8),
    ],
)
def test_moe_topk(rand_batch, repeat_batch, n, k, device="tpu"):
    # 第一步：生成随机输入，float32，范围[-1000, 1000]
    x = torch.stack(
        [torch.randperm(n, dtype=torch.bfloat16) for _ in range(rand_batch)]
    ).repeat(repeat_batch, 1)

    # 第二步：分别用 cpu/tpu 实现跑
    values_cpu, indices_cpu = topk_cpu(x, k)
    values_tpu, indices_tpu = topk_tpu(x, k, device)
    # 第三步：用 cmp_topk 对比
    cmp_topk(values_cpu, indices_cpu, values_tpu, indices_tpu)


# reproduce topk precision bugs @2025.07.21
# failed when batch size larger than 16
@pytest.mark.parametrize(
    "rand_batch,repeat_batch,n,groups,k",
    [
        # (1, 1, 256, 32, 2),
        # (8, 1, 256, 32, 2),
        (16, 1, 256, 32, 2),  # ok
        (17, 1, 256, 32, 2),  # failed
        (128, 1, 256, 32, 2),  # failed
    ],
)
def test_mid_topk(rand_batch, repeat_batch, n, groups, k, device="tpu"):
    # 第一步：生成随机输入，float32，范围[-1000, 1000]
    x = (
        torch.stack(
            [torch.randperm(n, dtype=torch.bfloat16) for _ in range(rand_batch)]
        )
        .repeat(repeat_batch, 1)
        .reshape(rand_batch * repeat_batch, groups, -1)
    )

    # 第二步：分别用 cpu/tpu 实现跑
    values_cpu, indices_cpu = topk_cpu(x, k)
    values_tpu, indices_tpu = topk_tpu(x, k, device)
    # 第三步：用 cmp_topk 对比
    cmp_topk(values_cpu, indices_cpu, values_tpu, indices_tpu)


if __name__ == "__main__":
    test_mid_topk(17, 1, 256, 32, 2, device="tpu")
