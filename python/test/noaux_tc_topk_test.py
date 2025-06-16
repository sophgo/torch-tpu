import torch
import torch.nn.functional as F
import torch_tpu

device = "tpu"

import numpy as np


def gt_noaux_tc_topk(scores, n_groups, topk_groups, top_k):
    bs = scores.shape[0]
    # 将分数重塑为组的形式
    scores = scores.view(bs, n_groups, -1)
    # 计算每个组的前2个最高分的和
    group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
    # 选择得分最高的 topk_groups 个组

    indices = group_scores.topk(topk_groups, dim=-1)[1]
    # 创建掩码，将选中的组标记为False
    mask = scores.new_ones(bs, n_groups, dtype=bool).scatter_(1, indices, False)
    # 将未选中的组的分数设置为负无穷
    scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
    # 选择最终的前8个最高分
    values, indices = torch.topk(scores, top_k, dim=-1)

    return (
        values,
        indices,
        {
            "group_scores": group_scores,
            "indices": indices,
            "mask": mask,
            "scores": scores,
        },
    )


def noaux_topk_v1_float(scores, n_groups, topk_groups, top_k):
    bs = scores.shape[0]
    scores = scores.clone().to(device)
    # torch.ops.my_ops.enable_profile(1024, 2)
    scores = scores.view(bs, n_groups, -1)
    group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)

    indices = group_scores.topk(topk_groups, dim=-1)[1]

    mask = torch.ones(bs, n_groups, dtype=torch.float32).to(device)
    mask.scatter_add_(1, indices, -torch.ones_like(indices, dtype=torch.float32))
    scores = scores.masked_fill_(mask.bool().unsqueeze(-1), float("-inf")).flatten(1)
    values, indices = torch.topk(scores, top_k, dim=-1)
    # torch.ops.my_ops.disable_profile()
    return (
        values.cpu(),
        indices.cpu(),
        {
            "group_scores": group_scores.cpu(),
            "mask": mask.cpu(),
            "scores": scores.cpu(),
        },
    )


def noaux_topk_v1(scores, n_groups, topk_groups, top_k):
    bs = scores.shape[0]
    scores = scores.to(torch.bfloat16).clone().to(device)
    # torch.ops.my_ops.enable_profile(1024, 2)
    # 16, 8, 32
    scores = scores.view(bs, n_groups, -1)

    # 16, 8
    group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)

    # 16, 4
    indices = group_scores.topk(topk_groups, dim=-1)[1]

    # 16, 8
    mask = torch.ones(bs, n_groups, dtype=torch.bfloat16).to(device)
    mask.scatter_add_(1, indices, -torch.ones_like(indices, dtype=torch.bfloat16))

    # 16, 8, 1 => 16, 8, 32
    scores = scores.masked_fill_(mask.bool().unsqueeze(-1), float("-inf")).flatten(1)
    values, indices = torch.topk(scores, top_k, dim=-1)

    # torch.ops.my_ops.disable_profile()
    return (
        values.cpu(),
        indices.cpu(),
        {
            "group_scores": group_scores.cpu(),
            "indices": indices.cpu(),
            "mask": mask.cpu(),
            "scores": scores.cpu(),
        },
    )


def noaux_topk_v1_2(scores, n_groups, topk_groups, top_k):
    # TODO: [ERROR scatter.cpp:47] Expected dim == input.dim - 1 to be true
    bs = scores.shape[0]
    # 将分数重塑为组的形式
    scores = scores.view(bs, n_groups, -1).to(torch.bfloat16).to(device)
    # 计算每个组的前2个最高分的和
    group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
    # 选择得分最高的 topk_groups 个组

    indices = group_scores.topk(topk_groups, dim=-1, largest=False)[1]

    print(indices.shape)
    print(indices.unsqueeze(-1).repeat(1, 1, 32).shape)

    indices = indices.unsqueeze(-1).repeat(1, 1, 32)

    scores.scatter_add_(
        1, indices, torch.full_like(indices, float("-inf"), dtype=torch.bfloat16)
    )
    scores = scores.flatten(1)

    values, indices = torch.topk(scores, top_k, dim=-1)

    # torch.ops.my_ops.disable_profile()
    return (
        values.cpu(),
        indices.cpu(),
        {
            "group_scores": group_scores.cpu(),
            "indices": indices.cpu(),
            "scores": scores.cpu(),
        },
    )


def noaux_topk_v1_1(scores, n_groups, topk_groups, top_k):
    bs = scores.shape[0]
    scores = scores.to(torch.bfloat16).clone().to(device)
    # torch.ops.my_ops.enable_profile(1024, 2)
    scores = scores.view(bs, n_groups, -1)
    group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)

    indices = group_scores.topk(topk_groups, dim=-1)[1]

    mask = torch.ones(bs, n_groups, dtype=torch.bfloat16).to(device)
    mask.scatter_add_(1, indices, -torch.ones_like(indices, dtype=torch.bfloat16))

    scores = scores.masked_fill_(mask.bool().unsqueeze(-1), float("-inf")).flatten(1)

    values, indices = torch.topk(scores, top_k, dim=-1)

    # torch.ops.my_ops.disable_profile()
    return (
        values.cpu(),
        indices.cpu(),
        {
            "group_scores": group_scores.cpu(),
            "indices": indices.cpu(),
            "mask": mask.cpu(),
            "scores": scores.cpu(),
        },
    )


def noaux_topk_v2(scores, n_groups, topk_groups, top_k):
    # myops noaux_tc_topk

    values = torch.empty(scores.shape[0], top_k, dtype=scores.dtype).to(device)
    indices = torch.empty(scores.shape[0], top_k, dtype=torch.int32).to(device)

    scores = scores.clone().to(device)
    # torch.ops.my_ops.enable_profile(1024, 2)

    torch.ops.my_ops.noaux_tc_topk(
        values, indices, scores, n_groups, topk_groups, top_k
    )
    # torch.ops.my_ops.disable_profile()

    return values.cpu(), indices.cpu(), {}


torch.manual_seed(42)
batch_size = 16
scores = (
    torch.stack(
        [torch.randperm(256, dtype=torch.float32) for i in range(batch_size)]
    ).reshape(batch_size, -1)
    # .to(torch.float32)
    .to(torch.bfloat16)
)
scores = scores.repeat(128, 1)


def test_noaux_tc_topk():

    # TODO fix bfloat16 overflow
    # scores = scores / scores.sum(dim=-1, keepdim=True)

    n_groups = 8
    topk_groups = 4
    top_k = 8

    values, indices, middle = gt_noaux_tc_topk(
        scores.clone(), n_groups, topk_groups, top_k
    )

    values_v1, indices_v1, middle = noaux_topk_v2(
        scores.clone(), n_groups, topk_groups, top_k
    )

    assert torch.allclose(values.float(), values_v1.to(torch.float32))
    assert torch.allclose(indices, indices_v1.long())


if __name__ == "__main__":
    test_noaux_tc_topk()
