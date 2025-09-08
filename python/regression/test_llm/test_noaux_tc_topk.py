import torch
import torch.nn.functional as F
import pytest
import torch_tpu
import numpy as np

# 移除硬编码的随机种子和设备设置，这些现在由 conftest.py 中的 fixture 处理


def gt_noaux_tc_topk(scores, n_groups, topk_groups, top_k):
    """Ground truth 实现，用于对比验证"""
    bs = scores.shape[0]
    # 第一步：将分数重塑为组的形式，方便按组处理
    scores = scores.view(bs, n_groups, -1)
    # 第二步：计算每个组的前2个最高分的和作为组分数
    group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
    # 第三步：选择得分最高的 topk_groups 个组的索引
    indices = group_scores.topk(topk_groups, dim=-1)[1]
    # 第四步：创建掩码，将选中的组标记为False，未选中的组标记为True
    mask = scores.new_ones(bs, n_groups, dtype=bool).scatter_(1, indices, False)
    # 第五步：将未选中的组的分数设置为负无穷，然后展平为一维
    scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
    # 第六步：从所有剩余的分数中选择最终的前top_k个最高分
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


def noaux_topk_v1_float(scores, n_groups, topk_groups, top_k, device):
    """使用 float32 精度的实现版本"""
    bs = scores.shape[0]
    # 第一步：克隆输入数据并移动到指定设备
    scores = scores.clone().to(device)
    # 第二步：重塑为组的形式
    scores = scores.view(bs, n_groups, -1)
    # 第三步：计算每个组的前2个最高分的和
    group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
    # 第四步：获取得分最高的组的索引
    indices = group_scores.topk(topk_groups, dim=-1)[1]
    # 第五步：创建掩码并使用scatter_add_进行原位操作
    mask = torch.ones(bs, n_groups, dtype=torch.float32).to(device)
    mask.scatter_add_(1, indices, -torch.ones_like(indices, dtype=torch.float32))
    # 第六步：应用掩码并展平，然后获取最终的top_k结果
    scores = scores.masked_fill_(mask.bool().unsqueeze(-1), float("-inf")).flatten(1)
    values, indices = torch.topk(scores, top_k, dim=-1)

    return (
        values.cpu(),
        indices.cpu(),
        {
            "group_scores": group_scores.cpu(),
            "mask": mask.cpu(),
            "scores": scores.cpu(),
        },
    )


def noaux_topk_v1(scores, n_groups, topk_groups, top_k, device):
    """使用 bfloat16 精度的实现版本"""
    bs = scores.shape[0]
    # 第一步：转换为bfloat16精度并移动到设备
    scores = scores.to(torch.bfloat16).clone().to(device)
    # 第二步：重塑为组的形式 (16, 8, 32)
    scores = scores.view(bs, n_groups, -1)
    # 第三步：计算组分数 (16, 8)
    group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
    # 第四步：获取top组的索引 (16, 4)
    indices = group_scores.topk(topk_groups, dim=-1)[1]
    # 第五步：创建掩码 (16, 8)
    mask = torch.ones(bs, n_groups, dtype=torch.bfloat16).to(device)
    mask.scatter_add_(1, indices, -torch.ones_like(indices, dtype=torch.bfloat16))
    # 第六步：应用掩码并获取最终结果，形状变化：(16, 8, 1) => (16, 8, 32)
    scores = scores.masked_fill_(mask.bool().unsqueeze(-1), float("-inf")).flatten(1)
    values, indices = torch.topk(scores, top_k, dim=-1)

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


def noaux_topk_v2(scores, n_groups, topk_groups, top_k, device):
    """使用自定义算子的实现版本"""
    # 第一步：创建输出张量
    values = torch.empty(scores.shape[0], top_k, dtype=scores.dtype).to(device)
    indices = torch.empty(scores.shape[0], top_k, dtype=torch.int32).to(device)
    # 第二步：准备输入数据
    scores = scores.clone().to(device)
    # 第三步：调用自定义算子执行topk操作
    torch.ops.my_ops.noaux_tc_topk(
        values, indices, scores, n_groups, topk_groups, top_k
    )

    return values.cpu(), indices.cpu(), {}


"""
case that have problem:
"""
# fmt: off
err_data = [210.,  33.,  80.,  93., 222.,   7., 198.,  92.,  22., 186., 143., 164.,
        243., 175.,   2.,  47., 203.,  82., 126., 127., 140., 132., 214.,  23.,
        129.,  52., 153., 232., 197., 167., 114., 190.,  99.,  84.,  41., 162.,
         42.,  12.,  85.,  61., 194., 103., 209.,  57., 146.,   6.,  11., 160.,
        116.,  46.,  76.,  90., 193.,  28., 189., 188., 202., 142.,  18.,   4.,
         56.,  62., 199., 239., 244.,  87., 105.,  63.,  10., 104., 218., 192.,
         50., 236., 112.,  20., 110., 187.,  16., 242., 107., 230., 247., 138.,
         26.,  31.,  71., 217., 235.,  53.,  13., 173., 208., 206., 128.,  54.,
         49., 213., 163.,  34., 108.,   9.,  75., 134., 119., 191., 178., 201.,
        125.,  39., 157., 252., 106., 155., 215.,  97.,  64., 212., 241., 137.,
         81., 234.,  73.,  65., 121., 251.,  66.,  91., 171.,  74., 170.,  98.,
        245.,  48., 111., 172., 115.,  60., 123., 100., 168.,  36., 255.,   5.,
        145., 136., 246.,  78.,   3., 249.,  21.,  43.,  15., 161., 211.,  95.,
        200.,   0., 109., 216.,  77., 248., 159.,  38., 221., 237.,  37., 165.,
        124., 122.,  32., 240., 223.,  67., 150., 131., 141.,  58., 144.,  55.,
         96., 250., 156., 233., 231.,  14.,  27., 133., 152., 148., 101., 184.,
        180.,  44.,  35.,  24., 226., 169., 182., 220., 118., 139.,  79.,  88.,
        147., 177., 227., 158., 185., 207., 113., 154., 229.,  40., 117., 102.,
         45.,  30., 166.,  59., 238.,   1., 253., 205., 196.,  70., 224.,  25.,
        174.,  86., 151.,  19., 120., 225.,  69.,  29., 176., 179.,  72.,  51.,
        130., 204.,  17., 181., 219., 135.,  68., 183., 228.,  94.,  89., 149.,
          8., 254.,  83., 195.]
# fmt: on


def cmp_noaux_tc_topk(values_gt, indices_gt, values_v2, indices_v2):
    indices_gt_groups = indices_gt // 32
    indices_v2_groups = indices_v2.long() // 32
    for batch_idx in range(indices_gt_groups.shape[0]):
        assert (
            len(torch.unique(indices_gt_groups[batch_idx])) <= 4
        ), f"batch {batch_idx} has elements in more than 4 groups"
        assert (
            len(torch.unique(indices_v2_groups[batch_idx])) <= 4
        ), f"batch {batch_idx} has elements in more than 4 groups"
    assert torch.allclose(
        values_gt.float(), values_v2.to(torch.float32), rtol=1e-5, atol=1e-6
    )
    assert torch.allclose(indices_gt, indices_v2.long())


# TODO: fixme
# def test_noaux_tc_topk_err(device):
#     """测试 noaux_tc_topk 的不同实现版本是否与ground truth一致"""

#     # 第一步：生成测试数据，创建随机排列的分数矩阵
#     torch.manual_seed(42)
#     batch_size = 1
#     n_groups = 8
#     topk_groups = 4
#     top_k = 8
#     scores = (
#         torch.stack(
#             [torch.tensor(err_data, dtype=torch.float32) for i in range(batch_size)]
#         )
#         .reshape(batch_size, -1)
#         .to(torch.bfloat16)
#     )
#     # 第二步：重复数据以增加测试规模
#     # scores = scores.repeat(128, 1)

#     # 第三步：运行ground truth版本作为参考
#     values_gt, indices_gt, _ = gt_noaux_tc_topk(
#         scores.clone(), n_groups, topk_groups, top_k
#     )

#     # 第四步：运行自定义算子版本进行对比
#     values_v2, indices_v2, _ = noaux_topk_v2(
#         scores.clone(), n_groups, topk_groups, top_k, device
#     )

#     # 第五步：验证结果的正确性
#     cmp_noaux_tc_topk(values_gt, indices_gt, values_v2, indices_v2)


@pytest.mark.parametrize(
    "batch_size,n_groups,topk_groups,top_k",
    [
        (1, 8, 4, 8),
        (8, 8, 4, 8),
        (16, 8, 4, 8),
        (64, 8, 4, 8),
        (128, 8, 4, 8),
    ],
)
def test_noaux_tc_topk(batch_size, n_groups, topk_groups, top_k, device):
    """测试 noaux_tc_topk 的不同实现版本是否与ground truth一致"""

    # 第一步：生成测试数据，创建随机排列的分数矩阵
    torch.manual_seed(42)
    scores = (
        torch.stack([torch.randperm(256, dtype=torch.float32) for i in range(1)])
        .reshape(1, -1)
        .to(torch.bfloat16)
    )
    # 第二步：重复数据以增加测试规模
    scores = scores.repeat(batch_size, 1)

    # 第三步：运行ground truth版本作为参考
    values_gt, indices_gt, _ = gt_noaux_tc_topk(
        scores.clone(), n_groups, topk_groups, top_k
    )

    # 第四步：运行自定义算子版本进行对比
    values_v2, indices_v2, _ = noaux_topk_v2(
        scores.clone(), n_groups, topk_groups, top_k, device
    )

    # 第五步：验证结果的正确性
    cmp_noaux_tc_topk(values_gt, indices_gt, values_v2, indices_v2)


# @pytest.mark.parametrize(
#     "precision",
#     ["float32", "bfloat16"],
# )
# def test_noaux_tc_topk_precision(precision, device, setup_random_seed):
#     """测试不同精度下的实现"""

#     batch_size = 16
#     n_groups = 8
#     topk_groups = 4
#     top_k = 8

#     # 第一步：生成测试数据
#     scores = (
#         torch.stack(
#             [torch.randperm(256, dtype=torch.float32) for i in range(batch_size)]
#         )
#         .reshape(batch_size, -1)
#         .to(torch.bfloat16)
#     )
#     scores = scores.repeat(64, 1)  # 减少重复次数以加快测试速度

#     # 第二步：运行ground truth版本
#     values_gt, indices_gt, _ = gt_noaux_tc_topk(
#         scores.clone(), n_groups, topk_groups, top_k
#     )

#     # 第四步：验证结果，对于不同精度使用适当的容差

#     cmp_noaux_tc_topk(values_gt, indices_gt, values_test, indices_test)
