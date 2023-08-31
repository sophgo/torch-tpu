from test_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

# 创建一个输入张量
input_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32).reshape(
    1, 1, 2, 2
)  # (batch_size, channels, height, width)

# 指定目标上采样尺寸
target_size = (4, 4)  # (target_height, target_width)

# 使用F.upsample_bilinear进行双线性上采样，使用align_corners参数为True
upsampled_tensor_true_corners = torch.ops.aten.upsample_bilinear2d(
    input_tensor, output_size=target_size, align_corners=True
)
print(upsampled_tensor_true_corners.shape)

# 使用F.upsample_bilinear进行双线性上采样，同时指定scale_factor参数
# scale_factor = 2.0
# upsampled_tensor_with_scale_factor = torch.ops.aten.upsample_bilinear2d(
#     input_tensor, output_size=target_size, scale_factor=scale_factor, align_corners=False
# )


class TestUpsampling(nn.Module):
    def forward(self, x):
        return [
            torch.ops.aten.upsample_bilinear2d(
                x, target_size, align_corners=False
            )
        ]


def case1():
    ipts = [input_tensor]
    mem = []
    # print(TestUpsampling()(input_tensor))
    Evaluator().add_abs_evalute().evavlute([TestUpsampling()], ipts, mem=mem)
    # print("pass")


if __name__ == "__main__":
    case1()
