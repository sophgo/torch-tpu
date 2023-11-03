from test_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

# 创建一个输入张量
input_tensor = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32).reshape(
    1, 1, 2, 2
)  # (batch_size, channels, height, width)

# 指定目标上采样尺寸
target_size = (5,5)  # (target_height, target_width)

# 使用F.upsample_bilinear进行双线性上采样，使用align_corners参数为True
upsampled_tensor_true_corners = torch.ops.aten.upsample_bilinear2d(
    input_tensor, output_size=target_size, align_corners=False
)
print(upsampled_tensor_true_corners.shape)

# 使用F.upsample_bilinear进行双线性上采样，同时指定scale_factor参数
# scale_factor = 2.0
# upsampled_tensor_with_scale_factor = torch.ops.aten.upsample_bilinear2d(
#     input_tensor, output_size=target_size, scale_factor=scale_factor, align_corners=False
# )


class TestUpsampling(nn.Module):
    def forward(self, x):
        res = [
            torch.ops.aten.upsample_bilinear2d(
                x, target_size, align_corners=False
            ),
            torch.ops.aten.upsample_bilinear2d(
                x, target_size, align_corners=True
            ),
            torch.ops.aten.upsample_bilinear2d(
                x, target_size, align_corners=False, scales_w=3, scales_h=3
            ),
            torch.ops.aten.upsample_bilinear2d(
                x, target_size, align_corners=True, scales_w=5, scales_h=5
            ),
            torch.ops.aten.upsample_nearest2d(
                x, target_size
            ),
            torch.ops.aten.upsample_nearest2d(
                x, target_size
            ),
            torch.ops.aten.upsample_nearest2d(
                x, target_size, 3, 3
            ),
            torch.ops.aten.upsample_nearest2d(
                x, target_size, 3, 3
            )
        ]
        # print([i.cpu() for i in res])
        return res


def case1():
    ipts = [input_tensor]
    mem = []
    # print(TestUpsampling()(input_tensor))
    Evaluator().add_abs_evalute().evavlute([TestUpsampling()], ipts, mem=mem)
    # try:
    # except:
    #     print(mem)

def test_upsample_nearest2d_backward():
    # 创建示例梯度输出张量
    src_size = 3
    dst_size = 6
    grad1 = torch.range(1, 1 * src_size * src_size).view(1, 1, src_size, src_size)  # 根据需要修改维度
    grad_out = torch.range(1, dst_size * dst_size).view(1, 1, dst_size, dst_size)

    # 定义输出和输入大小
    output_size = (1,1,src_size,src_size)
    input_size = (dst_size,dst_size)

    # 可选的上采样比例（默认为None）
    scales_h = None
    scales_w = None

    # 创建一个与grad_output形状相同的梯度输入张量
    grad_o1 = torch.zeros_like(grad1)
    grad_in = torch.zeros_like(grad1)
    torch.ops.aten.upsample_nearest2d_backward(grad_out, input_size, output_size, grad_input=grad_in)
    print("cpu")
    print(grad_in)

    grad_out_tpu = grad_out.clone().to(device)
    grad_in_tpu = grad_o1.clone().to(device)
    # out_tpu = torch.ops.aten.upsample_nearest2d_backward(grad_out_tpu, size2, size1, grad_input=grad_in_tpu)
    out_tpu = torch.ops.aten.upsample_nearest2d_backward(grad_out_tpu, input_size, output_size)
    print("tpu")
    print(out_tpu.cpu())
    
if __name__ == "__main__":
    # case1()
    test_upsample_nearest2d_backward()
