import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import copy
import torch_tpu

torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def cos_sim(vector_a, vector_b):
    vector_a = vector_a.reshape(-1)
    vector_b = vector_b.reshape(-1)
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    with np.errstate(invalid='ignore'):
        cos = np.nan_to_num(num / denom)
    sim = 0.5 + 0.5 * cos
    return sim

class FastLinear(torch.nn.Module):
    def __init__(
        self,
        weight,
        bias,
    ) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(weight, requires_grad=False)
        if bias is not None:
            self.bias = torch.nn.Parameter(bias, requires_grad=False)
        else:
            self.bias = None

    @classmethod
    def load(cls, config, prefix: str, weights, bias: bool):
        weight = weights.get_tensor(f"{prefix}.weight")
        if bias:
            bias = weights.get_tensor(f"{prefix}.bias")
        else:
            bias = None
        return cls(weight, bias)

    def forward(self, input: torch.Tensor, **kwargs) -> torch.Tensor:
        output = kwargs.get('output', None)
        return F.linear(input, self.weight, self.bias,out=output)


def check_mlp():
    batch_size = 8192#4096
    input_w = 8192
    middle_w = 5120

    torch.set_default_dtype(torch.bfloat16)

    x = torch.randn(batch_size, input_w) # 输入数据 (batch_size=10, in_features=5)
    w0 = torch.randn(middle_w, input_w) # 量化权重 (out_features=8, in_features=5) torch.float8_e4m3fn
    b0 = torch.randn(middle_w)
    # net_cpu = FastLinear(w0, None)  # 无偏置
    net_cpu = FastLinear(w0, b0)
    out_cpu = net_cpu(x)

    print(f"output_cpu 输出形状: {out_cpu.shape}")


    x_tpu = x.to(device)
    w0_tpu = w0.to(device)
    b0_tpu = b0.to(device)
    # import pdb
    # pdb.set_trace()

    # net_tpu = FastLinear(w0_tpu, None)
    net_tpu = FastLinear(w0_tpu, b0_tpu)
    out_tpu = net_tpu(x_tpu)
    print(f"out_tpu 输出形状: {out_tpu.shape}")
    print(f"w0_tpu 输出形状: {w0_tpu.shape}")

    out_tpu = out_tpu.to("cpu")
    print(out_tpu)
    out_cpu = out_cpu.float().flatten()
    out_tpu = out_tpu.float().flatten()
    out_diff = out_cpu - out_tpu
    ratio = abs(out_diff/torch.max(abs(out_cpu), abs(out_tpu)))
    key = torch.argmax(ratio)
    print(out_tpu[key], out_cpu[key], ratio[key])
    cosm = cos_sim(out_cpu.numpy(), out_tpu.numpy())
    print(cosm)
    return


if __name__ == "__main__":
    check_mlp()