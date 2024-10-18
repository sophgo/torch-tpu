import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():
    n, m = 10, 10
    # 生成实部
    real = torch.arange(n).unsqueeze(1).expand(n, m) * m + torch.arange(m).unsqueeze(0).expand(n, m)
    real = real.float()

    # 生成虚部为负的实部
    imag = -real

    # 通过 real 和 imag 生成复数 tensor
    complex_tensor = torch.complex(real, imag)
    #print("conj start:")
    complex_tensor_tpu = complex_tensor.to(device)
    #complex_tensor_out = torch.ops.prims.imag(complex_tensor_tpu).cpu()
    complex_tensor_out = torch.ops.prims.conj(complex_tensor_tpu).cpu()
    print("Input:\n",complex_tensor)
    print("Output:\n",complex_tensor_out,"\nOutput size:\n",complex_tensor_out.size(),"\nOutput dtype:\n",complex_tensor_out.dtype)

def case2():
    complextensor=torch.tensor(1+2j)
    print("Input:\n",complextensor,"\nInput size:\n",complextensor.size(),"\nInput dim:\n",complextensor.dim())
    complextensor_tpu=complextensor.to(device)
    print("Output:\n",torch.ops.prims.conj(complextensor).cpu())


if __name__ == "__main__":
    print("case1:\n")
    case1()
    print("\ncase2:\n")
    case2()