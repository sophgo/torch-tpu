import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():

    # tensor_complex = torch.tensor([[1 + 2j, 3 + 4j, 5 + 6j],
    #                                [1 + 2j, 3 + 4j, 5 + 6j],
    #                                [1 + 2j, 3 + 4j, 5 + 6j],
    #                                [1 + 2j, 3 + 4j, 5 - 6j]], dtype=torch.complex64)
    # tensor_complex = torch.tensor([1 + 2j, 3 + 4j], dtype=torch.complex64)
    # # Generate a complex tensor with shape (100, 100)
    # tensor_complex= torch.randn(100, 100, dtype=torch.complex64)

    # #tensor_complex = torch.tensor([[1 ,2, 3, 4, 5, 6, 7, 8, 9, 10]], dtype=torch.complex64)
    # print("Original complex tensor:", tensor_complex,"\n",tensor_complex.stride(),tensor_complex.size())
    # tensor_complex_tpu = tensor_complex.to(device)
    # output_tpu=torch.ops.prims.real(tensor_complex_tpu).cpu()
    # print("real : ",output_tpu ,output_tpu.dtype)
    # a = torch.rand(2,2)
    # a_tpu = a.to(device)
    # b_tpu = a_tpu.contiguous().cpu()
    # print("b_tpu:", b_tpu)

    # # Define the dimensions
    # n, m = 10, 10

    # # Create real and imaginary parts
    # real = torch.arange(n).unsqueeze(1).expand(n, m).float()  # a in each row
    # imag = -torch.arange(m).unsqueeze(0).expand(n, m).float() # -b in each column

    # # Convert to complex64 tensor
    # tensor2 = torch.complex(real, imag)
    # tensor2_tpu = tensor2.to(device)
    # outtensor2 = torch.ops.prims.imag(tensor2_tpu).cpu()
    # #print("tensor2real:",outtensor2)

    # print(tensor2,"end\n")
    # 
    n, m = 10, 10

    # 生成实部
    real = torch.arange(n).unsqueeze(1).expand(n, m) * m + torch.arange(m).unsqueeze(0).expand(n, m)
    real = real.float()

    # 生成虚部为负的实部
    imag = -real

    # 通过 real 和 imag 生成复数 tensor
    complex_tensor = torch.complex(real, imag)
    complex_tensor_tpu = complex_tensor.to(device)
    complex_tensor_out = torch.ops.prims.real(complex_tensor_tpu).cpu()
    #complex_tensor_out = torch._conj(complex_tensor_tpu).cpu()
    print("Input:\n",complex_tensor)
    print("Output:\n",complex_tensor_out,"\nOutput size:\n",complex_tensor_out.size(),"\nOutput dtype:\n",complex_tensor_out.dtype)
def case2():
    complextensor=torch.tensor(1+2j)
    print("Input:\n",complextensor,"\nInput size:\n",complextensor.size(),"\nInput dim:\n",complextensor.dim())
    complextensor_tpu=complextensor.to(device)
    print("Output:\n",torch.ops.prims.real(complextensor).cpu())


if __name__ == "__main__":
    print("case1:\n")
    case1()
    print("\ncase2:\n")
    case2()