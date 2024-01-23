import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def test_reshape():

    a_cpu = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print('a_cpu : ', a_cpu)
    print('a_cpu.shape(2, 6) : ', torch.reshape(a_cpu, (2, 6)))
    print('a_cpu : ', a_cpu)
    print('a_cpu.shape(12) : ', a_cpu.reshape(12))
    print('a_cpu.shape(-1, 3) : ', a_cpu.reshape(-1, 3))
    print('a_cpu.shape(1, 2, 6) : ', a_cpu.reshape(2, 1, 6))
    print('a_cpu.shape(2, 2, 1, 1, 3) : ', a_cpu.reshape(2, 2, 1, 1, 3).shape)

    a_tpu = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]).to(device)
    print('a_tpu : ', a_tpu.cpu())
    print('a_tpu.shape(2, 6) : ', torch.reshape(a_tpu, (2, 6)).cpu())
    print('a_tpu : ', a_tpu.cpu())
    print('a_tpu.shape(12) : ', a_tpu.reshape(12).cpu())
    print('a_tpu.shape(-1, 3) : ', a_tpu.reshape(-1, 3).cpu())
    print('a_tpu.shape(1, 2, 6) : ', a_tpu.reshape(2, 1, 6).cpu())
    print('a_tpu.shape(2, 2, 1, 1, 3) : ', a_tpu.reshape(2, 2, 1, 1, 3).cpu().shape)


if __name__ == "__main__":
    test_reshape()
