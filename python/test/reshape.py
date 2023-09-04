import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case1():

    a_cpu = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print('a_cpu : ', a_cpu)
    print('a_cpu.shape(2, 6) : ', a_cpu.reshape(2, 6))
    print('a_cpu : ', a_cpu)
    print('a_cpu.shape(1, 12) : ', a_cpu.reshape(1, 12))
    print('a_cpu.shape(12) : ', a_cpu.reshape(12))
    print('a_cpu.shape(-1, 3) : ', a_cpu.reshape(-1, 3))
    print('a_cpu.shape(1, 2, 6) : ', a_cpu.reshape(2, 1, 6))
    print('a_cpu.shape(2, 2, 1, 1, 3) : ', a_cpu.reshape(2, 2, 1, 1, 3))

    a_tpu = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]]).to(device)
    print('a_tpu : ', a_tpu.cpu())
    print('a_tpu.shape(2, 6) : ', torch.reshape(a_tpu, (2, 6)).cpu())
    print('a_tpu : ', a_tpu.cpu())
    print('a_tpu.shape(1, 12) : ', a_tpu.reshape(1, 12).cpu())
    print('a_tpu.shape(12) : ', a_tpu.reshape(12).cpu())
    print('a_tpu.shape(-1, 3) : ', a_tpu.reshape(-1, 3).cpu())
    print('a_tpu.shape(1, 2, 6) : ', a_tpu.reshape(2, 1, 6).cpu())
    print('a_tpu.shape(2, 2, 1, 1, 3) : ', a_tpu.reshape(2, 2, 1, 1, 3).cpu())


if __name__ == "__main__":
    case1()