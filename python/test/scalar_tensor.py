import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def test_scalar_tensor():
    out_cpu = torch.scalar_tensor(5)
    print(out_cpu.device)
    print(out_cpu)

    out_tpu = torch.scalar_tensor(5, device=device)
    print(out_tpu.device)
    print(out_tpu.cpu())

if __name__ == '__main__':
    test_scalar_tensor()
