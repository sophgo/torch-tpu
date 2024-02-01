import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
device = "tpu:0"

def case1():

    # self = 13 * torch.rand((35,35,1,55), dtype=torch.float32)
    # other = 15 * torch.rand((35,35,25,55), dtype=torch.float32)

    self = 13 * torch.rand((25,55), dtype=torch.float32)
    other = 15 * torch.rand((35,35,25,55), dtype=torch.float32)

    # self = 13 * torch.rand((35,35,25,55), dtype=torch.float32)
    # other = 15 * torch.rand((35,35,25,55), dtype=torch.float32)

    # self = 13 * torch.rand((35,35,25,55), dtype=torch.float32)
    # other = torch.tensor(5., dtype=torch.float32)

    # self = torch.tensor(5., dtype=torch.float32)
    # other = 13 * torch.rand((35,35,25,55), dtype=torch.float32)

    self_tpu = self.to(device)
    other_tpu = other.to(device)
    cpu_res = self.hypot(other)
    tpu_res = self_tpu.hypot(other_tpu).cpu()
    print("origin self : ", self)
    print("origin other : ", other)
    print("cpu res : ", cpu_res)
    print("tpu res : ", tpu_res)
    print("max diff : ", torch.max(torch.div(torch.abs(cpu_res - tpu_res), cpu_res)))

if __name__ == "__main__":
    case1()