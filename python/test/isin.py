import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():
    """isin.Tensor_Tensor_out
    """
    inp    =  torch.tensor([[1, 2], [3, 4]])
    inp2   =  torch.tensor([2, 3])
    inp_tpu = inp.to(device)
    inp2_tpu = inp2.to(device)

    o_cpu = torch.isin(inp, inp2)
    o_tpu = torch.isin(inp_tpu, inp2_tpu)

    # diff = o_cpu - o_tpu.cpu()
    # print(torch.max(abs(diff)))
    import pdb; pdb.set_trace()

def case2():
    """isin.Tensor_Scalar_out
    """
    inp    =  torch.tensor([[1, 2], [3, 4]])
    inp2   =  torch.tensor([2, 3])
    inp_tpu = inp.to(device)
    inp2_tpu = inp2.to(device)

    o_cpu = torch.isin(inp, inp2)
    o_tpu = torch.isin(inp_tpu, inp2_tpu)

    # diff = o_cpu - o_tpu.cpu()
    # print(torch.max(abs(diff)))
    import pdb; pdb.set_trace()


def case3():
    inp    =  torch.tensor([[1, 2], [3, 4]])
    inp2   =  2
    inp_tpu = inp.to(device)
    inp2_tpu = inp2.to(device)

    o_cpu = torch.isin(inp, inp2)
    o_tpu = torch.isin(inp_tpu, inp2_tpu)

    # diff = o_cpu - o_tpu.cpu()
    # print(torch.max(abs(diff)))
    import pdb; pdb.set_trace()

if __name__ == "__main__":
    case1()