import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():
    index_num = 63;
    select_num = 98;
    inner_num = 10;
    outer_num = 3;
    dim = 1;
    assert(index_num <= select_num)
    input_shape = [outer_num, select_num, inner_num]
    add_shape   = [outer_num, index_num,  inner_num]
    x = torch.ones(input_shape, dtype=torch.float16)
    t = torch.randn(add_shape, dtype = torch.float16)
    index = torch.randint(0, select_num, (index_num,))

    print(index)
    print("============================")

    x0 = x.to(device)
    t0 = t.to(device)
    index0 = index.to(device)

    x.index_add_(dim, index, t)
    x0.index_add_(dim, index0, t0)
    x0 = x0.to("cpu")
    print(x0)
    print("============================")
    print(x)

    print (torch.max(abs(x0 - x)))

if __name__ == "__main__":
    case1()

