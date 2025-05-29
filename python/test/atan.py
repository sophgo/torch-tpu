import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():
    len = 100
    input_origin=torch.randn(len)*1
    o  = torch.atan(input_origin)


    input_tpu=input_origin.to(device)
    o_t=torch.atan(input_tpu)
    diff = abs(o - o_t.cpu())
    # print(f'input          : {input_origin}')
    # print(f'o              : {o}')    
    # print("output_tpu_aten : ", o_t.cpu())
    # print("delta_aten      : ", abs(o - o_t.cpu()))
    
    for i in range(len):
        print(f'{input_origin[i]}, {o[i]}, {o_t[i]}, {diff[i]}')
    print(torch.max(diff))
if __name__ == "__main__":
    case1()