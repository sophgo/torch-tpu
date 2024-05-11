import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import copy
from top_utest import TensorComparator

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

class Test_Module(nn.Module):
        def __init__(self):
            super(Test_Module, self).__init__()
        def forward(self, q, k):
            return torch.matmul(q, k)
            
def check_llama_matmul(B, M, K, N):

    net_cpu = Test_Module()
    left = torch.rand([B, M, K], requires_grad=False)
    right = torch.rand([B, K, N], requires_grad=False)
    left_tpu = copy.deepcopy(left).to(device).half()
    right_tpu = copy.deepcopy(right).to(device).half()
    out_cpu = net_cpu(left, right)

    net_tpu = copy.deepcopy(net_cpu).to(device).half()

    out_tpu = net_tpu(left_tpu, right_tpu).float().to("cpu")

    comparator = TensorComparator()
    status = comparator.cmp_result(out_cpu.detach(), out_tpu.cpu().detach().float())

    return status


if __name__ == "__main__":
    #This example shows all  [], [[]] is acceptable
    test_dict = {
         'MM_QKV' : [1, 16, 8192, 1280],
         'Attention_FC' : [1, 16, 1024, 8192],
        # 'Input Embedding' : [1, 16, 32000, 8192],
        # 'Output Embedding' : [1, 16, 8192, 32000]
    }

    for case_name in test_dict:
        B = test_dict[case_name][0]
        M = test_dict[case_name][1]
        K = test_dict[case_name][2]
        N = test_dict[case_name][3]

        status = check_llama_matmul(B, M, K, N)
        if status == -1:
            print(f"[Failed] llama_{case_name} compare failed!")
            sys.exit(255)
        print(f"[Success] llama_{case_name} compare pass!")