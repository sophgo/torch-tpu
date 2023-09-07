import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from top_utest import set_bacis_info, Tester_Basic

def test_native_group_norm():
    #step0:set seed
    seed=1000
    set_bacis_info(seed)
    N = 2
    C = 2
    H = 3
    W = 3
    group_num = 2
    eps = 1e-5


    #step1: define test model
    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
            def forward(self, a1):
                return torch.native_group_norm(a1, None,None, N, C, H*W, group_num, eps)

    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    #list is also acceptable

    input_data = [
        [torch.randn(N, C, H, W)],
    ]
    metric_table = ['max_diff','MAE']
    chip_arch_dict = {"bm1684x":1, 'sg2260':0}
    epsilon_dict = {'bm1684x':{'f32':1e-6,'f16':1e-2},'sg2260':{'f32':1e-6,'f16':1e-2}}
    case_name =  __file__.split('.py')[0]# You can change your name
    dump_flag = True #it will dump alll wrong cases
    device = torch.device("privateuseone:0")

    My_Tester = Tester_Basic(case_name, chip_arch_dict, device, metric_table, epsilon_dict,seed, dump_flag)
    return My_Tester.Torch_Test_Execution_Function(Test_Module(), input_data)

if __name__ == "__main__":
    test_native_group_norm()

#######################
##  case1():forward [[T,T]]
########################