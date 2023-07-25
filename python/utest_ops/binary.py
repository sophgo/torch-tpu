

import torch
import torch.nn as nn
import torch.nn.functional as F
from top_utest import set_requires_grad, move_to, set_bacis_info, Tester_Basic

def case1():
    #step0:set seed
    seed=1000
    set_bacis_info(seed)
    #step1: define test model
    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
            def forward(self,  x0, x1):
                y = x0 + x1
                y = y + x0
                y = y + 3.0
                y = y * 4.0
                y = 2.0 - y
                return y

    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    input_data = {
         "simple0": [torch.rand(6, 1024, 1, 768), torch.rand(1, 1024, 1, 768)],
    }
    # list is also acceptable
    # if backward , input_cpu you can not set requires_grad here
    # otherwise input_tpu will be a leaf node without grad!
    input_data = [
        [torch.rand(6, 1024, 1, 768), torch.rand(1, 1024, 1, 768)],
    ]
    metric_table = ['max_diff','MAE']
    epsilon_dict = {'f32':1e-6, 'f16':1e-2}
    case_name =  __file__.split('.py')[0]# You can change your name
    dump_flag = True #it will dump alll wrong cases
    device = torch.device("privateuseone:0")

    My_Tester = Tester_Basic(case_name, device, metric_table, epsilon_dict,seed, dump_flag)
    return My_Tester.Torch_Test_Forward_Function(Test_Module(), input_data)


if __name__ == "__main__":
    case1()

#######################
##  case1():forward [[T,T]]
########################