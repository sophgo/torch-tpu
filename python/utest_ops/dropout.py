import torch
import torch.nn as nn
import torch.nn.functional as F
from top_utest import set_bacis_info, Tester_Basic

def case1():
    #step0:set seed
    seed=1000
    set_bacis_info(seed)

    #step1: define test model
    Batch = 8
    nHead = 12
    Hidden = 768
    sequence = 1024
    p_drop = 0.5
    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
                self.drop = nn.Dropout(p_drop,inplace=True)
            def forward(self, x):
                return self.drop(x)


    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    input_data = {
         "simple0": [ torch.randn((Batch, nHead, sequence, sequence))]
    }
    #list is also acceptable
    input_data = [
         [  torch.randn((Batch, nHead, sequence, sequence))],
    ]
    metric_table = ['max_diff','MAE']
    epsilon_dict = {'f32':12}
    case_name =  __file__.split('.py')[0]# You can change your name
    dump_flag = True #it will dump alll wrong cases

    device = torch.device("privateuseone:0")
    My_Tester = Tester_Basic(case_name, device, metric_table, epsilon_dict,seed, dump_flag)
    return My_Tester.Torch_Test_Forward_Function(Test_Module, input_data)


if __name__ == "__main__":
    case1()