import torch
import torch.nn as nn
import torch.nn.functional as F
from top_utest import set_bacis_info, Tester_Basic

def case1():
    #step0:set seed
    seed=1000
    set_bacis_info(seed)

    #step1: define test model
    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
            def forward(self, casual_mask, attn_weights, mask_value):
                return torch.where(casual_mask, attn_weights, mask_value)
    batch = 1
    sequence = 8
    head_size = 3
    max_position = 8
    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    input_data = [
         [ torch.rand(batch, head_size, sequence, sequence).bool(), torch.tensor(-1e4) , torch.tril(torch.ones((max_position, max_position),dtype=torch.uint8)) \
                        .view(1,1,max_position, max_position)],
    ]
    metric_table = ['max_diff','MAE']
    epsilon_dict = {'f32':1e-6,'f16':1e-2}
    case_name =  __file__.split('.py')[0]
    dump_flag = True #it will dump alll wrong cases

    device = torch.device("privateuseone:0")
    My_Tester = Tester_Basic(case_name, device, metric_table, epsilon_dict,seed, dump_flag)
    return My_Tester.Torch_Test_Forward_Function(Test_Module, input_data)

if __name__ == "__main__":
    case1()