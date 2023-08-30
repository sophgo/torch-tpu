import torch
import torch.nn as nn
import torch.nn.functional as F
from top_utest import set_bacis_info, Tester_Basic

def case1():
    #step0:set seed
    seed=1000
    set_bacis_info(seed)

    #step1: define test model
    batch = 32
    head_num = 12
    hidden_size = 768
    sequence = 256
    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
            def forward(self, inp):
                    q,k,v = inp.split(hidden_size, -1)
                    q1 = q.view(batch, sequence, head_num, hidden_size // head_num)
                    q2 = q1.permute(0, 2, 1, 3)
                    q3 = q2.transpose(-1,-2)
                    q4 = q3.contiguous()
                    return q4
    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    input_data = [
         [torch.rand(batch, sequence, hidden_size * 3)],
    ]
    metric_table = ['max_diff','MAE']
    chip_arch_dict = {"bm1684x":1, 'sg2260':1}
    epsilon_dict = {'bm1684x':{'f32':1e-6,'f16':1e-2},'sg2260':{'f32':1e-6,'f16':1e-2}}
    case_name =  __file__.split('.py')[0]# You can change your name
    dump_flag = True #it will dump alll wrong cases

    device = torch.device("privateuseone:0")
    My_Tester = Tester_Basic(case_name, chip_arch_dict, device, metric_table, epsilon_dict,seed, dump_flag)
    return My_Tester.Torch_Test_Execution_Function(Test_Module(), input_data)


if __name__ == "__main__":
    case1()

#######################
##  case1():forward [[T,T]]
########################