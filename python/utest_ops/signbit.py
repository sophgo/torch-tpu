import torch
import torch.nn as nn
import torch.nn.functional as F
from top_utest import set_requires_grad, move_to, set_bacis_info, Tester_Basic

def case1():
    #step0:set seed
    seed = 1000
    set_bacis_info(seed)

    #step1: define test model

    class Test_Module(nn.Module):
        def __init__(self):
            super(Test_Module, self).__init__()

        def forward(self, input_tensor,sequence,hidden_size):
            b=torch.signbit(input_tensor)
            a=torch.zeros(sequence,hidden_size)
            for i in range(sequence):
                for j in range(hidden_size):
                    if a[i,j]:
                        a[i,j]=1
            return a
    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    sequence = 5257
    hidden_size = 768

    input_data = [
        [torch.randn(sequence, hidden_size),sequence,hidden_size],
    ]
    metric_table = ['max_diff', 'MAE']
    chip_arch_dict = {"bm1684x": 1, 'sg2260': 1}
    epsilon_dict = {'bm1684x': {'f32': 1e-6, 'f16': 1e-2}, 'sg2260': {'f32': 1e-6, 'f16': 1e-2}}
    case_name = __file__.split('.py')[0]
    dump_flag = True
    device = torch.device("privateuseone:0")

    My_Tester = Tester_Basic(case_name, chip_arch_dict, device, metric_table, epsilon_dict,seed, dump_flag)
    return My_Tester.Torch_Test_Execution_Function(Test_Module(), input_data)

if __name__ == "__main__":
    #This example shows all  [], [[]] is acceptable
    case1()

#######################
##  case1():forward + backward [[T]]
########################
