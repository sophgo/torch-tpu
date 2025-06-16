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
            def forward(self, dq_tpu, dk_tpu, dv_tpu):
                return torch.concatenate((dq_tpu, dk_tpu, dv_tpu), dim=-1)
    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    batch = 2
    sequence = 8
    hidden_size = 768
    #list is also acceptable
    input_data = [
         [ torch.rand(batch,sequence,hidden_size),torch.rand(batch,sequence,hidden_size),torch.rand(batch,sequence,hidden_size)],
    ]
    metric_table = ['max_diff','MAE']
    chip_arch_dict = {"bm1684x":1, 'sg2260':1}
    epsilon_dict = {'bm1684x':{'f32':1e-6,'f16':1e-2},'sg2260':{'f32':1e-6,'f16':1e-2}}
    case_name =  __file__.split('.py')[0]# You can change your name
    dump_flag = True #it will dump alll wrong cases

    device = torch.device("privateuseone:0")
    My_Tester = Tester_Basic(case_name, chip_arch_dict, device, metric_table, epsilon_dict,seed, dump_flag)
    return My_Tester.Torch_Test_Execution_Function(Test_Module(), input_data)

def case2(): # test empty tensor
    #step0:set seed
    seed=1000
    set_bacis_info(seed)

    #step1: define test model
    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
            def forward(self, a, b):
                return torch.cat([a, b[..., 128:]], dim=-1)
    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    #list is also acceptable
    input_data = [
         [ torch.randn(128, 1, 28, 128), torch.rand(128, 1, 28, 128)],
    ]
    metric_table = ['max_diff','MAE']
    chip_arch_dict = {"bm1684x":1, 'sg2260':1}
    epsilon_dict = {'bm1684x':{'f32':1e-6,'f16':1e-2},'sg2260':{'f32':1e-6,'f16':1e-2}}
    case_name =  __file__.split('.py')[0]# You can change your name
    dump_flag = True #it will dump alll wrong cases

    device = torch.device("privateuseone:0")
    My_Tester = Tester_Basic(case_name, chip_arch_dict, device, metric_table, epsilon_dict,seed, dump_flag)
    return My_Tester.Torch_Test_Execution_Function(Test_Module(), input_data)

def case3():
    '''
    tensor concat batch
    '''

    seed=1000
    set_bacis_info(seed)

    class Test_Module(nn.Module):
        def __init__(self):
            super(Test_Module, self).__init__()
        def forward(self, a, b):
            return torch.cat([a, b], dim=0)

    input_data = [
        [ torch.randn(9472, 3584), torch.rand(9472, 3584)],
    ]

    metric_table = ['max_diff','MAE']
    chip_arch_dict = {"bm1684x":1, 'sg2260':1}
    epsilon_dict = {'bm1684x':{'f32':1e-6,'f16':1e-2},'sg2260':{'f32':1e-6,'f16':1e-2}}
    case_name =  __file__.split('.py')[0]
    dump_flag = True

    device = torch.device("privateuseone:0")
    My_Tester = Tester_Basic(case_name, chip_arch_dict, device, metric_table, epsilon_dict,seed, dump_flag)
    return My_Tester.Torch_Test_Execution_Function(Test_Module(), input_data)

if __name__ == "__main__":
    case1()
    case2()
    case3()

#######################
##  case1():forward [[T,T]]
########################