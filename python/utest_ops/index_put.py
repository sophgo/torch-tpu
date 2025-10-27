import torch
import torch.nn as nn
from top_utest import set_bacis_info, Tester_Basic

def case1():
    """Test basic index_put with replace mode"""
    seed = 1000
    set_bacis_info(seed)
    
    class Test_Module(nn.Module):
        def __init__(self):
            super(Test_Module, self).__init__()
        
        def forward(self, self_tensor, idx, value):
            self_tensor[idx] = value
            return self_tensor
    
    self_tensor = torch.zeros(5).float()
    idx = torch.tensor([1, 3], dtype=torch.int32)
    value = torch.tensor([10.0, 20.0]).float()
    
    input_data = [[self_tensor, idx, value]]
    metric_table = ['max_diff', 'MAE']
    chip_arch_dict = {"bm1684x": 0, 'sg2260': 1}
    epsilon_dict = {'bm1684x': {'f32': 1e-6, 'f16': 1e-2}, 'sg2260': {'f32': 1e-6, 'f16': 1e-2}}
    case_name = __file__.split('.py')[0]
    dump_flag = True
    device = torch.device("privateuseone:0")
    
    My_Tester = Tester_Basic(case_name, chip_arch_dict, device, metric_table, epsilon_dict, seed, dump_flag)
    return My_Tester.Torch_Test_Execution_Function(Test_Module(), input_data)

def case2():
    """Test index_put with accumulate mode"""
    seed = 1000
    set_bacis_info(seed)
    
    class Test_Module(nn.Module):
        def __init__(self):
            super(Test_Module, self).__init__()
        
        def forward(self, self_tensor, idx, value):
            self_tensor.index_put_([idx], value, accumulate=True)
            return self_tensor
    
    self_tensor = torch.ones(5).float()
    idx = torch.tensor([1, 1, 1], dtype=torch.int32)
    value = torch.tensor([1.0, 2.0, 3.0]).float()
    
    input_data = [[self_tensor, idx, value]]
    metric_table = ['max_diff', 'MAE']
    chip_arch_dict = {"bm1684x": 0, 'sg2260': 1}
    epsilon_dict = {'bm1684x': {'f32': 1e-6, 'f16': 1e-2}, 'sg2260': {'f32': 1e-6, 'f16': 1e-2}}
    case_name = __file__.split('.py')[0] + "_case2"
    dump_flag = True
    device = torch.device("privateuseone:0")
    
    My_Tester = Tester_Basic(case_name, chip_arch_dict, device, metric_table, epsilon_dict, seed, dump_flag)
    return My_Tester.Torch_Test_Execution_Function(Test_Module(), input_data)

def case3():
    """Test multi-dimensional index_put"""
    seed = 1000
    set_bacis_info(seed)
    
    class Test_Module(nn.Module):
        def __init__(self):
            super(Test_Module, self).__init__()
        
        def forward(self, self_tensor, i0, i1, value):
            self_tensor[i0, i1] = value
            return self_tensor
    
    self_tensor = torch.zeros(5, 3).float()
    i0 = torch.tensor([0, 2, 4], dtype=torch.int32)
    i1 = torch.tensor([0, 1, 2], dtype=torch.int32)
    value = torch.tensor([1.0, 2.0, 3.0]).float()
    
    input_data = [[self_tensor, i0, i1, value]]
    metric_table = ['max_diff', 'MAE']
    chip_arch_dict = {"bm1684x": 0, 'sg2260': 1}
    epsilon_dict = {'bm1684x': {'f32': 1e-6, 'f16': 1e-2}, 'sg2260': {'f32': 1e-6, 'f16': 1e-2}}
    case_name = __file__.split('.py')[0] + "_case3"
    dump_flag = True
    device = torch.device("privateuseone:0")
    
    My_Tester = Tester_Basic(case_name, chip_arch_dict, device, metric_table, epsilon_dict, seed, dump_flag)
    return My_Tester.Torch_Test_Execution_Function(Test_Module(), input_data)

if __name__ == "__main__":
    case1()
    case2()
    case3()

