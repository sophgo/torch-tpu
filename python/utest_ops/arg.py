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
            def forward(self, input_origin, dim):
                return torch.ops.aten.min.dim(input_origin,dim)[0]

    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    #Of course , 0-dim value is acceptable
    dim = 4
    shape = [10,10,20,10,2,4,5]
    input_origin = (torch.rand(shape)*(-200)+10)
    #list is also acceptable
    #Of course , 0-dim value is acceptable
    input_data = [
         [input_origin,dim]
    ]
    metric_table = ['max_diff','MAE']
    chip_arch_dict = {"bm1684x":1, 'sg2260':1}
    epsilon_dict = {'bm1684x':{'f32':1e-6},'sg2260':{'f32':1e-6}}
    case_name =  __file__.split('.py')[0]# You can change your name
    dump_flag = True #it will dump alll wrong cases
    device = torch.device("privateuseone:0")

    My_Tester = Tester_Basic(case_name, chip_arch_dict, device, metric_table, epsilon_dict,seed, dump_flag)
    return My_Tester.Torch_Test_Execution_Function(Test_Module(), input_data)

def case2():
    #step0:set seed
    seed=1000
    set_bacis_info(seed)

    #step1: define test model
    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
            def forward(self, input_origin, dim):
                return torch.cat(torch.ops.aten.max.dim(input_origin,dim),0)

    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    #Of course , 0-dim value is acceptable
    dim = 1
    shape = [10,50,20,10]
    input_origin = (torch.rand(shape)*(-20000)+10000)
    #list is also acceptable
    #Of course , 0-dim value is acceptable
    input_data = [
         [input_origin,dim]
    ]
    metric_table = ['max_diff','MAE']
    chip_arch_dict = {"bm1684x":1, 'sg2260':1}
    epsilon_dict = {'bm1684x':{'f32':1e-6},'sg2260':{'f32':1e-6}}
    case_name =  __file__.split('.py')[0]# You can change your name
    dump_flag = True #it will dump alll wrong cases
    device = torch.device("privateuseone:0")

    My_Tester = Tester_Basic(case_name, chip_arch_dict, device, metric_table, epsilon_dict,seed, dump_flag)
    return My_Tester.Torch_Test_Execution_Function(Test_Module(), input_data)

def case3():
    #step0:set seed
    seed=1000
    set_bacis_info(seed)

    #step1: define test model
    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
            def forward(self, a1):
                return a1.argmax(-1)

    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    #Of course , 0-dim value is acceptable

    input_origin = torch.randint(0, 1000, (1,10))

    #list is also acceptable
    #Of course , 0-dim value is acceptable
    input_data = [
         [input_origin]
    ]
    metric_table = ['max_diff','MAE']
    chip_arch_dict = {"bm1684x":1, 'sg2260':1}
    epsilon_dict = {'bm1684x':{'f32':1e-6,'f16':1e-2},'sg2260':{'f32':1e-6,'f16':1e-2}}
    case_name =  __file__.split('.py')[0]# You can change your name
    dump_flag = True #it will dump alll wrong cases
    device = torch.device("privateuseone:0")

    My_Tester = Tester_Basic(case_name, chip_arch_dict, device, metric_table, epsilon_dict,seed, dump_flag)
    return My_Tester.Torch_Test_Execution_Function(Test_Module(), input_data)

def case4():
    #step0:set seed
    seed=1000
    set_bacis_info(seed)

    #step1: define test model
    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
            def forward(self, a1,dim):
                return torch.argmin(a1,dim)

    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    #Of course , 0-dim value is acceptable
    dim = 0
    input_origin=torch.rand(2,4,2,4,4,6)

    #list is also acceptable
    #Of course , 0-dim value is acceptable
    input_data = [
         [input_origin,dim]
    ]
    metric_table = ['max_diff','MAE']
    chip_arch_dict = {"bm1684x":1, 'sg2260':1}
    epsilon_dict = {'bm1684x':{'f32':1e-6},'sg2260':{'f32':1e-6}}
    case_name =  __file__.split('.py')[0]# You can change your name
    dump_flag = True #it will dump alll wrong cases
    device = torch.device("privateuseone:0")

    My_Tester = Tester_Basic(case_name, chip_arch_dict, device, metric_table, epsilon_dict,seed, dump_flag)
    return My_Tester.Torch_Test_Execution_Function(Test_Module(), input_data)



if __name__ == "__main__":
    case1()
    case2()
    case3()
    case4()

#######################
##  case1():forward  [[value]]
##
##
########################