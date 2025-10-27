import torch
import torch.nn as nn
import torch.nn.functional as F
from top_utest import set_bacis_info, Tester_Basic, TensorComparator
import sys

def case1():
    #step0:set seed
    seed=1000
    set_bacis_info(seed)

    #step1: define test model
    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
            def forward(self, inp):
                return torch.sum(inp, dim=0, keepdim=False)
    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    input_data = [
        #  [torch.rand((999,1111))], #case1
        #  [torch.rand((32,256,768))], #case2
        [torch.rand((32,64,213))], #case2
    ]
    metric_table = ['max_diff','MAE']
    chip_arch_dict = {"bm1684x":1, 'sg2260':1}
    epsilon_dict = {'bm1684x':{'f32':1e-5,'f16':1e-2},'sg2260':{'f32':1e-5,'f16':1e-2}}
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
            def forward(self, inp):
                return torch.prod(inp, dim=3, keepdim=False)
    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    input_data = [
        #  [torch.rand((999,1111))], #case1
        #  [torch.rand((32,256,768))], #case2
        [-2 + (2 - (-2)) * torch.rand(5, 3, 35, 55)], #case2
    ]
    metric_table = ['max_diff','MAE']
    chip_arch_dict = {"bm1684x":1, 'sg2260':1}
    epsilon_dict = {'bm1684x':{'f32':1e-5,'f16':1e-2},'sg2260':{'f32':1e-5,'f16':1e-2}}
    case_name =  __file__.split('.py')[0]# You can change your name
    dump_flag = True #it will dump alll wrong cases

    device = torch.device("privateuseone:0")
    My_Tester = Tester_Basic(case_name, chip_arch_dict, device, metric_table, epsilon_dict,seed, dump_flag)
    return My_Tester.Torch_Test_Execution_Function(Test_Module(), input_data)


def case3():
    seed=1000
    set_bacis_info(seed)
    device = torch.device("privateuseone:0")
    a = torch.rand((1, 8, 8, 40, 128), dtype=torch.float16, device=device)
    b = torch.sum(a, dim=[1,2], keepdim=False)
    b_cpu = torch.sum(a.float().cpu(), dim=[1,2], keepdim=False)

    c = torch.sum(a, dim=[1,2], keepdim=True)
    c_cpu = torch.sum(a.float().cpu(), dim=[1,2], keepdim=True)

    d = torch.sum(a, dim=1, keepdim=False)
    d_cpu = torch.sum(a.float().cpu(), dim=1, keepdim=False)

    cmp = TensorComparator()
    status1 = cmp.cmp_result(b.float().cpu(), b_cpu)
    status2 = cmp.cmp_result(c.float().cpu(), c_cpu)
    status3 = cmp.cmp_result(d.float().cpu(), d_cpu)
    print(f"status1: {status1}, status2: {status2}, status3: {status3}")
    if status1 and status2 and status3:
        print("case3 passed")
    else:
        print("case3 failed")
        sys.exit(255)

if __name__ == "__main__":
    case1()
    case2()
    case3()

#######################
##  case1():forward[[T]]
########################
