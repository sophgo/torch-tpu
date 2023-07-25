import torch
import torch.nn as nn
import torch.nn.functional as F
from top_utest import set_requires_grad,move_to, set_bacis_info, Tester_Basic

def case1():
    #step0:set seed
    seed=1000
    set_bacis_info(seed)

    #step1: define test model
    Batch = 1
    nHead = 1
    Hidden = 8
    sequence = 8
    p_drop = 0.5
    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
                self.drop = nn.Dropout(p_drop) #inplace=True
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
    # you can write your own forward and backward fucntion
    #assuming net_tpu and input_sample_tpu has loaded on tpu
    #In this way you can define backward propagation by your self.
    def customized_execute_function(input_sample_cpu, input_sample_tpu, net_cpu, net_tpu, dtype):
        set_requires_grad(input_sample_cpu, True)
        set_requires_grad(input_sample_tpu, True)

        output_cpu = net_cpu(*input_sample_cpu)
        output_tpu = net_tpu(*input_sample_tpu)

        grad_o = torch.ones(output_cpu.shape)
        grad_o_tpu =  move_to(grad_o, device, dtype)  #grad_o.to(device)

        output_cpu.backward(grad_o)
        output_tpu.backward(grad_o_tpu)
        #tpu first
        return input_sample_tpu[0].grad, input_sample_cpu[0].grad

    My_Tester = Tester_Basic(case_name, device, metric_table, epsilon_dict,seed, dump_flag)
    My_Tester.customized_execute_function = customized_execute_function
    return My_Tester.Torch_Test_Forward_Function(Test_Module(), input_data)

def case2():
    #step0:set seed
    seed=1000
    set_bacis_info(seed)

    #step1: define test model
    Batch = 1
    nHead = 1
    Hidden = 8
    sequence = 8
    p_drop = 0.5
    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
                self.drop = nn.Dropout(p_drop) #inplace=True
            def forward(self, x):
                return self.drop(x)


    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    input_data = {
         "simple0": [ torch.randn((Batch, nHead, sequence, sequence))]
    }
    #list is also acceptable
    input_data = [
           torch.randn((Batch, nHead, sequence, sequence)),
           torch.randn((Batch, nHead, sequence, sequence)),
    ]
    metric_table = ['max_diff','MAE']
    epsilon_dict = {'f32':12}
    case_name =  __file__.split('.py')[0]# You can change your name
    dump_flag = True #it will dump alll wrong cases

    device = torch.device("privateuseone:0")
    # you can write your own forward and backward fucntion
    #assuming net_tpu and input_sample_tpu has loaded on tpu
    #In this way you can define backward propagation by your self.
    def customized_execute_function(input_sample_cpu, input_sample_tpu, net_cpu, net_tpu, dtype):
        set_requires_grad(input_sample_cpu, True)
        set_requires_grad(input_sample_tpu, True)

        output_cpu = net_cpu(input_sample_cpu)
        output_tpu = net_tpu(input_sample_tpu)

        grad_o = torch.ones(output_cpu.shape)
        grad_o_tpu =  move_to(grad_o, device, dtype)  #grad_o.to(device)

        output_cpu.backward(grad_o)
        output_tpu.backward(grad_o_tpu)
        #tpu first
        return input_sample_tpu.grad, input_sample_cpu.grad

    My_Tester = Tester_Basic(case_name, device, metric_table, epsilon_dict,seed, dump_flag)
    My_Tester.customized_execute_function = customized_execute_function
    return My_Tester.Torch_Test_Forward_Function(Test_Module(), input_data)

if __name__ == "__main__":
    case1() #[[T]] as input
    # case2() #T as input

#######################
##  case1():forward + backward [[T]]
##  case2():forward + backward T
########################