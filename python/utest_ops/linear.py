import torch
import torch.nn as nn
import torch.nn.functional as F
from top_utest import set_requires_grad,move_to, set_bacis_info, Tester_Basic
from torch.optim import AdamW, Adam


def case1():
    #step0:set seed
    seed=1000
    set_bacis_info(seed)

    #step1: define test model
    batch = 8
    sequence = 1024
    hidden_size = 768
    out_size = 3

    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
                self.linear = nn.Linear(hidden_size, out_size)
            def forward(self, x):
                return self.linear(x)


    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    input_data = {
         "simple0": [ torch.rand(batch, sequence, hidden_size)]
    }
    #list is also acceptable
    input_data = [
         [  torch.rand(batch, sequence, hidden_size)],
    ]
    metric_table = ['max_diff','MAE']
    chip_arch_dict = {"bm1684x":0, 'sg2260':0}
    epsilon_dict = {'bm1684x':{'f32':1e-6,'f16':1e-2},'sg2260':{'f32':1e-6,'f16':1e-2}}
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

        grad_o = torch.rand(output_cpu.shape)
        grad_o_tpu =  move_to(grad_o, device, dtype)  #grad_o.to(device)

        output_cpu.backward(grad_o)
        output_tpu.backward(grad_o_tpu)


        optimizer_cpu = Adam(net_cpu.parameters(), lr = 1)
        optimizer_tpu = Adam(net_tpu.parameters(), lr = 1)
        optimizer_cpu.zero_grad()
        optimizer_tpu.zero_grad()

        optimizer_cpu.step()
        optimizer_tpu.step()
        #tpu first
        return input_sample_tpu[0].grad, input_sample_cpu[0].grad

    My_Tester = Tester_Basic(case_name, chip_arch_dict, device, metric_table, epsilon_dict,seed, dump_flag)
    My_Tester.customized_execute_function = customized_execute_function
    return My_Tester.Torch_Test_Execution_Function(Test_Module(), input_data)

if __name__ == "__main__":
    case1() #[[T]] as input

#######################
##  case1():forward + Optimizer+ step +backward [[T]]
########################
