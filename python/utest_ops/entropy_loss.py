import torch
import torch.nn as nn
import torch.nn.functional as F
from top_utest import set_requires_grad,move_to, set_bacis_info, Tester_Basic

def case1():
    #step0:set seed
    seed=1000
    set_bacis_info(seed)

    #step1: define test model
    batch = 32
    sequence = 8
    vtable_size = 4
    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
                self.loss_fct = nn.CrossEntropyLoss()
            def forward(self, inp, label):
                return self.loss_fct(inp.view(-1, vtable_size), label.view(-1)) 
    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    input_data = [
         [ torch.ones(batch, sequence, vtable_size), torch.randint(0, vtable_size, (batch, sequence))],
    ]
    metric_table = ['max_diff','MAE']
    epsilon_dict = {'f32':1e-6}
    case_name =  __file__.split('.py')[0]# You can change your name
    dump_flag = True #it will dump alll wrong cases
    device = torch.device("privateuseone:0")

    # you can write your own forward and backward fucntion
    #assuming net_tpu and input_sample_tpu has loaded on tpu
    #In this way you can define backward propagation by your self.
    def customized_execute_function(input_sample_cpu, input_sample_tpu, net_cpu, net_tpu, dtype):
        #  only Tensors of floating point and complex dtype can require gradients
        # set_requires_grad(input_sample_cpu, True)
        # set_requires_grad(input_sample_tpu, True)

        output_cpu = net_cpu(*input_sample_cpu)
        output_tpu = net_tpu(*input_sample_tpu)

        output_cpu.backward()
        output_tpu.backward()
        #tpu-first
        return input_sample_tpu[0].grad, input_sample_cpu[0].grad #Notice [0] because input_data has [],[]
    
    My_Tester = Tester_Basic(case_name, device, metric_table, epsilon_dict,seed, dump_flag)
    My_Tester.customized_execute_function = customized_execute_function
    # return My_Tester.Torch_Test_Forward_Function(Test_Module, input_data)


if __name__ == "__main__":
    #This example shows all  [], [[]] is acceptable
    case1() 
