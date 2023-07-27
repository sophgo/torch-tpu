import torch
import torch.nn as nn
import torch.nn.functional as F
from top_utest import set_requires_grad,move_to, set_bacis_info, Tester_Basic

def case1():
    #step0:set seed
    seed=1000
    set_bacis_info(seed)

    #step1: define test model
    batch = 8
    sequence = 32#1024
    hidden_size = 128#3072
    layer_norm_epsilon = 1e-5
    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
                self.LayerNorm = nn.LayerNorm(hidden_size, eps=layer_norm_epsilon)
            def forward(self, a1):
                return self.LayerNorm(a1)
    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    input_data = [
         [ torch.randn(batch, sequence, hidden_size)],
    ]
    metric_table = ['max_diff','MAE']
    epsilon_dict = {'f32':1e-6, 'f16':1e-2}
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
        #tpu-first
        return input_sample_tpu[0].grad, input_sample_cpu[0].grad #Notice [0] because input_data has [],[]

    My_Tester = Tester_Basic(case_name, device, metric_table, epsilon_dict,seed, dump_flag)
    My_Tester.customized_execute_function = customized_execute_function
    return My_Tester.Torch_Test_Execution_Function(Test_Module(), input_data)


if __name__ == "__main__":
    #This example shows all  [], [[]] is acceptable
    case1()

#######################
##  case1():forward + backward [[T]]
########################