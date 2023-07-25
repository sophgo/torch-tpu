import torch
import torch.nn as nn
import torch.nn.functional as F
from top_utest import set_requires_grad,move_to, set_bacis_info, Tester_Basic

def case1():
    #step0:set seed
    seed=1000
    set_bacis_info(seed)

    #step1: define test model
    batch = 2
    sequence = 8
    hidden_size = 3
    layer_norm_epsilon = 1e-5
    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
            def forward(self, x, label):
                return x[..., :-1, :].contiguous(),label[..., 1:].contiguous()
    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    inp = torch.ones(batch, sequence, hidden_size)
    label = torch.randint(0, hidden_size, (batch, sequence)).float()
    input_data = [
         [ inp, label],
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
        set_requires_grad(input_sample_cpu, True)
        set_requires_grad(input_sample_tpu, True)

        output_cpu = net_cpu(*input_sample_cpu)
        output_tpu = net_tpu(*input_sample_tpu)
        #tpu_first
        return output_tpu, output_cpu

        loss_fct = nn.CrossEntropyLoss()
        loss_cpu = loss_fct(output_cpu[0].view(-1, hidden_size), output_cpu[1].view(-1))
        loss_tpu = loss_fct(output_tpu[0].view(-1, hidden_size), output_tpu[1].view(-1))


        loss_cpu.backward()
        loss_tpu.backward()
        #tpu_first
        return input_sample_tpu[0].grad , input_sample_cpu[0].grad#Notice [0] because input_data has [],[]

    My_Tester = Tester_Basic(case_name, device, metric_table, epsilon_dict,seed, dump_flag)
    My_Tester.customized_execute_function = customized_execute_function
    return My_Tester.Torch_Test_Forward_Function(Test_Module(), input_data)


if __name__ == "__main__":
    #This example shows all  [], [[]] is acceptable
    case1()

#######################
##  case1():forward[[T]]
########################