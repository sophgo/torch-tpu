

import torch
import torch.nn as nn
import torch.nn.functional as F
from top_utest import set_requires_grad, move_to, set_bacis_info, Tester_Basic

def case1():
    #step0:set seed
    seed=1000
    set_bacis_info(seed)

    #step1: define test model
    batch = 2
    sequence = 8
    hidden_size = 768
    num_heads = 12
    attn_head_size = hidden_size // num_heads
    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
            def forward(self, inp):
                # a1,a2,a3 = inp.split(hidden_size, dim=2)
                # y = a1.contiguous()+a2.contiguous()+a3.contiguous()
                # return y

                # 2.split
                q,k,v = inp.split(hidden_size, dim=2)

                # 3.view
                new_shape = q.size()[:-1] + (num_heads, attn_head_size)
                viewed_q = q.view(*new_shape)
                viewed_k = k.view(*new_shape)
                viewed_v = v.view(*new_shape)

                # 4.permute
                permuted_q = viewed_q.permute(0, 2, 1, 3)# (batch, head, seq_length, head_features)
                permuted_k = viewed_k.permute(0, 2, 1, 3)# (batch, head, seq_length, head_features)
                permuted_v = viewed_v.permute(0, 2, 1, 3)# (batch, head, seq_length, head_features)
                return permuted_q, permuted_k, permuted_v

    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    input_data = [
        [torch.rand((batch, sequence, hidden_size* 3))],
    ]
    metric_table = ['max_diff','MAE']
    epsilon_dict = {'f32':1e-6, 'f16':1e-2}
    case_name =  __file__.split('.py')[0]# You can change your name
    dump_flag = True #it will dump alll wrong cases
    device = torch.device("privateuseone:0")

    My_Tester = Tester_Basic(case_name, device, metric_table, epsilon_dict,seed, dump_flag)
    return My_Tester.Torch_Test_Forward_Function(Test_Module(), input_data)


def case2():
    #step0:set seed
    seed=1000
    set_bacis_info(seed)

    #step1: define test model
    batch = 2
    sequence = 8
    hidden_size = 768
    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
            def forward(self, inp):
                a1,a2,a3 = inp.split(hidden_size, dim=2)
                y = a1.contiguous()+a2.contiguous()+a3.contiguous()
                return y

    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    input_data = [
        [torch.rand((batch, sequence, hidden_size* 3))],
    ]
    metric_table = ['max_diff','MAE']
    epsilon_dict = {'f32':1e-6, 'f16':1e-2}
    case_name =  __file__.split('.py')[0]# You can change your name
    dump_flag = True #it will dump alll wrong cases
    device = "privateuseone:0"

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
        return input_sample_tpu[0].grad,input_sample_cpu[0].grad #Notice [0] because input_data has [],[]

    My_Tester = Tester_Basic(case_name, device, metric_table, epsilon_dict,seed, dump_flag)
    My_Tester.customized_execute_function = customized_execute_function
    return My_Tester.Torch_Test_Forward_Function(Test_Module(), input_data)

if __name__ == "__main__":
    #This example shows all  [], [[]] is acceptable
    case1()


#######################
##  case1():forward + backward[[T]]
########################