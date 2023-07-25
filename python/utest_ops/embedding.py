import torch
import torch.nn as nn
import torch.nn.functional as F
from top_utest import set_requires_grad,move_to, set_bacis_info, Tester_Basic

def case1():
    #step0:set seed
    seed=1000
    set_bacis_info(seed)

    #step1: define test model

    batch      = 1
    sequence   = 72
    vocab_size = 100
    embed_dim  = 768

    batch      = 1
    sequence   = 3
    vocab_size = 8
    embed_dim  = 4

    batch      = 32
    sequence   = 256
    vocab_size = 50257
    embed_dim  = 768
    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
                self.Embedding = nn.Embedding(vocab_size, embed_dim)
                self.Embedding.weight = nn.Parameter(torch.rand((vocab_size, embed_dim)))

            def forward(self, a1):
                return self.Embedding(a1)
    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    input_data = [
         #[torch.randint(0, vocab_size, (batch, sequence)).to(torch.long)],
         #[torch.range(0, batch * sequence - 1).view((batch, sequence)).to(torch.long)],
         [torch.randint(0, vocab_size, (batch, sequence)).to(torch.int32)],
         [torch.range(0, batch * sequence - 1).view((batch, sequence)).to(torch.int32)]
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
        # set_requires_grad(input_sample_cpu, True)
        # set_requires_grad(input_sample_tpu, True)
        output_cpu = net_cpu(*input_sample_cpu)
        output_tpu = net_tpu(*input_sample_tpu)
        #tpu-first
        return output_tpu, output_cpu

        grad_o =  torch.ones(batch, sequence, embed_dim)
        grad_o_tpu =  move_to(grad_o, device, dtype)

        output_cpu.backward(grad_o)
        output_tpu.backward(grad_o_tpu)

        #tpu-first
        return net_tpu.Embedding.weight.grad, net_cpu.Embedding.weight.grad
    My_Tester = Tester_Basic(case_name, device, metric_table, epsilon_dict,seed, dump_flag)
    My_Tester.customized_execute_function = customized_execute_function
    return My_Tester.Torch_Test_Forward_Function(Test_Module(), input_data)


if __name__ == "__main__":
    #This example shows all  [], [[]] is acceptable
    case1()

#######################
##  case1():forward + backward [[T]]
########################