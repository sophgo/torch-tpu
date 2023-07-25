import torch
import torch.nn as nn
import torch.nn.functional as F
from top_utest import set_requires_grad,move_to, set_bacis_info, Tester_Basic
import torchvision

def case1():
    #step0:set seed
    seed=1000
    set_bacis_info(seed)

    #step1: define test model
    B = 1
    C = 3
    H = 224
    W = 224
    num_class = 1000
    Resnet = torchvision.models.resnet18()
    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    label = torch.randint(0, num_class-1, (B, ))
    input_data = [
         [ torch.randn((B, C, H, W)), label],
    ]
    metric_table = ['max_diff','MAE']
    epsilon_dict = {'f32':1e-6,'f16':1e-2}
    case_name =  __file__.split('.py')[0]# You can change your name
    dump_flag = True #it will dump alll wrong cases
    device = torch.device("privateuseone:0")

    # you can write your own forward and backward fucntion
    #assuming net_tpu and input_sample_tpu has loaded on tpu
    #In this way you can define backward propagation by your self.
    def customized_execute_function(input_sample_cpu, input_sample_tpu, net_cpu, net_tpu, dtype):
        set_requires_grad(input_sample_cpu, True)
        set_requires_grad(input_sample_tpu, True)

        net_cpu.train()
        net_tpu.train()

        output_cpu = net_cpu(input_sample_cpu[0])
        output_tpu = net_tpu(input_sample_tpu[0])


        loss_fct = nn.CrossEntropyLoss()
        optimizer_cpu = torch.optim.AdamW(net_cpu.parameters(), lr=0.01)
        optimizer_tpu = torch.optim.AdamW(net_tpu.parameters(), lr=0.01)
        optimizer_cpu.zero_grad()
        optimizer_tpu.zero_grad()

        loss_cpu = loss_fct(output_cpu, input_sample_cpu[1])
        loss_tpu = loss_fct(output_tpu, input_sample_tpu[1])

        loss_cpu.backward()
        loss_tpu.backward()
        optimizer_cpu.step()
        optimizer_tpu.step()

        #tpu-first
        return input_sample_tpu[0].grad, input_sample_cpu[0].grad #Notice [0] because input_data has [],[]

    My_Tester = Tester_Basic(case_name, device, metric_table, epsilon_dict,seed, dump_flag)
    My_Tester.customized_execute_function = customized_execute_function
    #When you pass a model not a class of nn.module do not use()
    return My_Tester.Torch_Test_Forward_Function(Resnet, input_data)


if __name__ == "__main__":
    #This example shows all  [], [[]] is acceptable
    case1()

#######################
##  case1():forward + backward [[T]]
########################