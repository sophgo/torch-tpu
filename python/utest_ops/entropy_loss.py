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
                #remember to using nn.Parameter, otherwise model.to cannot correctly move it to TPU.
                self.label =  nn.Parameter(torch.randint(0, vtable_size, (batch, sequence)),requires_grad=False)
                #register_buffer is also acceptable here, but still you need nn.Parameter
                # self.label = self.register_buffer('label', nn.Parameter(torch.randint(0, vtable_size, (batch, sequence)),requires_grad=False))
            def forward(self, inp):
                return self.loss_fct(inp.view(-1, vtable_size), self.label.view(-1))
    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    input_data = [
         [ torch.ones(batch, sequence, vtable_size)],
    ]
    metric_table = ['max_diff','MAE']
    chip_arch_dict = {"bm1684x":1, 'sg2260':1}
    epsilon_dict = {'bm1684x':{'f32':1e-6,'f16':1e-2},'sg2260':{'f32':1e-6,'f16':1e-2}}
    case_name =  __file__.split('.py')[0]# You can change your name
    dump_flag = True #it will dump alll wrong cases
    device = torch.device("privateuseone:0")

    # you can write your own forward and backward fucntion
    #assuming net_tpu and input_sample_tpu has loaded on tpu
    #In this way you can define backward propagation by your self.
    def customized_execute_function(input_sample_cpu, input_sample_tpu, net_cpu, net_tpu, dtype):
        #  only Tensors of floating point and complex dtype can require gradients
        set_requires_grad(input_sample_cpu, True)
        set_requires_grad(input_sample_tpu, True)

        output_cpu = net_cpu(*input_sample_cpu)
        output_tpu = net_tpu(*input_sample_tpu)

        output_cpu.backward()
        output_tpu.backward()
        #tpu-first
        return input_sample_tpu[0].grad, input_sample_cpu[0].grad #Notice [0] because input_data has [],[]

    My_Tester = Tester_Basic(case_name, chip_arch_dict, device, metric_table, epsilon_dict,seed, dump_flag)
    My_Tester.customized_execute_function = customized_execute_function
    return My_Tester.Torch_Test_Execution_Function(Test_Module(), input_data)

def case2():
    #step0:set seed
    seed=10
    set_bacis_info(seed)

    #step1: define test model
    batch = 639
    sequence = 1
    vtable_size =32000
    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
                self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
                #remember to using nn.Parameter, otherwise model.to cannot correctly move it to TPU.
                self.label =  nn.Parameter(torch.randint(0, vtable_size, (batch, )),requires_grad=False)
                self.label[-5:]=-100

                #register_buffer is also acceptable here, but still you need nn.Parameter
                # self.label = self.register_buffer('label', nn.Parameter(torch.randint(0, vtable_size, (batch, sequence)),requires_grad=False))
            def forward(self, inp):
                return self.loss_fct(inp, self.label)
    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    input_data = [
         [ torch.rand(batch,  vtable_size)],
    ]
    metric_table = ['max_diff','MAE']
    chip_arch_dict = {"bm1684x":1, 'sg2260':1}
    epsilon_dict = {'bm1684x':{'f32':1e-6},'sg2260':{'f32':1e-6}}
    case_name =  __file__.split('.py')[0]# You can change your name
    dump_flag = True #it will dump alll wrong cases
    device = torch.device("privateuseone:0")

    # you can write your own forward and backward fucntion
    # assuming net_tpu and input_sample_tpu has loaded on tpu
    # In this way you can define backward propagation by your self.
    def customized_execute_function(input_sample_cpu, input_sample_tpu, net_cpu, net_tpu, dtype):
        #  only Tensors of floating point and complex dtype can require gradients
        set_requires_grad(input_sample_cpu, True)
        set_requires_grad(input_sample_tpu, True)

        output_cpu = net_cpu(*input_sample_cpu)
        output_tpu = net_tpu(*input_sample_tpu)

        output_cpu.backward()
        output_tpu.backward()
        # tpu-first
        print("tpu_grad0",input_sample_tpu[0].grad[0:7].cpu())
        print("tpu_grad1",input_sample_tpu[0].grad[-7:].cpu())
        print("cpu_grad0",input_sample_cpu[0].grad[0:7])
        print("cpu_grad1",input_sample_cpu[0].grad[-7:])
        return input_sample_tpu[0].grad, input_sample_cpu[0].grad #Notice [0] because input_data has [],[]

    My_Tester = Tester_Basic(case_name, chip_arch_dict, device, metric_table, epsilon_dict,seed, dump_flag)
    My_Tester.customized_execute_function = customized_execute_function
    return My_Tester.Torch_Test_Execution_Function(Test_Module(), input_data)

if __name__ == "__main__":
    #This example shows all  [], [[]] is acceptable
    case1()
    case2()

#######################
##  case1():forward + backward [[T]]; Notice No set_requires_grad
########################