import torch
import torch.nn as nn
import torch.nn.functional as F
from top_utest import set_bacis_info, Tester_Basic, set_requires_grad, move_to

def case1():
    #step0:set seed
    seed=1000
    set_bacis_info(seed)

    #step1: define test model
    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
            def forward(self, input, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True):
                return torch.nn.functional.avg_pool2d(input,kernel_size, stride, padding)
    #list is also acceptable
    input_data = [
    [torch.tensor([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
    ], dtype=torch.float32).unsqueeze(0).unsqueeze(0), 2, 2, 0],
    ]
    metric_table = ['max_diff','MAE']
    chip_arch_dict = {"bm1684x":0, 'sg2260':1}
    epsilon_dict = {'bm1684x':{'f32':1e-6,'f16':1e-2},'sg2260':{'f32':1e-6,'f16':1e-2}}
    case_name =  __file__.split('.py')[0]# You can change your name
    dump_flag = True #it will dump alll wrong cases

    device = torch.device("privateuseone:0")
    # you can write your own forward and backward fucntion
    # assuming net_tpu and input_sample_tpu has loaded on tpu
    # In this way you can define backward propagation by your self.
    def customized_execute_function(input_sample_cpu, input_sample_tpu, net_cpu, net_tpu, dtype):
        set_requires_grad(input_sample_cpu[0], True)
        set_requires_grad(input_sample_tpu[0], True)

        output_cpu = net_cpu(*input_sample_cpu)
        output_tpu = net_tpu(*input_sample_tpu)

        grad_o = torch.ones(output_cpu.shape)
        grad_o_tpu =  move_to(grad_o, device, dtype)  #grad_o.to(device)

        output_cpu.backward(grad_o)
        output_tpu.backward(grad_o_tpu)
        #tpu first
        return (output_tpu, input_sample_tpu[0].grad), (output_cpu, input_sample_cpu[0].grad)

    My_Tester = Tester_Basic(case_name, chip_arch_dict, device, metric_table, epsilon_dict,seed, dump_flag)
    My_Tester.customized_execute_function = customized_execute_function
    return My_Tester.Torch_Test_Execution_Function(Test_Module(), input_data)


if __name__ == "__main__":
    case1()

#######################
##  case1():forward [[T]]
########################