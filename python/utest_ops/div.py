import torch
import torch.nn as nn
import torch.nn.functional as F
from top_utest import Torch_Test_Forward_Function
torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

def case1():
    #step0:set seed
    seed=1000
    torch.manual_seed(seed)
    #step1: define test model
    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
            def forward(self, a1,a2):
                return a1+a2
            
    #step2: prepare input data
    input_data = {
         "simple0": [torch.rand((5,5)),  torch.rand((5,5))],
         "simple1": [torch.rand((5,5,5)),  torch.rand((5,5,5))]
    }
    #list is also acceptable
    input_data = [
        [torch.rand((5,5)),  torch.rand((5,5))],
         [torch.rand((5,5,5)),  torch.rand((5,5,5))]
    ]
    metric_table = ['max_diff','MAE']
    epsilon_dict = {'f32':1e-6,'f16':1e-2}
    case_name =  __file__.split('.py')[0]
    return Torch_Test_Forward_Function(case_name, Test_Module, input_data, metric_table, epsilon_dict, seed)



if __name__ == "__main__":
    case1()