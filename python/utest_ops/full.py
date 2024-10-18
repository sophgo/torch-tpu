import torch
import torch.nn as nn
import torch.nn.functional as F
from top_utest import set_requires_grad, move_to, set_bacis_info, Tester_Basic

def case1():
    #step0:set seed
    seed = 1000
    set_bacis_info(seed)

    #step1: define test model
    # Define the class at the module level
    class Test_Module(nn.Module):
        def __init__(self):
            super(Test_Module, self).__init__()

        def forward(self, size,scalar):
            return torch.full(size,scalar)
    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    mask = torch.randint(0, 2, (10, 10),dtype=torch.bool)

    class ShapeIter:
        @staticmethod
        def any_shape(dim=2):
            base = [1] * dim
            index = 0
            for i in range(10):
                base = base[:]
                base[index] *= 5
                index = (index + 1) % dim
                yield base

    input_data = []
    for shape in ShapeIter.any_shape():
        for scalar in [1.0, -1.0, 5.0]:
            input_data.append([shape, scalar])

    metric_table = ['max_diff', 'MAE']
    chip_arch_dict = {"bm1684x": 1, 'sg2260': 1}
    epsilon_dict = {'bm1684x': {'f32': 1e-6, 'f16': 1e-2}, 'sg2260': {'f32': 1e-6, 'f16': 1e-2}}
    case_name = __file__.split('.py')[0]
    dump_flag = True
    device = torch.device("privateuseone:0")

    My_Tester = Tester_Basic(case_name, chip_arch_dict, device, metric_table, epsilon_dict,seed, dump_flag)
    return My_Tester.Torch_Test_Execution_Function(Test_Module(), input_data)

if __name__ == "__main__":
    #This example shows all  [], [[]] is acceptable
    case1()
