from typing import Any
from test_utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"
import copy

class TestClone:
    def __call__(self, x) -> Any:
        return self.forward(x)

    def forward(self, x):
        return torch.ops.aten.clone(x)


def case1():
    # test dtype

    for dtype in DTypeIter.all():
        for shape in ShapeIter.any_shape():
            cpu_data = torch.ones(*shape, dtype=dtype)
            Evaluator().add_abs_evalute().evavlute([TestClone()], [cpu_data])


def case2():
    '''clone
    '''
    a = torch.randint(0, 10, (5,5), dtype=torch.float)
    a_tpu       = a #.to(device)    
    a_tpu_clone = a_tpu.clone()
    print(f"a_tpu.data_ptr() : {a_tpu.data_ptr()}, a_tpu_clone.data_ptr() : {a_tpu_clone.data_ptr()}")
    print(f"id(a_tpu):  {id(a_tpu)}, id(a_tpu_clone) : {id(a_tpu_clone)}")
    print(f"max diff : {torch.max(abs(a_tpu-a_tpu_clone))}")
    import pdb; pdb.set_trace()

def case3():
    """ copy.copy
    """
    a = torch.randint(0, 10, (5,5), dtype=torch.float)
    a_tpu       = a.to(device)    
    a_tpu_clone = copy.copy(a_tpu)
    print(f"a_tpu.data_ptr() : {a_tpu.data_ptr()}, a_tpu_clone.data_ptr() : {a_tpu_clone.data_ptr()}")
    print(f"id(a_tpu):  {id(a_tpu)}, id(a_tpu_clone) : {id(a_tpu_clone)}")
    print(f"max diff : {torch.max(abs(a_tpu-a_tpu_clone))}")

def case4():
    """copy.deepcopy
    """
    a           = torch.randint(0, 10, (5,5), dtype=torch.float)
    a_tpu       = a.to(device)    
    a_tpu_clone = copy.deepcopy(a_tpu)
    print(f"a_tpu.data_ptr() : {a_tpu.data_ptr()}, a_tpu_clone.data_ptr() : {a_tpu_clone.data_ptr()}")
    print(f"id(a_tpu):  {id(a_tpu)}, id(a_tpu_clone) : {id(a_tpu_clone)}")
    print(f"max diff :  {torch.max(abs(a_tpu-a_tpu_clone))}")
if __name__ == "__main__":
    #case1()
    # case2()
    # case3()
    case4()
