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

def case5():
    out = []
    print("#"*5 + "case_clone" + "#"*5)
    for dtype_t in [torch.float32, torch.float16, torch.bfloat16]:
        a = torch.randn((1024,1,1,1024), dtype=dtype_t)*10
        b = torch.randn((1024,1,1,1024), dtype=dtype_t)*10

        cpu_out = a.clone()
        tpu_out = a.to(device)
        a_tpu_clone = tpu_out.clone()
        print(f"dtype: {dtype_t}")
        print(f"cpu_out: {cpu_out}")
        print(f"tpu_out: {tpu_out.cpu()}")
        print(f"a_tpu_clone: {a_tpu_clone.cpu()}")
        print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))

    for dtype_t in [torch.int32, torch.int16, torch.int8, torch.uint8]:
        a = torch.randint(0, 21, (256,1,1,7168), dtype=dtype_t)

        cpu_out = a.clone()
        tpu_out = a.to(device)
        a_tpu_clone = tpu_out.clone()

        print(f"max_diff: {torch.max(abs(cpu_out - tpu_out.cpu()))}")
        out.append(torch.max(abs(cpu_out - tpu_out.cpu())))

    print(out)
if __name__ == "__main__":
    #case1()
    # case2()
    # case3()
    case4()
    # case5()
