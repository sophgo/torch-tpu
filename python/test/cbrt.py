import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

# 如果需要测试，请进入Unary.cpp文件，注释bitwise_not_tpu相关函数，取消注释cbrt_tpu相关函数
def case1():
    input_origin=torch.tensor([[0, -8, 1],
                               [1, 8, 9],
                               [64, -27, 8]]).float()    # 因为bitwise_not接口的原因，暂时只支持float输入
    input_tpu=input_origin.to(device)
    
    output_tpu_prims=torch.ops.prims.bitwise_not(input_tpu).cpu()    # pytorch2.0.1版本库没有cbrt接口，借用bitwise_not接口实现
    
    print("input_origin : ",input_origin)
    print("output_tpu_prims : ", output_tpu_prims)
    
if __name__ == "__main__":
    case1()