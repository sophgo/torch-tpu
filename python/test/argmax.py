import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_tpu

torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():
    dim = None
    input_origin=torch.rand(1,1,3840,1)
    input_tpu=input_origin.to(device)
    output_cpu=torch.argmax(input_origin,dim=dim)
    output_tpu=torch.argmax(input_tpu,dim=dim).cpu()
    # print("input_origin : ",input_origin)
    print("output_cpu : ", output_cpu.shape)
    print("output_tpu : ", output_tpu.shape)
    print("delta : ",(output_cpu==output_tpu).all())

def case_sd():
    input_origin = torch.randint(0, 1000, (1,10))
    input_tpu = input_origin.to(device)
    o_cpu = input_origin.argmax(-1)
    o_tpu = input_tpu.argmax(-1)
    diff = o_cpu - o_tpu.cpu()
    print(diff)
    import pdb;pdb.set_trace()

if __name__ == "__main__":
    #case1()
    case_sd()


