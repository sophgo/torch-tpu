import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)

def case_CrossEntropyLoss():
    device = "privateuseone"
    batch = 8
    sequence = 8
    vtable_size = 10

    inp_cpu = torch.rand(batch, sequence, vtable_size)
    inp_tpu = copy.deepcopy(inp_cpu).to(device)
    inp_cpu.requires_grad = True
    inp_tpu.requires_grad = True

    label_cpu = torch.randint(0, vtable_size, (batch, sequence))
    label_tpu = copy.deepcopy(label_cpu).to(device)

    loss_fct = nn.CrossEntropyLoss()

    loss_cpu = loss_fct(inp_cpu.view(-1, vtable_size), label_cpu.view(-1))
    loss_tpu = loss_fct(inp_tpu.view(-1, vtable_size), label_tpu.view(-1))

    loss_cpu.backward()
    loss_tpu.backward()
    print("=======compare grad=======")
    print("inp_cpu.grad: ", inp_cpu.grad)
    print("inp_tpu.grad: ", inp_tpu.grad.cpu())
    diff = inp_cpu.grad - inp_tpu.grad.cpu()
    print("max diff: ", torch.max(abs(diff)))

if __name__ == "__main__":
    case_CrossEntropyLoss()