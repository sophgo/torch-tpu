import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")

def case1():
    """
    slice + contiguous + crossentroy loss + backward
    """
    device = "privateuseone"
    batch = 2
    sequence = 8
    hidden_size = 3

    inp = torch.ones(batch, sequence, hidden_size)
    inp.retain_grad = True
    inp.requires_grad = True

    inp_tpu = torch.ones(batch, sequence, hidden_size).to(device)
    inp_tpu.retain_grad = True
    inp_tpu.requires_grad = True
    
    label = torch.randint(0, hidden_size, (batch, sequence))

    loss_fct = nn.CrossEntropyLoss()

    shift_inp     = inp[..., :-1, :]
    shift_inp_tpu = inp_tpu[..., :-1, :]
    print("after slice")

    cshift_inp     =  shift_inp.contiguous()
    cshift_inp_tpu =  shift_inp_tpu.contiguous().float().cpu()
    print("after contiguous")

    print("=======forward results")
    print("==tpu")
    print(cshift_inp_tpu.cpu())
    print("==cpu")
    print(cshift_inp)

    shift_label = label[..., 1:].contiguous()

    print("=========begin calculate loss")
    loss = loss_fct(cshift_inp_tpu.view(-1, hidden_size), shift_label.view(-1))
    loss_cpu = loss_fct(cshift_inp.view(-1, hidden_size), shift_label.view(-1))
    print("=========end calculate loss")

    print("=========begin backward")
    loss_cpu.backward()
    loss.backward()
    print("=========end backward")


    print("====== grad info")
    print("----tpu inp")
    print(inp_tpu.grad.cpu())
    print("----cpu inp")
    print(inp.grad)
    print("tpu_inp.grad - cpu_inp.grad=")
    diff = inp_tpu.grad.cpu() - inp.grad
    min_value = torch.min(diff)
    max_val   = torch.max(diff)
    #print(diff)
    print("max val:", max_val)
    print("min val: ", min_value)


def case2():
    """
    slice backward
    """
    device = "privateuseone"
    batch = 2
    sequence = 8
    hidden_size = 3
    

    inp_tpu = torch.ones(batch, sequence, hidden_size).to(device)
    inp_tpu.retain_grad = True
    inp_tpu.requires_grad = True

    print("======begin slice")
    shift_inp_tpu = inp_tpu[..., :-1, :]
    print("======end slice")

    print("======begin contiguous")
    shift_inp_tpu.contiguous()
    print(shift_inp_tpu.is_contiguous())
    print("======end contiguous")
    
    print("begin to cpu")
    print(shift_inp_tpu.cpu())
    print("end to cpu")

    dshift_inp_tpu = torch.ones(batch, sequence-1, hidden_size).to(device)

    print("start backward")
    shift_inp_tpu.backward(dshift_inp_tpu)

    print("---- grad info")
    print(inp_tpu.grad.cpu())

    


def case3_slice_backward():
    """
    slice's backward is equal to copy.
    this case use copy to simulate slice's backward.
    """
    device = "privateuseone"
    batch = 2
    sequence = 8
    hidden_size = 3

    inp = torch.rand(batch, sequence, hidden_size)
    inp_tpu = inp.to(device)

    dinp = torch.ones(batch, sequence - 1, hidden_size)
    dinp_tpu = dinp.to(device)

    print("start backward")
    inp[..., :-1, :] = dinp
    inp_tpu[..., :-1, :] = dinp_tpu

def case4():
    """
    slice + slice
    """
    device = "privateuseone"
    batch = 2
    sequence = 8
    hidden_size = 3

    inp = torch.rand(batch, sequence, hidden_size)
    inp_tpu = inp.to(device)

    shift_inp_tpu = inp_tpu[1:, ...]
    shift_inp_tpu = shift_inp_tpu[:,1:,...]

    shift_inp     = inp[1:, ...]
    shift_inp = shift_inp[:,1:,...]

    print("after slice")
    cshift_inp_tpu =  shift_inp_tpu.contiguous().float().cpu()
    cshift_inp     =  shift_inp.contiguous().float().cpu()
    print("after contiguous")
    print("==cpu")
    print(cshift_inp_tpu.cpu())
    print("==tpu")
    print(cshift_inp)

if __name__ == "__main__":
    case1()