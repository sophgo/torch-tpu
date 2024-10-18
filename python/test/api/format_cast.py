import torch
import torch_tpu
import torch.nn.functional as F
import torch.nn as nn
import copy
import time

device = "tpu"
def test_format_cast_api():
    N = 3
    C = 3
    H = 3
    W = 3
    a = torch.range(1, N * C * H * W).view((N, C, H, W)).to(dtype=torch.float16).to(device)
    b = torch_tpu.tpu_format_cast(a, 3)
    print(a.cpu())
    print(b.cpu())

def test_conv_function():
    inp = torch.randn((2, 3, 16, 16))
    w   = torch.randn((4, 3, 3, 3))
    o_cpu = F.conv2d(inp, w)
    print("cpu",o_cpu)

    inp_tpu = inp.to(device)
    w_tpu   = w.to(device)
    o_tpu_f32 = F.conv2d(inp_tpu, w_tpu)
    print("tpu f32", o_tpu_f32.cpu())

    inp_tpu_f16 = inp_tpu.half()
    w_tpu_f16   = w_tpu.half()
    o_tpu_f16   = F.conv2d(inp_tpu_f16, w_tpu_f16)
    print("tpu f16", o_tpu_f16.cpu())

    w_tpu_f16_32ic = torch_tpu.tpu_format_cast(w_tpu_f16, 1)
    o_tpu_f16_format = F.conv2d(inp_tpu_f16, w_tpu_f16_32ic)
    print("tpu f16 formated", o_tpu_f16_format.cpu())

    diff1 = abs(o_tpu_f16_format.cpu() - o_cpu)
    diff2 = abs(o_tpu_f16_format.cpu() - o_tpu_f16.cpu())
    print(torch.max(diff1), torch.max(diff2))

def test_conv_module_eval():
    with torch.no_grad():
        inp = torch.randn((2, 3, 16, 16))
        net_cpu = nn.Conv2d(3, 8, (3,3))
        out_cpu = net_cpu(inp)

        inp_tpu = inp.to(device)
        net_tpu = net_cpu.to(device)
        out_tpu = net_tpu(inp_tpu)

        inp_tpu_fp16 = inp_tpu.half()
        net_tpu_fp16 = net_tpu.to(torch.float16)
        out_tpu_fp16 = net_tpu_fp16(inp_tpu_fp16)

        diff = abs(out_tpu.cpu() - out_cpu)        
        diff2 = abs(out_tpu_fp16.cpu() - out_cpu)
        print(torch.max(diff), torch.max(diff2))


def test_conv_module_train_forward():
    inp = torch.randn((2, 3, 16, 16))
    net_cpu = nn.Conv2d(3, 8, (3,3))
    net_cpu.train()
    out_cpu = net_cpu(inp)
    import pdb;pdb.set_trace()

    inp_tpu = inp.to(device)
    net_tpu = net_cpu.to(device)
    out_tpu = net_tpu(inp_tpu)
    import pdb;pdb.set_trace()

    inp_tpu_fp16 = inp_tpu.half()
    net_tpu_fp16 = net_tpu.to(torch.float16)
    out_tpu_fp16 = net_tpu_fp16(inp_tpu_fp16)
    import pdb;pdb.set_trace()

    diff = abs(out_tpu.cpu() - out_cpu)        
    diff2 = abs(out_tpu_fp16.cpu() - out_cpu)
    print(torch.max(diff), torch.max(diff2))
    import pdb;pdb.set_trace()

class Net(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv = nn.Conv2d(3, 3, (3,3))
        self.conv2 = nn.Conv2d(3, 3, (3,3))
        # self.conv.weight = nn.Parameter(torch.ones_like(self.conv.weight))
        # self.conv2.weight= nn.Parameter(torch.ones_like(self.conv2.weight))
    def forward(self, x):
        y = self.conv(x)
        return self.conv2(y)

def test_conv_module_trian_forward_and_backward():
    inp = torch.randn((2, 3, 16, 16))
    inp.requires_grad = True
    inp_tpu_fp16 = inp.clone().detach().to(device).half()
    inp_tpu_fp16.requires_grad = True

    net_cpu = Net()
    net_cpu.train()
    net_tpu_fp16 = copy.deepcopy(net_cpu).to(device).to(torch.float16)
    out_cpu = net_cpu(inp)
    out_tpu_fp16 = net_tpu_fp16(inp_tpu_fp16)

    grad_o = torch.ones_like(out_cpu)
    grad_o_tpu_fp16 = grad_o.to(device).half()

    out_cpu.backward(grad_o)
    out_tpu_fp16.backward(grad_o_tpu_fp16)

    #diff = abs(out_tpu.cpu() - out_cpu)
    diff2 = abs(out_tpu_fp16.cpu() - out_cpu)
    print("==== compare forward", torch.max(diff2))

    #diff_in_g  = abs(inp_tpu.grad.cpu() - inp.grad)
    diff_in_g2 = abs(inp_tpu_fp16.grad.cpu() - inp.grad)
    print("==== compare inp.grad", torch.max(diff_in_g2))


    #diff_w_g  = abs(net_tpu.weight.grad.cpu() - net_cpu.weight.grad)
    diff_w_g = abs(net_tpu_fp16.conv.weight.grad.cpu() - net_cpu.conv.weight.grad)
    diff_w_g2 = abs(net_tpu_fp16.conv2.weight.grad.cpu() - net_cpu.conv2.weight.grad)
    print("==== compare conv.weight.grad", torch.max(diff_w_g) / torch.max(abs(net_cpu.conv.weight.grad)))
    print("==== compare conv2.weight.grad", torch.max(diff_w_g2)/ torch.max(net_cpu.conv2.weight.grad))
    import pdb;pdb.set_trace()
    print("net_cpu.conv.weight.grad" , net_cpu.conv.weight.grad)
    print("net_tpu_fp16.conv.weight.grad.cpu()", net_tpu_fp16.conv.weight.grad.cpu())

def test_conv_module_trian_forward_and_backward_and_update():
    net_cpu = Net()
    net_cpu.train()
    net_tpu_fp16 = copy.deepcopy(net_cpu).to(device).to(torch.float16)

    optimizer = torch.optim.SGD(net_cpu.parameters(), lr=0.01)
    optimizer_tpu = torch.optim.SGD(net_tpu_fp16.parameters(), lr = 0.01)
    optimizer.zero_grad()
    optimizer_tpu.zero_grad()

    inp = torch.randn((2, 3, 16, 16))
    inp.requires_grad = True
    inp_tpu_fp16 = inp.clone().detach().to(device).half()
    inp_tpu_fp16.requires_grad = True

    print("====== forward phase ===========")
    out_cpu = net_cpu(inp)
    t1 = time.time()
    out_tpu_fp16 = net_tpu_fp16(inp_tpu_fp16)
    print(f"forward time = {time.time() - t1}")

    diff2 = abs(out_tpu_fp16.cpu() - out_cpu)
    print("==== compare forward", torch.max(diff2))

    ###### start backward
    print("====== backward phase ===========")
    grad_o = torch.ones_like(out_cpu) * 100
    grad_o_tpu_fp16 = grad_o.to(device).half()

    out_cpu.backward(grad_o)
    t1 = time.time()
    out_tpu_fp16.backward(grad_o_tpu_fp16)
    print(f"backward time = {time.time() - t1}")

    diff_in_g2 = abs(inp_tpu_fp16.grad.cpu() - inp.grad)
    print("==== compare inp.grad", torch.max(diff_in_g2))

    diff_w_g = abs(net_tpu_fp16.conv.weight.grad.cpu() - net_cpu.conv.weight.grad)
    diff_w_g2 = abs(net_tpu_fp16.conv2.weight.grad.cpu() - net_cpu.conv2.weight.grad)
    print("==== compare conv.weight.grad", torch.max(diff_w_g) / torch.max(abs(net_cpu.conv.weight.grad)))
    print("==== compare conv2.weight.grad", torch.max(diff_w_g2)/ torch.max(net_cpu.conv2.weight.grad))

    diff_w = abs(net_tpu_fp16.conv.weight.cpu() - net_cpu.conv.weight)
    diff_w_2 = abs(net_tpu_fp16.conv2.weight.cpu() - net_cpu.conv2.weight)
    print("==== compare conv.weight", torch.max(diff_w) / torch.max(abs(net_cpu.conv.weight)))
    print("==== compare conv2.weight", torch.max(diff_w_2)/ torch.max(net_cpu.conv2.weight))

    print("====== update phase ===========")
    optimizer.step()
    t1 = time.time()
    optimizer_tpu.step()
    print(f"update weight time = {time.time() - t1}")

    diff_w = abs(net_tpu_fp16.conv.weight.cpu() - net_cpu.conv.weight)
    diff_w_2 = abs(net_tpu_fp16.conv2.weight.cpu() - net_cpu.conv2.weight)
    print("==== compare conv.weight", torch.max(diff_w) / torch.max(abs(net_cpu.conv.weight)))
    print("==== compare conv2.weight", torch.max(diff_w_2)/ torch.max(net_cpu.conv2.weight))
    import pdb;pdb.set_trace()
    print("net_cpu.conv.weight" , net_cpu.conv.weight)
    print("net_tpu_fp16.conv.weight.cpu()", net_tpu_fp16.conv.weight.cpu())

if __name__ == "__main__":
    #test_format_cast_api()
    #test_conv_function()
    #test_conv_module_eval()
    #test_conv_module_train_forward()
    #test_conv_module_trian_forward_and_backward()
    test_conv_module_trian_forward_and_backward_and_update()