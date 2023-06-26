import torch
import torch.nn as nn
import torchvision
import copy
import time
from utils import Optimer

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
device = torch.device("privateuseone:0")
OPT = Optimer()

def R50_update(use_fp16 = False):
    B = 1
    C = 3
    H = 224
    W = 224
    num_class = 1000

    inp = torch.randn((B, C, H, W))
    inp_tpu = inp.clone().to(device)
    if use_fp16: inp_tpu = inp_tpu.half()

    label = torch.randint(0, num_class-1, (B))
    label_tpu = label.int().to(device)

    net = torchvision.models.resnet50()
    net_tpu = copy.deepcopy(net).to(device)
    if use_fp16: net_tpu = net_tpu.half()
    net.train()
    net_tpu.train()

    loss_fct = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.01)
    optimizer_tpu = torch.optim.AdamW(net_tpu.parameters(), lr=0.01)

    t1 = time.time()
    optimizer.zero_grad()
    print("[zero_grad] cpu time: ", time.time() - t1)
    t1 = time.time()
    optimizer_tpu.zero_grad()
    print("[zero_grad] tpu time: ", time.time() - t1)

    t1 = time.time()
    out = net(inp)
    print("[forward] cpu time: ", time.time() - t1)

    t1 = time.time()
    OPT.reset()
    out_tpu = net_tpu(inp_tpu)
    OPT.dump()
    print("[forward] tpu time: ", time.time() - t1)
    
    loss_cpu = loss_fct(out, label)
    loss_tpu = loss_fct(out_tpu, label_tpu)

    t1 = time.time()
    loss_cpu.backward()
    print("[backward] cpu time: ", time.time() - t1)

    t1 = time.time()
    loss_tpu.backward()
    print("[backward] tpu time: ", time.time() - t1)

    t1 = time.time()
    optimizer.step()
    print("Cpu AdamW update time : {} s".format(time.time() - t1))

    t1 = time.time()
    optimizer_tpu.step()
    print("Tpu AdamW update time : {} s".format(time.time() - t1))

    

def R50_backward(use_fp16=False):
    B = 1
    C = 3
    H = 224
    W = 224
    num_class = 1000

    inp = torch.randn((B, C, H, W))
    inp_tpu = inp.clone().to(device)
    if use_fp16: inp_tpu = inp_tpu.half()

    ref = torch.randn((B, num_class))
    ref_tpu = ref.clone().to(device)
    if use_fp16: ref_tpu = ref_tpu.half()

    net = torchvision.models.resnet50()
    net_tpu = copy.deepcopy(net).to(device)
    if use_fp16: net_tpu = net_tpu.half()
    net.train()
    net_tpu.train()

    t1 = time.time()
    out = net(inp)
    print("[forward] cpu time: ", time.time() - t1)

    t1 = time.time()
    OPT.reset()
    out_tpu = net_tpu(inp_tpu)
    OPT.dump()
    print("[forward] tpu time: ", time.time() - t1)
    
    t1 = time.time()
    out.backward(ref)
    print("[backward] cpu time: ", time.time() - t1)

    t1 = time.time()
    out_tpu.backward(ref_tpu)
    print("[backward] tpu time: ", time.time() - t1)


def R50_forward(use_fp16=False):
    B = 64
    C = 3
    H = 224
    W = 224
    inp = torch.randn((B, C, H, W))
    inp_tpu = inp.clone().to(device)
    if use_fp16: inp_tpu = inp_tpu.half()

    net = torchvision.models.resnet50()
    net_tpu = copy.deepcopy(net).to(device)
    if use_fp16: net_tpu = net_tpu.half()
    net.train()
    net_tpu.train()

    t1 = time.time()
    out = net(inp)
    print("[forward] cpu time: ", time.time() - t1)
    print(out.shape)

    t1 = time.time()
    OPT.reset()
    out_tpu = net_tpu(inp_tpu)
    OPT.dump()
    print("[forward] tpu time: ", time.time() - t1)

    diff = out - out_tpu.cpu()
    print("max diff : ", torch.max(abs(diff)))



if __name__ == "__main__":
    R50_forward(True)