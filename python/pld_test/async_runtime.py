import torch
import torch.nn.functional as F
import copy
import time
import torch_tpu
torch.manual_seed(1000)

def case1():
    """
    确认torch是异步的
    """
    device = torch.device("tpu:0")
    batch = 10
    sequence = 1024
    hidden_size = 768
    
    inp_cpu = torch.rand(batch, sequence, hidden_size)
    out_cpu = F.gelu(inp_cpu)
    inp_tpu = copy.deepcopy(inp_cpu)
    import pdb;pdb.set_trace()
    t1 = time.time()
    inp_tpu = inp_tpu.to(device, non_blocking=True)
    print("after h2d")
    out_tpu = F.gelu(inp_tpu)
    print("after cal")
    out_tpu1 = out_tpu.to("cpu", non_blocking = True)
    print("after d2h")
    t2 = time.time()
    out_tpu2 = out_tpu.to("cpu", non_blocking = False)
    t3 = time.time()
    print(f"[time] t2 - t1 = {t2 - t1}; the time only contian cpu send task's time.")
    print(f"[time] t3 - t2 = {t3 - t2}; the time is sync with tpu's time.")
    print(f"[time] t3 - t1 = {t3 - t1}; the time is all time of the process")
    print("compare results:")
    diff = out_cpu - out_tpu2
    print("max diff = ", torch.max(abs(diff)))

def case_time2():
    """
    确认队列是排在scalar
    ==================
    torch_tpu.tpu.synchronize(): 200ms

    """
    LOOPS = [1e2, 1e3, 1e4, 1e5]
    LOOPS = [1] + [it * 1e1 for it in LOOPS]
    times_w_kernel = []
    times_wo_kernel = []
    device = torch.device("tpu:0")
    inp = torch.randn(8)
    inp_tpu = copy.deepcopy(inp).to(device, non_blocking = False)
    for LOOP in LOOPS:
        t1 = time.time()
        for _ in range(int(LOOP)):
            out_tpu = torch.ops.my_ops.dummy(inp_tpu)
        torch_tpu.tpu.synchronize()
        t2 = time.time()
        times_w_kernel.append(t2 - t1)

        t1 = time.time()
        for _ in range(int(LOOP)):
            out_tpu = torch.ops.my_ops.dummy_no_kernel_launch(inp_tpu)
        torch_tpu.tpu.synchronize()
        t2 = time.time()
        times_wo_kernel.append(t2 - t1) 
    print("compare results:")
    diff = out_tpu.to("cpu", non_blocking = False) - inp
    print("max diff = ", torch.max(abs(diff)))    
    print("dummy : ", times_w_kernel)
    print("dummy_no_kernel_launch(pure host time) : ", times_wo_kernel)


if __name__ == "__main__":
    case1()
    case_time2()