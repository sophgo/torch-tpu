import torch
import torch_tpu
import os

#os.environ["ModelRtRunWithTorchTpu"] = '1'
#os.environ["TorchTpuSaveKernelModule"] = '1'

device='tpu:0'
def case1():
    from torch_tpu.tpu import BmodelRunner
    bmodel = './MM_0_f16_sg2260/compilation.bmodel' # weight is ones with shape[16, 34]
    
    input = torch.ones((16, 34), device=device)
    weight  = torch.ones((32,16), device=device)
    out_with_torch_runtime = torch.mm(weight, input)
    
    # ##
    runner = BmodelRunner(model_path=bmodel, device_id=int(device.split(':')[1]))
    out_with_model_runtime = runner.forward_sync(input)

    diff = abs(out_with_torch_runtime - out_with_model_runtime)
    print(torch.max(diff))

def case2():
    """ io-copy case """
    from torch_tpu.tpu.bmrt import BmodelRuner
    torch_tpu.tpu.set_device(0)
    input = torch.ones((16, 34), device=device)
    weight  = torch.ones((32,16), device=device)
    out_with_torch_runtime = torch.mm(weight, input)

    bmodel = './MM_0_f16_sg2260/compilation.bmodel' # weight is ones with shape[16, 34]
    runner = BmodelRuner(bmodel_path=bmodel, device_id=int(device.split(':')[1]))

    out_with_model_runtime = torch.ones((32, 34), device=device)
    runner.forward(input, out_with_model_runtime)

    diff = abs(out_with_torch_runtime - out_with_model_runtime)
    print(torch.max(diff))

def case3():
    """ io-inplace case """
    from torch_tpu.tpu.bmrt import BmodelRuner
    torch_tpu.tpu.set_device(0)
    input = torch.ones((16, 34), device=device)
    weight  = torch.ones((32,16), device=device)
    out_with_torch_runtime = torch.mm(weight, input)

    bmodel = './MM_0_f16_sg2260/compilation.bmodel' # weight is ones with shape[16, 34]
    runner = BmodelRuner(bmodel_path=bmodel, device_id=int(device.split(':')[1]))

    in_, out_ = runner.genInplaceIO()
    in_[0].copy_(input)
    runner.forward(in_, out_)

    diff = abs(out_with_torch_runtime - out_[0])
    print(torch.max(diff))

def case4():
    """ io-copy case """
    from torch_tpu.tpu.bmrt import BmodelModule
    input = torch.ones((16, 34), device=device)
    weight  = torch.ones((32,16), device=device)
    out_with_torch_runtime = torch.mm(weight, input)
    bmodel = './MM_0_f16_sg2260/compilation.bmodel' # weight is ones with shape[16, 34]
    runner = BmodelModule(bmodel_path=bmodel, device=device)
    out = runner.forward(input)
    diff = abs(out_with_torch_runtime - out[0])
    print(torch.max(diff))

def case5():
    """ io-alone case """
    from torch_tpu.tpu.bmrt import BmodelModule
    input = torch.ones((16, 34), device=device)
    weight  = torch.ones((32,16), device=device)
    out_with_torch_runtime = torch.mm(weight, input)
    bmodel = './MM_0_f16_sg2260/compilation.bmodel' # weight is ones with shape[16, 34]
    runner = BmodelModule(bmodel_path=bmodel, device=device)
    ins, outs = runner.get_inplace_io()
    ins[0].copy_(input)
    out = runner.forward(ins[0])
    diff = abs(out_with_torch_runtime - out[0])
    print(torch.max(diff))

if __name__ == "__main__":
    #case1()
    #case2()
    case3()
    #case4()
    #case5()