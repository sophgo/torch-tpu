import torch
import torch_tpu
from torch_tpu.tpu import BmodelRunner

device='tpu:0'
def case1():
    bmodel = './MM_0_f16_sg2260/compilation.bmodel' # weight is ones with shape[16, 34]
    
    input = torch.ones((16, 34), device=device)
    weight  = torch.ones((32,16), device=device)
    out_with_torch_runtime = torch.mm(weight, input)
    
    # ##
    runner = BmodelRunner(model_path=bmodel, device_id=int(device.split(':')[1]))
    out_with_model_runtime = runner.forward_sync(input)

    diff = abs(out_with_torch_runtime - out_with_model_runtime)
    print(torch.max(diff))


if __name__ == "__main__":
    case1()