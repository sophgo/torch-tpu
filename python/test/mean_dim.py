import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def test_mean_dim():
    input_tensor = torch.tensor(
    [[1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]], dtype=torch.float32)
    input_tensor_tpu = input_tensor.clone().to(device)
    cpu_output_0 = torch.mean(input_tensor, 0, True)   
    tpu_output_0 = torch.mean(input_tensor_tpu, 0, True)
    cpu_output_1 = torch.mean(input_tensor, 1, False)
    tpu_output_1 = torch.mean(input_tensor_tpu, 1, False)
    print("Input:", input_tensor)
    print("CPU_Output_0:", cpu_output_0)
    print("TPU_Output_0:", tpu_output_0.cpu())
    print("CPU_Output_1:", cpu_output_1)
    print("TPU_Output_1:", tpu_output_1.cpu())
  
if __name__ == "__main__":
    test_mean_dim()
