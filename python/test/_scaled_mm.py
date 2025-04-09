import torch
import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"
size = (16, 16)
x = torch.empty(size, device=device, dtype=torch.float8_e4m3fn)
y = torch.empty(size, device=device, dtype=torch.float8_e4m3fn).t()
scale_a = torch.tensor(1.5, device=device)
scale_b = torch.tensor(0.66, device=device)
out_fp8, amax_fp8 = torch._scaled_mm(x, y)