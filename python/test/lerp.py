import torch_tpu
import torch
import torch.nn.functional as F
import time
torch.manual_seed(1000)

device = "tpu"

start = torch.arange(1., 5.).tpu()
end = torch.empty(4).fill_(10).tpu()
out =  torch.lerp(start, end, 0.5)
print(out.cpu())