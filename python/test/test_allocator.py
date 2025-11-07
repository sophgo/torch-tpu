
import torch
import torch_tpu

t = torch.empty(1, 128, 1024, 1024, dtype=torch.float32)
for i in range(2**10):
  print(f'-------------------------------Allocate round {i}')
  newT = t.to('tpu:0')
  print(newT.to('cpu')[0, 0, 0, :])
