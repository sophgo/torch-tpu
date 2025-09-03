import torch
import torch_tpu

device = "tpu:0"

src = torch.tensor([[1,2,3,4,5],[6,7,8,9,10]], dtype=torch.float16, device=device)
index = torch.tensor([[0, 1, 2, 0]], dtype=torch.int64, device=device)
self_t = torch.ones((3, 5), dtype=torch.float16, device=device)
self_t.scatter_(0, index, src)

src2 = torch.tensor([[1,2,3,4,5],[6,7,8,9,10]], dtype=torch.float16, device='cpu')
index2 = torch.tensor([[0, 1, 2, 0]], dtype=torch.int64, device='cpu')
self_t2 = torch.ones((3, 5), dtype=torch.float16, device='cpu')
self_t2.scatter_(0, index2, src2)

diff = self_t.cpu() - self_t2

import pdb; pdb.set_trace()