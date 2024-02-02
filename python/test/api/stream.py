import torch
import torch_tpu

# 创建CUDA流
stream = torch_tpu.tpu.Stream()

# 创建在指定流上的张量
with torch_tpu.tpu.stream(stream):
    a = torch.randn(1000, device='cuda')
    b = torch.randn(1000, device='cuda')

# 在默认流上执行操作
c = torch.matmul(a, b)

# 等待流上的操作完成
torch_tpu.tpu.synchronize()