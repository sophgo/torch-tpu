
======
1. 编译安装 torch-tpu wheel
```shell
source scripts/envsetup.sh bm1684x stable
new_clean
new_build
```
2. 检查安装成功
```shell
pip list ｜grep torch-tpu
```

3. example使用
```python
import torch
import torch-tpu

batch = 8
sequence = 1024
hidden_size = 768
out_size = 3

inp = torch.rand(batch, sequence, hidden_size).to(device)
ln_net = nn.Linear(hidden_size, out_size).to(device)
out = ln_net(inp)
print(out.cpu())
```