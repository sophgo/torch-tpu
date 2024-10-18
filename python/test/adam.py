import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
# 设置随机种子
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

# 创建初始张量
a = [torch.randn((48, 48, 94, 26), requires_grad=True).to(device) for _ in range(2)]
b = [torch.randn_like(a[i]).to(device) for i in range(2)]  # 模拟梯度
c = [torch.zeros_like(a[i]).to(device) for i in range(2)]  # 一阶矩估计
d = [torch.zeros_like(a[i]).to(device) for i in range(2)]  # 二阶矩估计
e = [torch.zeros_like(a[i]).to(device) for i in range(2)]  # 最大二阶矩估计
state_steps = [torch.tensor([1]).to(device) for _ in range(2)] # 时间步数

# 其他参数
lr = 0.001
beta1 = 0.9
beta2 = 0.999
weight_decay = 0.01
eps = 1e-8
amsgrad = False
maximize = False
grad_scale = None
found_inf = None

# 使用标准 Adam 优化器计算更新
def standard_adam_update(a, b, lr, beta1, beta2, weight_decay, eps, amsgrad=False, maximize=False):
    optimizer = torch.optim.Adam(a, lr=lr, betas=(beta1, beta2), weight_decay=weight_decay, eps=eps, amsgrad=amsgrad)
    
    # 假设 b 是梯度
    optimizer.zero_grad()
    for i in range(len(a)):
        a[i].grad = b[i]
    optimizer.step()
    
    return a

# 使用标准 Adam 优化器计算更新
a_standard = [a[i].clone().detach().requires_grad_() for i in range(2)]
a_standard_updated = standard_adam_update(a_standard, b, lr, beta1, beta2, weight_decay, eps, amsgrad, maximize)

# 调用_fused_adam_函数
a_fused = [a[i].clone().detach().requires_grad_() for i in range(2)]
torch._fused_adam_(
    a_fused, b, c, d, e, state_steps,
    lr=lr, beta1=beta1, beta2=beta2, weight_decay=weight_decay, eps=eps, amsgrad=amsgrad, maximize=maximize,
    grad_scale=grad_scale, found_inf=found_inf
)

output_file = "adam_updates.txt"
with open(output_file, 'w') as f:
    print("\nStandard Adam Update:", file=f)
    for i in range(len(a_standard_updated)):
       print(f"Parameter {i}:", file=f)
       print(a_standard_updated[i].cpu(), file=f)

    print("\nFused Adam Update:", file=f)
    for i in range(len(a_fused)):
        print(f"Parameter {i}:", file=f)
        print(a_fused[i].cpu(), file=f)

    # 验证结果是否一致
    print("\nVerification:", file=f)
    for i in range(len(a_standard_updated)):
       print(f"Standard vs Fused (Parameter {i}):", torch.allclose(a_standard_updated[i].cpu(), a_fused[i].cpu(), atol=1e-2), file=f)

print(f"Output has been written to {output_file}")

