import torch
import torch.nn as nn
import torch_tpu
import torchvision.models as models
import torch.optim as optim

torch.manual_seed(42)

model = models.resnet50(pretrained=False).half().to("tpu")

model = model.train().to("tpu")

opt = optim.SGD(model.parameters(), lr=0.01)

n = 16
inp         = torch.randn((n, 3, 224, 224),dtype = torch.float16).to("tpu")
target      = torch.randint(0, 1000, (n,), dtype=torch.int64).to("tpu")
loss_fn     = nn.CrossEntropyLoss()

for _ in range(10):
    opt.zero_grad()
    res  = model(inp)
    loss = loss_fn(res, target)
    loss.backward()
    opt.step()
    print(loss.item())
