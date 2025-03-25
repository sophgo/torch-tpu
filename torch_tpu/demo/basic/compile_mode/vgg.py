from TPUCompile import TPUCompile
import torch
from   torch import nn
import torch.optim as optim
import torchvision.models as models

@TPUCompile(bmodel_path="vgg16_8.bmodel", config_path="vgg16_info.json", device_id=0, chip="bm1684x")
class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = models.vgg16(pretrained=False)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input, target):
        predict = self.model(input)
        loss = self.loss_fn(predict.float(), target.long())
        return loss, predict.detach()

model  = Model()
opt    = optim.SGD(model.parameters(), lr=0.01)
n      = 8
inp    = torch.randn((n, 3, 224, 224),dtype = torch.float16)
target = torch.randint(0, 1000, (n,), dtype=torch.int64)

for _ in range(10):
    opt.zero_grad()
    res = model(inp, target)
    loss = res[0]
    print(loss.item())
    loss.backward()
    opt.step()

model.model.cpu()
torch.save(model.state_dict(), "after_vgg16.pth")