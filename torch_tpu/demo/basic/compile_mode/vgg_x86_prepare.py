import torch
import os
import pdb
import torch.nn as nn
from scompile.compile.aot import SophonJit
import argparse
import numpy as np
import torchvision.models as models

torch.manual_seed(42)

@SophonJit(model_name="vgg16", chip="bm1684x", fp="fp16", batch=8, trace_joint=True, output_loss_index=0, dump_io=False)
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
batch  = 8
fake_input  = torch.randn((batch, 3, 224, 224),dtype = torch.float32)
fake_target = torch.randint(0, 1000, (batch,), dtype=torch.int64)
# 使用compile触发模型编译,注意 只能传入tensor，而且数量和shape必须和要求符合
model.compile(fake_input, fake_target)
# 编译完可以继续使用model