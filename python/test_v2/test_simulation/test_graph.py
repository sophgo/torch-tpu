from torch import nn
import torch


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


class EncodeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.attention = Attention()


    def forward(self, x):
        x = self.attention(x)
        x = self.linear(x)
        return x


class Transformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.lm_head = nn.Linear(10, 10)
        self.encode_layers = nn.ModuleList([EncodeLayer() for _ in range(6)])

    def forward(self, x):
        for layer in self.encode_layers:
            x = layer(x)
        x = self.lm_head(x)
        return x


def test_graph():
    x = torch.rand(10, 10)
    model = Transformer()
    y = model(x)
    from torch_tpu.utils.reflection.recorder import print_graph_summary

    breakpoint()
    print_graph_summary()
