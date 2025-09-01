import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "tpu:0"

def case1():
    scale = torch.ones([1]).to(device)
    growth_tracker = torch.zeros([1], dtype=torch.int32).to(device)
    found_inf_combined = torch.ones([1]).to(device)
    backoff_factor = 0.5
    growth_interval = 2000
    growth_factor = 2.0
    torch._amp_update_scale_(scale,
                            growth_tracker,
                            found_inf_combined,
                            growth_factor,
                            backoff_factor,
                            growth_interval)
    print("scale:", scale.cpu())
    print("growth_tracker:", growth_tracker.cpu())

def case2():
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.linear_relu_stack = nn.Sequential(
                nn.Linear(28*28, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )

        def forward(self, x):
            x = self.flatten(x)
            logits = self.linear_relu_stack(x)
            return logits
    x = torch.randn((4, 28*28)).to(device).half()
    model = NeuralNetwork().to(device).half()

    y = model(x)
    y_grad = torch.rand_like(y).to(device).half()
    y.backward(y_grad)
    import pdb; pdb.set_trace()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
    import pdb; pdb.set_trace()


if __name__ == "__main__":
    #case1()
    case2()