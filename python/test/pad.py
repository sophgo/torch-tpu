import torch
import torch.nn.functional as F
import torch_tpu

def case1():
    pad = (0, 1, 0, 1)
    hidden_states = torch.randn((1, 128, 128, 128)).tpu().half()
    for i in range(10):
        F.pad(hidden_states, pad, mode="constant", value=0)


if __name__ == "__main__":
    case1()