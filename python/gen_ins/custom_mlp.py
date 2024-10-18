import sys
import os
import torch

torch.set_num_threads(1)

BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
from custom_op.mlp import MlpBlock
from utils import ForwardHack, BackwardHack, DumpIns
DI = DumpIns()


if __name__ == "__main__":
    device = "tpu"
    B = 6
    sequence = 4096
    hidden_size = 12288
    TP = 16
    DI.dump("init")
    inp = torch.randn((B, sequence, hidden_size)).to(device).half()
    net = MlpBlock(int(hidden_size), int(hidden_size*4/TP), True, True).to(device).half()
    DI.dump("Custom_Mlp")
    out = net(inp)