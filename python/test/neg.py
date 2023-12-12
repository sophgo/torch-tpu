import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../build/torch_tpu/libtorch_tpu.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"


def case_neg():
    x = torch.randn((2, 52, 64, 45, 64), dtype=torch.float16)
    # x = torch.randint(0, 128, (2, 3, 4), dtype=torch.uint8)
    out_cpu = torch.neg(x)
    out_tpu = torch.neg(x.to(device))

    # print(f"cpu out: {out_cpu}")
    # print(f"tpu out: {out_tpu.cpu()}")

    print(f"max diff: {torch.max(abs(out_cpu - out_tpu.cpu()))}")


if __name__ == "__main__":
    case_neg()