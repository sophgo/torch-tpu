import torch
import torch.nn as nn
import torch.nn.functional as F

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
torch.set_printoptions(precision=6)
device = "privateuseone:0"

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


if __name__ == "__main__":
    case1()