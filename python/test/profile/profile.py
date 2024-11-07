import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_tpu
torch.manual_seed(1000)
device = "tpu:0"

def profile_case():
    B = 6
    S = 512
    H = 128
    max_record_num = 100

    a = torch.randn((B,S,H))
    b = torch.randn((S,H))

    # test fp16
    a_tpu = a.half().to(device)
    b_tpu = b.half().to(device)
    # part 0
    torch.ops.my_ops.enable_profile(max_record_num, True)  # enable profile with cmd
    # torch.ops.my_ops.enable_profile(max_record_num, False)  # enable profile without cmd (pure pmu) 
    _ = a_tpu * b_tpu
    torch.ops.my_ops.disable_profile() # disable profile and dump data (cdm_profile_data_dev0-0)

    # part 1
    torch.ops.my_ops.enable_profile(max_record_num, False)
    _ = a_tpu + b_tpu
    torch.ops.my_ops.disable_profile() # (cdm_profile_data_dev0-1)
    # part ....

    # use tpu-mlir
    # cd tpu-mlir
    # source envsetup.sh
    # tpu_profile.py cdm_profile_data_dev0-0/ cdm_out_0 --mode perfAI --arch BM1690
    # tpu_profile.py cdm_profile_data_dev0-1/ cdm_out_1 --mode perfAI --arch BM1690
    # ....

if __name__ == "__main__":
    profile_case()
