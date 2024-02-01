import torch

import torch_tpu
torch.manual_seed(1000)

if __name__ == "__main__":
    device = "tpu"
    batch = 1
    sequence = 8
    head_size = 3
    max_position = 8

    attn_weights = torch.rand(batch, head_size, sequence, sequence)
    mask_value   = torch.tensor(-1e4)
    casual_mask = torch.tril(torch.ones((max_position, max_position),dtype=torch.uint8)) \
                        .view(1,1,max_position, max_position)

    attn_weights_tpu =  attn_weights.to(device)
    mask_value_tpu = mask_value.to(device)
    casual_mask_tpu = casual_mask.to(device)


    res_cpu = torch.where(casual_mask, attn_weights, mask_value)
    res_tpu = torch.where(casual_mask_tpu, attn_weights_tpu, mask_value_tpu)
    print("cpu ======")
    print(res_cpu)
    print("tpu ======")
    print(res_tpu.cpu())

    diff = abs(res_cpu - res_tpu.cpu())
    idx = diff.argmax()

    print("max_diff: ", torch.max(diff))
    print("idx: ", idx)
    print("cpu:", res_cpu.flatten()[idx])
    print("tpu:", res_tpu.cpu().flatten()[idx])