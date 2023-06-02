import torch

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)

if __name__ == "__main__":
    device = "privateuseone"
    batch = 32
    sequence = 1024
    head_size = 12
    max_position = 1024
    dtype_list = ["float32", "float16"]

    for t in dtype_list:
        if t == "float32":
            attn_weights = torch.rand(batch, head_size, sequence, sequence).to("privateuseone")
            mask_value   = torch.tensor(-1e4).to("privateuseone")
        elif t == "float16":
            attn_weights = torch.rand(batch, head_size, sequence, sequence).to("privateuseone").half()
            mask_value   = torch.tensor(-1e4).to("privateuseone").half()
        else:
            continue

        casual_mask = torch.tril(torch.ones((max_position, max_position),dtype=torch.uint8)) \
                        .view(1,1,max_position, max_position).to("privateuseone")

        attn_weight_ = torch.where(casual_mask.bool(), attn_weights, mask_value)
        attn_weight_ = attn_weight_.float().to("cpu")
        print(attn_weight_)