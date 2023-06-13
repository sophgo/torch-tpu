import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2Model
from utils import compare_model_grad, Optimer
PLUGIN_PATH = "../../libtorch_plugin/build/liblibtorch_plugin.so"
torch.ops.load_library(PLUGIN_PATH)
optimer = Optimer(PLUGIN_PATH)


if __name__ == "__main__":
    from transformers import GPT2Config
    import copy
    import time

    configure = GPT2Config()
    configure.attn_pdrop = 0
    configure.embd_pdrop = 0
    configure.resid_pdrop = 0
    configure.n_layer= 1
    configure.activation_function= "gelu"

    batch = 32
    sequence = 256

    inp = torch.randint(0,configure.vocab_size,(batch, sequence))
    inp_tpu = inp.clone().int().to("privateuseone:0")
    ref = torch.rand(batch, sequence, configure.hidden_size)
    ref_tpu = ref.to("privateuseone:0").half()


    net = GPT2Model(configure).train()
    net_tpu = copy.deepcopy(net)
    net_tpu.to("privateuseone:0").half()

    print("start forward")
    t1 = time.time()
    out_cpu = net(input_ids = inp)
    t2 = time.time()
    print("cpu time :", t2 - t1)

    t1 = time.time()
    optimer.reset()
    out_tpu = net_tpu(input_ids = inp_tpu)
    optimer.dump()
    t2 = time.time()
    print("tpu time :", t2 - t1)

    print("start backward")
    t1 = time.time()
    out_cpu[0].backward(ref)
    t2 = time.time()
    print("cpu time :", t2 - t1)

    t1 = time.time()
    optimer.reset()
    out_tpu[0].backward(ref_tpu)
    optimer.dump()
    t2 = time.time()
    print("tpu time :", t2 - t1)

    print(" ======== compare model's parameter grad =======")
    compare_model_grad(net, net_tpu)

    def my_print(out_cpu, out_tpu):
        for i in range(len(out_cpu)):
            o_c = out_cpu[i]
            if isinstance(o_c, torch.Tensor):
                o_t = out_tpu[i].to("cpu")
                print("cpu:")
                #print(o_c)
                print("tpu:")
                #print(o_t)
                print(torch.max(abs(o_c - o_t)))
            elif isinstance(o_c, tuple):
                my_print(out_cpu[i], out_tpu[i])
            else:
                return
    my_print(out_cpu, out_tpu)



