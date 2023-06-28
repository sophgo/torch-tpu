import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel
from utils import compare_model_grad, Optimer
PLUGIN_PATH = "../../libtorch_plugin/build/liblibtorch_plugin.so"
TPU = "privateuseone:0"
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
    configure.n_layer= 12
    configure.activation_function= "gelu"

    batch = 8
    sequence = 1024
    max_grad_norm = 1.0

    inp = torch.randint(0,configure.vocab_size,(batch, sequence))
    inp_tpu = inp.clone().int().to(TPU)

    net = GPT2LMHeadModel(configure).train()
    net_tpu = copy.deepcopy(net)
    net_tpu.to(TPU).half()

    print("start forward")
    t1 = time.time()
    out_cpu = net(input_ids = inp, labels = inp)
    t2 = time.time()
    print("cpu time :", t2 - t1)

    t1 = time.time()
    optimer.reset()
    out_tpu = net_tpu(input_ids = inp_tpu, labels = inp_tpu)
    optimer.dump()
    t2 = time.time()
    print("tpu time :", t2 - t1)

    print("start backward")
    t1 = time.time()
    out_cpu["loss"].backward()
    t2 = time.time()
    print("cpu time :", t2 - t1)

    t1 = time.time()
    optimer.reset()
    out_tpu["loss"].backward()
    optimer.dump()
    t2 = time.time()
    print("tpu time :", t2 - t1)

    t1 = time.time()
    torch.nn.utils.clip_grad_norm_(net.parameters(), max_grad_norm)
    t2 = time.time()
    print(" cpu clip_grad_norm time : ", t2 - t1)
    
    t1 = time.time()
    optimer.reset()
    torch.nn.utils.clip_grad_norm_(net_tpu.parameters(), max_grad_norm)
    optimer.dump()
    t2 = time.time()
    print(" tpu clip_grad_norm time : ", t2 - t1)

    print(" ======== compare model's parameter grad =======")
    #compare_model_grad(net, net_tpu)

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
    #my_print(out_cpu, out_tpu)



