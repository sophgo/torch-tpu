import numpy as np
import torch
from utils import compare_model_grad, compare_model_weight, get_model_weight, Optimer
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel, GPT2Model

torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
torch.manual_seed(1000)
optimer = Optimer("../../libtorch_plugin/build/liblibtorch_plugin.so")


if __name__ == "__main__":
    from transformers import GPT2Config
    import copy
    import time

    # ====== configure ======
    configure = GPT2Config()
    configure.attn_pdrop = 0
    configure.embd_pdrop = 0
    configure.resid_pdrop = 0
    configure.n_layer= 12
    configure.activation_function= "gelu"
    batch = 32
    sequence = 256
    # ====== configure ======

    ## 1.input
    inp = torch.randint(0,configure.vocab_size,(batch, sequence))
    inp_tpu = inp.clone().int().to("privateuseone:0")
    ref = torch.rand(batch, sequence, configure.hidden_size)
    ref_tpu = ref.to("privateuseone:0").half()

    ## 2.network
    net = GPT2LMHeadModel(configure).train()
    #net = GPT2Model(configure)
    net_tpu = copy.deepcopy(net)
    net_tpu.to("privateuseone:0").half()
    #net_tpu.lm_head.weight = net_tpu.transformer.wte.weight

    ## 3.optimizer
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.01)
    optimizer_tpu = torch.optim.AdamW(net_tpu.parameters(), lr = 0.01)

    ## 4. run
    print("start run")
    net.train()
    net_tpu.train()
    optimizer.zero_grad()
    optimizer_tpu.zero_grad()
    ## 4.1 forward 
    t1 = time.time()
    optimer.reset()
    for i in range(1):
        out_tpu = net_tpu(input_ids = inp_tpu, labels = inp_tpu)
    #out_tpu = net_tpu(input_ids = inp_tpu)
    optimer.dump()
    t2 = time.time()
    print("tpu time :", t2 - t1)

    t1 = time.time()
    out_cpu = net(input_ids = inp, labels = inp)
    #out_cpu = net(input_ids = inp)
    t2 = time.time()
    print("cpu time :", t2 - t1)
    
    ## 4.2 backward
    t1 = time.time()
    optimer.reset()
    out_tpu["loss"].backward()
    #out_tpu[0].backward(ref_tpu)
    optimer.dump()

    t2 = time.time()
    print("[backward] tpu time :", t2 - t1)

    t1 = time.time()
    out_cpu["loss"].backward()
    #out_cpu[0].backward(ref)
    t2 = time.time()
    print("[backward] cpu time :", t2 - t1)

    #model1_weight =  get_model_weight(net_tpu)
    #compare_model_weight(net, net_tpu)
    #compare_model_grad(net, net_tpu)
    ## 4.3 update parameter
    print("=====update param===")
    optimizer.step()
    optimer.reset()
    optimizer_tpu.step()
    optimer.dump()
    #model2_weight =  get_model_weight(net_tpu)
    # for k in model1_weight.keys():
    #     c_g = model1_weight[k]
    #     t_g = model2_weight[k]
    #     diff = abs(c_g - t_g)
    #     index_abs = diff.argmax()
    #     related_diff = abs(diff/c_g)
    #     index_related = related_diff.argmax()
    #     print(k, 
    #             ",max abs diff: ", np.max(diff), " exp:", c_g.flatten()[index_abs], ", got:", t_g.flatten()[index_abs],
    #             ",max rel diff: ", np.max(related_diff), ", exp: ", c_g.flatten()[index_related], ", got:", t_g.flatten()[index_related]
    #         )
    # compare_model_weight(net, net_tpu)
    import pdb;pdb.set_trace()

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


