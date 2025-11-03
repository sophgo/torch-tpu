import torch
import torch.nn as nn
import torch.nn.functional as F
from top_utest import set_requires_grad,move_to, set_bacis_info, Tester_Basic

def case1():
    #step0:set seed
    seed=1000
    set_bacis_info(seed)
    device = torch.device("privateuseone:0")

    #step1: define test model
    batch_size = 4
    length = 10
    embed_dim = 32
    intermediate_size = 128

    class GPT2Mlp(nn.Module):
        def __init__(self, embed_dim, intermediate_size):
            super().__init__()
            self.c_fc = nn.Linear(embed_dim, intermediate_size)
            self.c_proj = nn.Linear(intermediate_size, embed_dim)
            self.act = F.gelu

        def forward(self, x):
            x = self.c_fc(x)
            x = self.act(x)
            x = self.c_proj(x)
            return x


    class MlpFunc(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, w1, w2, b1, b2):
            B, M, N = x.shape
            D1 = w1.shape[1]
            D2 = w2.shape[1]
            assert w1.shape == (N, D1)
            assert w2.shape == (D1, D2)
            assert b1.shape == (D1,)
            assert b2.shape == (D2,)
            out1 = torch.empty(B, M, D1).type_as(x).to(device)
            p = torch.empty(B, M, D1).type_as(x).to(device)
            out2 = torch.empty(B, M, D2).type_as(x).to(device)

            torch.ops.my_ops.mlp_forward(x,
                                        w1,
                                        w2,
                                        b1,
                                        b2,
                                        out1,
                                        p,
                                        out2)

            ctx.save_for_backward(x, w1, w2, out1, p)

            return out2

        @staticmethod
        def backward(ctx, grad_output):
            x, w1, w2, out1, p = ctx.saved_tensors

            B, M, N = x.shape
            D1 = w1.shape[1]
            D2 = w2.shape[1]
            grad_output.to(device)
            grad_x = torch.ones(x.shape, dtype = x.dtype).to(device)
            grad_w1 = torch.ones(w1.shape, dtype = x.dtype).to(device)
            grad_w2 = torch.ones(w2.shape, dtype = x.dtype).to(device)

            grad_b1 = torch.ones((D1,), dtype = x.dtype).to(device)
            grad_b2 = torch.ones((D2,), dtype = x.dtype).to(device)

            torch.ops.my_ops.mlp_backward(grad_output,
                                            x,
                                            w1,
                                            w2,
                                            out1,
                                            p,
                                            grad_x,
                                            grad_w1,
                                            grad_w2,
                                            grad_b1,
                                            grad_b2)

            return grad_x, grad_w1, grad_w2, grad_b1, grad_b2


    class MlpBlock(nn.Module):
        def __init__(self):
            super().__init__()
            # self.w1 = w1.requires_grad_(True).to(device)
            # self.w2 = w2
            # self.b1 = b1
            # self.b2 = b2

        def forward(self, x, w1, w2, b1, b2):
            return MlpFunc.apply(x, w1, w2, b1, b2)

    GPT_2 = GPT2Mlp(embed_dim, intermediate_size)

    w1 = GPT_2.state_dict()['c_fc.weight'].clone().detach().transpose(0,1).contiguous()#.requires_grad_(True).to(device)   # TODO
    b1 = GPT_2.state_dict()['c_fc.bias'].clone().detach()#.requires_grad_(True).to(device)
    w2 = GPT_2.state_dict()['c_proj.weight'].clone().detach().transpose(0,1).contiguous()#.requires_grad_(True).to(device)
    b2 = GPT_2.state_dict()['c_proj.bias'].clone().detach()#.requires_grad_(True).to(device)


    MLP_Module = MlpBlock()

    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    input_data = [
         [ torch.randn(batch_size, length, embed_dim, requires_grad=True),w1, w2, b1, b2],
    ]
    metric_table = ['max_diff','MAE']
    chip_arch_dict = {"bm1684x":0, 'sg2260':0}
    epsilon_dict = {'bm1684x':{'f32':1e-6,'f16':1e-2},'sg2260':{'f32':1e-6,'f16':1e-2}}
    case_name =  __file__.split('.py')[0]# You can change your name
    dump_flag = True #it will dump alll wrong cases

    # you can write your own forward and backward fucntion
    #assuming net_tpu and input_sample_tpu has loaded on tpu
    #In this way you can define backward propagation by your self.
    def customized_execute_function(input_sample_cpu, input_sample_tpu, net_cpu, net_tpu, dtype):
        set_requires_grad(input_sample_cpu, True)
        set_requires_grad(input_sample_tpu, True)

        #in this case, data is from another model
        net_cpu =GPT_2
        x_cpu = input_sample_cpu[0]
        output_cpu = net_cpu(x_cpu)
        output_tpu = net_tpu(*input_sample_tpu)

        grad_o = torch.ones(output_cpu.shape)
        grad_o_tpu =  move_to(grad_o, device, dtype)  #grad_o.to(device)

        output_cpu.backward(grad_o)
        output_tpu.backward(grad_o_tpu)
        #tpu-first
        return input_sample_tpu[0].grad, input_sample_cpu[0].grad #Notice [0] because input_data has [],[]

    My_Tester = Tester_Basic(case_name, chip_arch_dict, device, metric_table, epsilon_dict,seed, dump_flag)
    My_Tester.customized_execute_function = customized_execute_function
    return My_Tester.Torch_Test_Execution_Function(MLP_Module.cpu(), input_data)


if __name__ == "__main__":
    #This example shows all  [], [[]] is acceptablep
    case1()

#######################
##  case1():forward + backward [[T]]
########################