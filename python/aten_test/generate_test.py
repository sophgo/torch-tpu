import argparse
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from parse_json import str2list
from python.utest_ops.top_utest import  set_bacis_info, Tester_Basic, set_requires_grad, move_to

def get_model(case_name: str)-> callable:
    try:
        return getattr(torch.ops.aten, case_name)
    except:
        assert(f"something wrong in case {case_name}, please check the case name in https://github.com/pytorch/pytorch/blob/v2.1.0/aten/src/ATen/native/native_functions.yaml")

def gen_data(shape: tuple, type: str)-> torch.tensor:
    if type in ["float","float32"]:
        return torch.randn(shape)/2.0 #improve F16 precision
    if type in ["half","float16"]:
        return torch.randn(shape).half()
    if type in ["int","int32"]:
        return torch.randint(0, 10, shape, dtype=torch.int32)
    if type in ["long","long int","int64"]:
        return torch.randint(0, 10, shape, dtype=torch.int64)
    if type == "bool":
        return (torch.rand(shape)>0.5)
    if type in ["Scalar", "ScalarList"]:
        return torch.tensor(shape)
    assert(f"not support type {type}")

def case_default():
    seed=1000
    set_bacis_info(seed)

    #step1: define test model
    class Test_Module(nn.Module):
            def __init__(self):
                super(Test_Module, self).__init__()
            def forward(self, a1):
                return  a1.abs()

    #step2: prepare input data, Notice that the input data will be adopted not only their shapes
    #list is also acceptable
    input_data = [
        [torch.rand((30, 2500, 500)) - 0.5],
    ]
    metric_table = ['max_diff','MAE']
    chip_arch_dict = {'sg2260':1}
    epsilon_dict = {'sg2260':{'f32':1e-6,'f16':1e-2}}

    case_name =  __file__.split('.py')[0]# You can change your name
    dump_flag = True #it will dump alll wrong cases
    device = torch.device("privateuseone:0")

    My_Tester = Tester_Basic(case_name, chip_arch_dict, device, metric_table, epsilon_dict,seed, dump_flag)
    return My_Tester.Torch_Test_Execution_Function(Test_Module(), input_data)

def case_json(args):
    json_file = args.json
    assert os.path.exists(json_file), f"Path {json_file} does not exist"
    include_cases = ['_convolution' if x == 'convolution' else x for x in args.cases]
    need_test_include_cases = len(include_cases) > 0
    with open(json_file, 'r', encoding='utf-8') as file:
        cases = json.load(file)

    seed = 1
    set_bacis_info(seed)
    if args.case_id != -1:
        cases = {args.case_id : cases[str(args.case_id)]}
    fail_ops = []

    for key,case in cases.items():
        name = case['name']
        if need_test_include_cases and (name not in include_cases):
            continue

        # step1: define test model
        class Test_Module(nn.Module):
            def __init__(self, name, params):
                super(Test_Module, self).__init__()
                self.model = get_model(name)
                self.params = params
            def forward(self, *a):
                return self.model(*a,*self.params)

        #step2: prepare input data, Notice that the input data will be adopted not only their shapes
        input_data = [
            [gen_data(s,d) for s, d in zip(case["shape"], case["dtype"])]
        ]

        if name in ["index_put_", "_index_put_impl_"]:
            rand_idx = [t for t in case["shape"][0]]
            N = torch.randint(0, sum(rand_idx[:-1]), (1, ))
            indices = [torch.randint(0, x, (N,)) for x in rand_idx[:-1]]
            values = torch.randn(N, rand_idx[-1])
            input_data[0].append(indices)
            input_data[0].append(values)

        metric_table = ['max_diff','MAE']
        chip_arch_dict = {'sg2260':1}

        epsilon_dict = {'sg2260':{'f32':1e-4,'f16':1e-2}}
        # epsilon_dict = {'sg2260':{'f16':1e-2}}
        if name in ["linear", "addmm", "mul", "max_pool2d_with_indice"]:
            epsilon_dict = {'sg2260':{'f16':1e-2}}
        if name == "atan":
            epsilon_dict = {'sg2260':{'f32':1e-4}}
        if name == "_convolution" and os.getenv("TORCHTPU_CONV_OP_DTYPE"):
            epsilon_dict = {'sg2260':{'f32':1e-2,'f16':1e-2}}
        if name == "convolution_backward" and os.getenv("TORCHTPU_CONVBWD_OP_DTYPE"):
            epsilon_dict = {'sg2260':{'f32':1e-2,'f16':1e-2}}

        print(f"#### test_id: {key} case {name} #####")
        dump_flag = False #no support multi-output dump now
        device = torch.device(f"privateuseone:{args.device_id}")
        My_Tester = Tester_Basic(name, chip_arch_dict, device, metric_table, epsilon_dict,seed, dump_flag)
        if name in ["max_pool2d_with_indices"]:
            def customized_execute_function(input_sample_cpu, input_sample_tpu, net_cpu, net_tpu, dtype):
                set_requires_grad(input_sample_cpu, True)
                set_requires_grad(input_sample_tpu, True)

                output_cpu = net_cpu(*input_sample_cpu)
                output_tpu = net_tpu(*input_sample_tpu)

                grad_o = torch.randn(output_cpu[0].shape) #multi output,chose output[0]
                grad_o_tpu =  move_to(grad_o, device, dtype)

                output_cpu[0].backward(grad_o)
                output_tpu[0].backward(grad_o_tpu)
                return input_sample_tpu[0].grad, input_sample_cpu[0].grad
            My_Tester.customized_execute_function = customized_execute_function
        test_res = My_Tester.Torch_Test_Execution_Function(Test_Module(name,case["params"]), input_data)
        if test_res:
            print(f"case {name} pass\n")
        else:
            print(f"case {name} fail\n")
            fail_ops.append(key)

    if len(fail_ops)>0:
        print("fail cases:")
        for k in fail_ops:
            print(f"id {k}: {cases[k]}")
        print("you can run single case by --case_id xxx")
    else:
        print("all cases pass!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", default="test_ops.json", help="load json file for test cases")
    parser.add_argument("--case_id", type=int, default=-1, help="index of case in test_json")
    parser.add_argument("--cases", type=str2list, default=list(),
                        help="If set, will test only given ops. i.e. _convolution,convolution_backward,max_pool2d_with_indices")
    parser.add_argument("--device_id", type=int, default=0, help="choose tpu device id")
    args = parser.parse_args()

    if args.json:
        case_json(args)
    else:
        case_default()