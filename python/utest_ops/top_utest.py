import torch
import torch.nn as nn
import torchgen
import yaml
import copy
import time
from inspect import getmembers, isfunction
import os
import subprocess
import numpy as np
import pathlib

# Over 2k Ops in Torch Now
# https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/native_functions.yaml
def get_torch_ops_support():
    path_native_yaml = str(torchgen.__path__[0])+ "/packaged/ATen/native/native_functions.yaml"
    with open(path_native_yaml, "r") as stream:
        try:
            yaml_string =yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
        function_list = []
        for i in range(len(yaml_string)): #nums=2493
            function_list += [yaml_string[i]['func'].split("(")[0].split(".")[0]]
        torch_ops = [*set(function_list)] #nums=1419

def get_TPU_ops_support():
    path_tpu_train_firmware_core = "../../firmware_core/src"
    TPU_ops_train = os.listdir(path_tpu_train_firmware_core)
    for i in range(len(TPU_ops_train)):
        TPU_ops_train[i]=TPU_ops_train[i].split(".c")[0].split("nodechip_")[1]
    print(TPU_ops_train)

# move a tensor/dict/list structto target device and dtype
def move_to(obj, device, dtype):
  if torch.is_tensor(obj):
    return obj.to(device=device,dtype=dtype)
  elif isinstance(obj, dict):
    res = {}
    for k, v in obj.items():
      res[k] = move_to(v, device, dtype)
    return res
  elif isinstance(obj, list):
    res = []
    for v in obj:
      res.append(move_to(v, device, dtype))
    return res
  elif isinstance(obj, tuple):
    res = ()
    for v in obj:
      res = res + (move_to(v, device, dtype),)
    return res
  elif isinstance(obj, int) or isinstance(obj, float):
     return obj
  else:
    raise TypeError("Invalid type for move_to")

# the total nums of output for a given model
def compute_output_num(obj):
  if torch.is_tensor(obj):
    return 1
  elif isinstance(obj, list):
    return len(obj)
  elif isinstance(obj, tuple):
    return len(obj)
  else:
    raise TypeError("Invalid type for compute_output_num")

#convert readable dtyoe to torch dtype
def convert_dtype(dtype):
   convert_table = {'f32':torch.float32,
                    'i32':torch.int32,
                    'f16':torch.half}

   if dtype not in convert_table.keys():
      assert 0,"{} not support".foramt(dtype)
   return convert_table[dtype]

#convert compute error and check whether error is larger than epsilon
def metric_compute(tensor_a, tensor_b, mode, epsilon):
   assert tensor_a.shape==tensor_b.shape
   if mode=="max_diff":
      temp = torch.max(torch.abs(tensor_a-tensor_b)).item()
      return  temp,  temp < epsilon
   if mode=="MAE":
      temp =  torch.sum(torch.abs(tensor_a-tensor_b))/torch.numel(tensor_a)
      temp = temp.item()
      return  temp,  temp < epsilon
   raise TypeError("Invalid mode for metric_compute")


#cmp every output,  nodechip vs torch
#output: #output_nums for 1 sample in 1 dtype {"metric_value", "metric_flag", "passed_flag"}
def cmp_output(metric_table,output_cpu, output_tpu, dtype, epsilon=0.01):
 num_output_cpu = compute_output_num(output_cpu)
 num_output_tpu = compute_output_num(output_tpu)
 assert (num_output_cpu==num_output_tpu)
 all_output_dict_for_one_sample_one_dtype = []
 for i in range(num_output_tpu):
   dict_each_output_value = {}
   dict_each_output_CMP_flag= {}
   for j in range(len(metric_table)):
     dict_each_output_value[metric_table[j]],dict_each_output_CMP_flag[metric_table[j]] = metric_compute(output_cpu[i], output_tpu[i],metric_table[j], epsilon)

   temp_cmp_pass_flag = True
   for j in range(len(metric_table)):
      temp_cmp_pass_flag = temp_cmp_pass_flag and (dict_each_output_CMP_flag[metric_table[j]])

   temp_result = { "metric_value":dict_each_output_value, "metric_flag":dict_each_output_CMP_flag, "passed_flag_one_sample_one_dtype":temp_cmp_pass_flag}
   all_output_dict_for_one_sample_one_dtype+=[temp_result]
 return all_output_dict_for_one_sample_one_dtype

def dump_path_gen(case_name):
   outfile_path = str(pathlib.Path().resolve()) + "/" + os.environ['CHIP_ARCH']
   new_path = outfile_path + "/" + case_name
   if not os.path.exists(new_path):
      os.makedirs(new_path)
   return new_path

def dump_network_info():
   #TODO
   return 0

def dump_tensor(dump_count, chip_case_path, tensor_tpu, tensor_cpu,dtype):
   ouput_tpu = move_to(tensor_tpu, "cpu",dtype) #tensor_tpu.cpu().numpy()
   output_cpu = move_to(tensor_cpu, "cpu",dtype) #tensor_cpu.cpu().numpy()
   tensor_path = chip_case_path + "/" + "dump_{}_sample_{}".format(str(dtype).split(".")[-1], dump_count)
   np.savez(tensor_path, ouput_tpu=ouput_tpu, output_cpu=output_cpu)

def dump_info(dump_count, case_name,tensor_tpu, tensor_cpu,dtype):
    chip_case_path = dump_path_gen(case_name)
    dump_network_info()
    dump_tensor(dump_count, chip_case_path, tensor_tpu, tensor_cpu, dtype)

# test a single input
def Torch_Test_Forward_Function_Per(sample_count, case_name, module_native, input_sample,  metric_table = ['max_diff','MAE']
               , epsilon_dict = {'f32':1e-6,'f16':1e-2,'i32':1e-6}, seed=1000, dump = False):\

    naive_dtype_list =list(epsilon_dict.keys())
    assert len(naive_dtype_list)>=0

    ####collect computing data
    final_result = {}
    for naive_dtype in naive_dtype_list:
        torch_dtype = convert_dtype(naive_dtype)
        device = "privateuseone"#"TODO: privateuseone"
        torch.ops.load_library("../../libtorch_plugin/build/liblibtorch_plugin.so")
        #### You cannot set seed here because the rand input has been given
        #### torch.manual_seed(seed)

        net_cpu = module_native()
        output_cpu = net_cpu(*input_sample)

        input_sample_tpu= move_to(input_sample, device,torch_dtype)
        net_tpu = copy.deepcopy(net_cpu).to(device)
        output_tpu = net_tpu(*input_sample_tpu)
        output_tpu = move_to(output_tpu, "cpu",torch_dtype)
        final_result[naive_dtype] = cmp_output(metric_table, output_cpu, output_tpu, torch_dtype, epsilon_dict[naive_dtype])
        pass_flag_all_outputs = True
        output_num = len(final_result[naive_dtype])
        for i in range(output_num):
           pass_flag_all_outputs = pass_flag_all_outputs and final_result[naive_dtype][i]["passed_flag_one_sample_one_dtype"]
        if dump and not pass_flag_all_outputs:
           dump_info(sample_count, case_name, output_cpu,output_tpu,torch_dtype)

    ###Value Cmp
    count = 0
    is_correct_dtype = {}
    for naive_dtype in naive_dtype_list:
       temp_result = False
       for each_metric in metric_table:
           temp_result = temp_result or  final_result[naive_dtype][0]['metric_flag'][each_metric]
       output_num = len(final_result[naive_dtype])

       if (temp_result):
         count = count +1
         for idx_output in range(output_num):
           print("[Result]",case_name,"{}  output-{} is success".format(naive_dtype, idx_output), final_result[naive_dtype][0]['metric_value'])
         is_correct_dtype[naive_dtype] = 1
       else:
         for idx_output in range(output_num):
           print("[Result]",case_name, "{}  output-{} is failed".format(naive_dtype, idx_output), final_result[naive_dtype][0]['metric_value'])
         is_correct_dtype[naive_dtype] = 0

    return count ==len(naive_dtype_list),is_correct_dtype



# print test info for all inputs, which means each input sharing same seed
def print_info(case_name, metric_table,epsilon_dict,seed):
    print(("*****************Test {} START INFO*************************").format(case_name))
    print("Seed:                   ", seed)
    print("Metrics for Compare:    ",  metric_table)
    print("Dtype and allowed error:", epsilon_dict)


# print shape for each inputs
def print_each_input_batch_shape(inputs):
   for idx,input_idx in enumerate(inputs):
      if torch.is_tensor(input_idx):
        print("Input {} shape is {}".format(idx, input_idx.shape))
      else:
        print("Input {} is 0-dim value".format(idx))

def Torch_Test_Forward_Function(case_name, module_native, input_sample_collection,  metric_table = ['max_diff','MAE']
               , epsilon_dict = {'f32':1e-6,'f16':1e-2,'i32':1e-6}, seed=1000, dump=False):

    input_sample_processed = input_sample_collection
    naive_dtype_list =list(epsilon_dict.keys())
    if isinstance(input_sample_collection, dict):
        input_sample_processed = list(input_sample_collection.values())
    elif isinstance(input_sample_collection, list):
       input_sample_processed =  input_sample_collection

    print_info(case_name, metric_table,epsilon_dict,seed)
    current_correct_num = 0
    static_corret_dtype_case = dict.fromkeys(naive_dtype_list, 0)
    for idx, input_sample in enumerate(input_sample_processed):
        print("--------------------------------------------------------")
        print("Case {} Sample {} is started".format(case_name, idx))
        print("--------------------------------------------------------")

        print_each_input_batch_shape(input_sample)
        temp_flag,is_correct_dtype = Torch_Test_Forward_Function_Per(idx, case_name, module_native, input_sample,  metric_table
               , epsilon_dict, seed, dump)
        current_correct_num += temp_flag == 1
        for dtype_per in naive_dtype_list:
            if (is_correct_dtype[dtype_per]):
                static_corret_dtype_case[dtype_per] +=1
    print("*****************Basic Stats CMP INFO*************************")
    correct_dtype_list = []
    for i in naive_dtype_list:
       if static_corret_dtype_case[i] == len(input_sample_processed):
            print("Case {}: dtype {} is all corrected".format(case_name, i))
            correct_dtype_list+=[i]
    wrong_dtype = list(set(naive_dtype_list) - set(correct_dtype_list))
    if (len(wrong_dtype)>0):
        print("Case {}: dtype {} exist errors".format(case_name, wrong_dtype))
    print("Case {}: {}/{} Samples is completely corrected".format(case_name, current_correct_num, len(input_sample_processed)))
    print(("*****************Test {} All End*************************").format(case_name))




