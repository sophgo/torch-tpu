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
import torch_tpu
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

def set_bacis_info(seed):
    torch.manual_seed(seed)
    torch.set_printoptions(precision=6)

#move a tensor/dict/list structto target device and dtype
#case support : T, int/float, list, tuple, dict
def move_to(obj, device, dtype):
  assert obj is not None
  if torch.is_tensor(obj):
     origin_dtype = obj.dtype
     if origin_dtype != torch.int32 and origin_dtype != torch.int64:
        obj = obj.to(device,dtype=dtype)
        assert obj.device ==device, (obj.device,device )
     elif origin_dtype == torch.int32:
        obj = obj.to(device)
        assert obj.device ==device, (obj.device,device )
        assert obj.dtype ==torch.int32, (obj.dtype  )
        print('\033[95m' + "[Warning] This tensor-dtype is {}, ensure such T.to is under your consideration".format(obj.dtype)+ '\033[0m')
     elif origin_dtype == torch.int64:
        obj = obj.to(device)
        assert obj.device ==device, (obj.device,device )
        assert obj.dtype ==torch.int64, (obj.dtype  )
        print('\033[95m' + "[Warning] This tensor-dtype is {}, ensure such T.to is under your consideration".format(obj.dtype)+ '\033[0m')
     return obj
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

#set T or [T,T] requires_grad
# This function can only apply in customized_execute_function
# And Must nearest to input_sample_isolation in case of leaf nodes
def set_requires_grad(obj, grad_flag= True):
  if torch.is_tensor(obj):
     obj.requires_grad = grad_flag
     return obj
  elif isinstance(obj, list):
    res =[]
    for idx, i in enumerate(obj):
       if i.dtype==torch.int32 or i.dtype==torch.int64:
          print('\033[95m' + "[Warning] Maybe input-{}-th is label as its  dtype is {}, do not set it in <Test_Module>, set it inside <Torch_Test_Execution_Function>!".format(idx, i.dtype)+ '\033[0m')
       else:
          i.requires_grad = grad_flag
       res +=[i]
    return res
  else:
    raise TypeError("Invalid type for move_to")

# Just a dump function
class Dumper():
  def __init__(self, case_name, device = torch.device("cpu")):
     self.case_name = case_name
     self.chip_case_path = ""
     self.device_cpu =device

  def dump_path_gen(self):
    outfile_path = str(pathlib.Path().resolve()) + "/" + os.environ['CHIP_ARCH']
    new_path = outfile_path + "/" + self.case_name
    if not os.path.exists(new_path):
        os.makedirs(new_path)
    self.chip_case_path = new_path

  def dump_network_info(self):
    #TODO
    return 0

  def dump_tensor(self, dump_count, dict_execute_result, dtype):
    tensor_cpu = dict_execute_result["output_cpu"]
    tensor_tpu = dict_execute_result["output_tpu"]
    ouput_tpu = move_to(tensor_tpu, self.device_cpu,dtype) #tensor_tpu.cpu().numpy()
    output_cpu = move_to(tensor_cpu, self.device_cpu,dtype) #tensor_cpu.cpu().numpy()
    tensor_path = self.chip_case_path + "/" + "dump_{}_sample_{}".format(str(dtype).split(".")[-1], dump_count)
    np.savez(tensor_path, ouput_tpu=ouput_tpu, output_cpu=output_cpu)

  def dump_info(self, dump_count,dict_execute_result,dtype):
      self.dump_path_gen()
      self.dump_network_info()
      self.dump_tensor(dump_count, dict_execute_result, dtype)


class Tester_Basic():
  def __init__(self, case_name = "None", chip_arch_dict = {'bm1684x':0, 'sg2260':0}, device_tpu=None, metric_table = ['max_diff','MAE'], epsilon_dict_multi_arch = {"bm1684x":{'f32':1e-6,'f16':1e-2,'i32':1e-6},"sg2260":{'f32':1e-6,'f16':1e-2,'i32':1e-6}}, seed = 1000, dump_flag = True):
      self.device = device_tpu
      self.device_cpu = torch.device('cpu')
      self.dump_flag = dump_flag
      self.case_name = case_name
      self.Dumper = Dumper(self.case_name, self.device_cpu)

      self.convert_table = {
                      'f32':torch.float32,
                      'i32':torch.int32,
                      'f16':torch.half,
                      'bf16':torch.torch.bfloat16}
      self.mankind_dtype_table = {
                      'fp32': 'f32',
                      'float32': 'f32',
                      'i32':'i32',
                      'fp16':'f16',
                      'bf16':'bf16'}
      self.default_chip_arch = {"bm1684x", "sg2260"}
      self.tested_dtype = set() # no_repeated
      self.metric_table = metric_table

      self.current_chip_arch = os.getenv("CHIP_ARCH")
      self.epsilon_dict_multi_arch = epsilon_dict_multi_arch
      self.epsilon_dict=None
      self.chip_arch_dict = chip_arch_dict
      self.flag_is_such_arch_ready_test = 0
      assert self.chip_arch_dict is not None, (self.case_name)
      self.epsilon_dict_process()
      assert isinstance(epsilon_dict_multi_arch, dict)
      assert isinstance(self.chip_arch_dict, dict)
      assert isinstance(self.epsilon_dict_multi_arch, dict)
      self.seed = seed
      ######customized
      assert isinstance(self.metric_table, list)
      self.num_metric = len(metric_table)
      self.naive_dtype_list =list(self.epsilon_dict.keys())
      self.dtype_num = len(self.naive_dtype_list)
      assert self.dtype_num >0
      self.num_multi_output = 1


  def epsilon_dict_process(self):
      if self.current_chip_arch not in self.chip_arch_dict:
         self.flag_is_such_arch_ready_test = 0
         return
      if self.current_chip_arch not in self.epsilon_dict_multi_arch:
         assert 0,("[ERROR]{} epsilon-setting not supported for {}!".format(self.current_chip_arch, self.case_name))
      self.epsilon_dict = self.epsilon_dict_multi_arch[self.current_chip_arch]
      self.flag_is_such_arch_ready_test = self.chip_arch_dict[self.current_chip_arch]
      if self.flag_is_such_arch_ready_test:
         assert self.epsilon_dict is not None

  # the total nums of output for a given model
  # Usually multi_output is a tuple struct
  # function case:
  # 1) T->1
  # 2) list->len
  # 3) tuple->len
  def compute_multi_output_num(self, obj):
    if torch.is_tensor(obj):
      self.num_multi_output = 1
      return 1
    elif isinstance(obj, list):
      self.num_multi_output = len(obj)
      return len(obj)
    elif isinstance(obj, tuple):
      self.num_multi_output = len(obj)
      return len(obj)
    else:
      raise TypeError("Invalid type for compute_multi_output_num")

  def dtype_mankind_friendly(self,human_dtype):
    if human_dtype in self.convert_table.keys():
       return human_dtype
    if human_dtype not in self.mankind_dtype_table.keys() and human_dtype not in self.convert_table.keys() :
        assert 0,"{} not support".format(human_dtype)
    return self.mankind_dtype_table[human_dtype]

  #convert readable dtyoe to torch dtype
  #function input: dtype 1 str
  #function output: dtype torch.dtype
  def convert_dtype_2_torch_style(self,human_dtype):
    machine_dtype = self.dtype_mankind_friendly(human_dtype)
    return self.convert_table[machine_dtype]


  #convert compute error and check whether error is larger than epsilon
  #only compute between tensor
  #function input:
  # 1)tensor_a, T
  # 2)tensor_b, T
  # 3)mode: 1 str
  # 4)epsilon: 1 float
  #function output:
  #1) metric_value: 1 float
  #2) flag: 1 bool  is metric_value < epsilon?
  def metric_compute(self, tensor_a, tensor_b, mode, epsilon):
    assert not isinstance(tensor_a, list) and not isinstance(tensor_b, list)
    assert not isinstance(tensor_a, tuple) and not isinstance(tensor_b, tuple)

    assert tensor_a.shape==tensor_b.shape, "Shape cpu {} vs tpu {}".format(tensor_a.shape,tensor_b.shape)
    if mode=="max_diff":
        metric_value = torch.max(torch.abs(tensor_a-tensor_b)).item()
        return  metric_value,  metric_value < epsilon
    elif mode=="MAE":
        metric_value =  torch.sum(torch.abs(tensor_a-tensor_b))/torch.numel(tensor_a)
        metric_value = metric_value.item()
        return  metric_value,  metric_value < epsilon
    raise TypeError("Invalid mode for metric_compute")


  # output whehter one output from multi_ouput is passed for all metrics
  # function input: dict  #keys = #metric
  # function output:
  # bool flag: is one output from one multiouput passed all metrics?
  def metric_post_processed(self, dict_each_output_CMP_flag):
      #whether this output is passed multi_metric == & every element in CMP_flag
      temp_cmp_pass_flag_all_metrics_one_ouput = True
      assert len(dict_each_output_CMP_flag)==self.num_metric
      for j in range(self.num_metric):
          temp_cmp_pass_flag_all_metrics_one_ouput &= dict_each_output_CMP_flag[self.metric_table[j]]
      assert temp_cmp_pass_flag_all_metrics_one_ouput==all(dict_each_output_CMP_flag.values())
      return temp_cmp_pass_flag_all_metrics_one_ouput


  #in case that output is 0-dim, where output[i] will be an fault
  #function output:
  ##case 1:  T    - > T
  ##case 2: (T,T,T)  -> (T,T,T)[idx]
  ##case 3: 0-dim  -> 0->dim
  def each_output_0_dim(self,obj,idx):
     if isinstance(obj, tuple):
        assert idx<=len(obj) - 1
        return obj[idx]
     elif obj.dim()==0 :
        assert idx==0
        return obj
     elif torch.is_tensor(obj):
        assert idx==0
        return obj
     else:
        raise TypeError("Invalid mode for each_output_0_dim")

  #cmp every output,  nodechip vs torch
  #function input: dict_execute_result dict {"output_cpu", "output_tpu"}
  #function output: #output_nums for 1 sample in 1 dtype {"metric_value", "metric_flag", "passed_flag"}
  #metric_value:
  #metric_flag: is this mertic < error_allowed
  #passed_flag is this sample of this dtype for all metrics passed?
  def cmp_multi_output(self, dict_execute_result, epsilon=0.01):
    output_cpu = dict_execute_result['output_cpu']
    output_tpu = dict_execute_result['output_tpu']
    num_output_cpu = self.compute_multi_output_num(output_cpu)
    num_output_tpu = self.compute_multi_output_num(output_tpu)
    assert (num_output_cpu==num_output_tpu)
    all_output_dict_for_one_sample_one_dtype = [] #order is output_1,output_2,...,output_n
    #cmp order-1: multi_output
    for i in range(num_output_tpu):
      dict_each_output_value, dict_each_output_CMP_flag = {}, {}
      #cmp order-2:  multi_metric
      for j in range(self.num_metric ):
        output_cpu_processed = self.each_output_0_dim(output_cpu, i)
        output_tpu_processed =self.each_output_0_dim(output_tpu, i)
        dict_each_output_value[self.metric_table[j]],dict_each_output_CMP_flag[self.metric_table[j]] = self.metric_compute(output_cpu_processed,output_tpu_processed,self.metric_table[j], epsilon)

      #whether this output is passed multi_metric == & every element in CMP_flag
      temp_cmp_pass_flag_all_metrics_one_ouput =  self.metric_post_processed(dict_each_output_CMP_flag)
      temp_result_one_output = { "metric_value":dict_each_output_value, "metric_flag":dict_each_output_CMP_flag, "passed_flag_one_sample_one_dtype":temp_cmp_pass_flag_all_metrics_one_ouput}
      # each input sample has cmp_out struct:  [#nums_output,3]
      all_output_dict_for_one_sample_one_dtype +=[temp_result_one_output]
    return all_output_dict_for_one_sample_one_dtype

  def default_execute_function(self, input_sample_cpu, input_sample_tpu, net_cpu, net_tpu, dtype):
     return 0

  #You must put this function nearest to input_sample_isolation
  def customized_execute_function(self, input_sample_cpu, input_sample_tpu, net_cpu, net_tpu, dtype):
    assert not (isinstance(input_sample_cpu, int) or isinstance(input_sample_cpu, float))
    assert torch.is_tensor(input_sample_cpu) or isinstance(input_sample_cpu,list)
    output_cpu = net_cpu(*input_sample_cpu)
    output_tpu = net_tpu(*input_sample_tpu)
    #tpu-first
    return output_tpu, output_cpu

  #move model to target device and type
  #ModuleList, ModuleArray are not supoorted now
  def move_model(self, obj, device, dtype):
    return obj.to(device=device,dtype=dtype)

  #output detach
  #case 1； T->T.detach()
  #case 2: (T,T)->(T.detach(), T.detach())
  def output_isolation(self, dict_execute_result):
    output_cpu = dict_execute_result['output_cpu']
    output_tpu = dict_execute_result['output_tpu']
    if torch.is_tensor(output_cpu):
        dict_execute_result['output_cpu'] = output_cpu.detach()
        dict_execute_result['output_tpu'] = output_tpu.detach()
    elif isinstance(output_cpu, tuple):
       output_cpu_new, output_tpu_new = (), ()
       for idx, obj in enumerate(output_cpu):
          output_cpu_new += (output_cpu[idx].detach(),)
          output_tpu_new += (output_tpu[idx].detach(),)
          dict_execute_result['output_cpu'] = output_cpu_new
          dict_execute_result['output_tpu'] = output_tpu_new
    else:
       assert 0, "output_isolation not support such output formula!"
    return dict_execute_result

  #lowest execute_function
  #function input:
  #1)module_native
  #2torch_dtype torch.dtype
  #3)input_sample_cpu: T or list T
  #function output: dict_execute_result dict {"output_cpu", "output_tpu"}
  def global_execute_function(self, module_native, torch_dtype, input_sample_cpu):
     #### [WARNING] You cannot set seed here because the rand input has been given
          #### [WARNING] torch.manual_seed(seed)
          #Note: 2nd detach isolation incase of leaf node, input_sample_isolation must be lowest enough to customized_execute_function
          input_sample_incase_leaf_node = self.input_sample_isolation(input_sample_cpu)
          fake_input = copy.deepcopy(input_sample_incase_leaf_node)
          input_sample_tpu= move_to(fake_input, self.device ,torch_dtype)
          #[Warning]]sometimes model will change input as model is initialized, so input_tpu must copy before net_model
          net_cpu = module_native
          net_tpu = copy.deepcopy(net_cpu)
          net_tpu = self.move_model(net_tpu, self.device ,torch_dtype)
          #tpu-first
          output_tpu, output_cpu = self.customized_execute_function(input_sample_incase_leaf_node, input_sample_tpu, net_cpu, net_tpu,torch_dtype)
          output_tpu = move_to(output_tpu, self.device_cpu,torch_dtype)
          dict_execute_result = {"output_tpu":output_tpu, "output_cpu":output_cpu}
          self.output_effectiveness_verification(dict_execute_result)
          dict_execute_result = self.output_isolation(dict_execute_result)
          return dict_execute_result

  def output_effectiveness_verification(self, dict_execute_result):
    output_cpu = dict_execute_result['output_cpu']
    output_tpu = dict_execute_result['output_tpu']
    if torch.is_tensor(output_cpu):
      assert output_cpu.device==self.device_cpu
      assert output_tpu.device==self.device_cpu
      assert torch.sum(torch.abs(output_cpu))>0, "You must ensure cpu output is non-zero Tensor"

    elif isinstance(output_cpu,tuple):
      for i  in range(self.num_multi_output):
        assert output_cpu[i].device==self.device_cpu
        assert output_tpu[i].device==self.device_cpu
        assert torch.sum(torch.abs(output_cpu[i]))>0, ("You must ensure {}-th-cpu-output is non-zero Tensor".format(i),output_cpu[i])

    else:
        assert 0,"Usually output is Tensor or Tuple of Tensors"

  # test a single input
  #function input:
  #1)sample_count int
  #module_native
  #3)input_sample_cpu: T or list T
  #function output:
  #1) bool flag: is one multioutput passed all metrics && for all dtypes?
  #2) dict: is correct for this_dtype via all metrics?
  def Torch_Test_Execution_Function_Per(self, sample_count, module_native, input_sample_cpu):
      #final_result -> dtype, output_nums, 3(vale,flag,all_flag)
      final_result = {}
      for naive_dtype in self.naive_dtype_list:
          print("Case {} Sample {} Net-Level-Dtype {} is started".format(self.case_name, sample_count, naive_dtype))
          print('\033[95m' + "[Warning]Parts of inputs might be int64/int32, then they will be kept in func <move_to>"+ '\033[0m')
          torch_dtype = self.convert_dtype_2_torch_style(naive_dtype)
          self.tested_dtype.add(torch_dtype)


          #execute step 2:start cmp metrics
          dict_execute_result = self.global_execute_function(module_native, torch_dtype, input_sample_cpu)

          #execute step 3:start cmp metrics
          final_result[naive_dtype] = self.cmp_multi_output(dict_execute_result, self.epsilon_dict[naive_dtype])
          pass_flag_all_outputs = True
          output_num = len(final_result[naive_dtype])
          for idx_output in range(output_num):
            pass_flag_all_outputs &= final_result[naive_dtype][idx_output]["passed_flag_one_sample_one_dtype"]
          if self.dump_flag and not pass_flag_all_outputs:
             self.Dumper.dump_info(sample_count,  dict_execute_result,torch_dtype)

      ###Value Cmp
      count_all_correct_dtype = 0
      dict_is_this_dtype_correct = {}
      for naive_dtype in self.naive_dtype_list:
        is_all_outputs_all_metrics_passed = False
        assert self.num_multi_output==len(final_result[naive_dtype])
        for each_metric in self.metric_table:
            for idx_output in range(self.num_multi_output):
                is_all_outputs_all_metrics_passed |=   final_result[naive_dtype][idx_output]['metric_flag'][each_metric]

        ###output info level-1, correct for each input_sample
        if (is_all_outputs_all_metrics_passed):
          count_all_correct_dtype +=1
          for idx_output in range(output_num):
            print("[Result]",self.case_name,"{}  output-{} is success".format(naive_dtype, idx_output), final_result[naive_dtype][idx_output]['metric_value'])
          dict_is_this_dtype_correct[naive_dtype] = 1
        else:
          for idx_output in range(output_num):
            print("[Result]",self.case_name, "{}  output-{} is failed".format(naive_dtype, idx_output), final_result[naive_dtype][idx_output]['metric_value'])
          dict_is_this_dtype_correct[naive_dtype] = 0
      return count_all_correct_dtype ==self.dtype_num,dict_is_this_dtype_correct


  # print test info for all inputs, which means each input sharing same seed
  def print_global_info(self):
      print(("*****************Test {} START INFO*************************").format(self.case_name))
      print("Seed:                   ", self.seed)
      print("Metrics for Compare:    ",  self.metric_table)
      print("Dtype and allowed error:", self.epsilon_dict)


  # print shape for each inputs
  def print_each_input_batch_shape(self, inputs):
    if torch.is_tensor(inputs):
       print("Single Input shape is {}".format( inputs.shape))
    elif isinstance(inputs, list):
      for idx,input_idx in enumerate(inputs):
          if torch.is_tensor(input_idx):
            print("Input {} shape is {}".format(idx, input_idx.shape))
          else:
            print("Input {} is 0-dim value".format(idx))
    elif isinstance(inputs, int) or isinstance(inputs, float):
       print("Single Input is 0-dim value")
    else:
      raise TypeError("Invalid format of input shape!")


#Note: 2nd detach isolation incase of leaf node, input_sample_isolation must be lowest enough to customized_execute_function
#input is same with output, only T is detached
#inputs->outputs
#case 1: Tensor -> Tensor.detach()
#case 2: [T, T] ->[T.detach(,T.detach())]
#case 3: [int ,flaot] , no detach
  def input_sample_isolation(self, inputs):
    if torch.is_tensor(inputs):
       return inputs.detach()
    elif isinstance(inputs, list):
      res = []
      for idx,input_idx in enumerate(inputs):
          if torch.is_tensor(input_idx):
            res +=[input_idx.detach()]
          else:
            res +=[input_idx] #ofcourse value has no detach
      return res
    elif isinstance(inputs, int) or isinstance(inputs, float):
       return inputs
    else:
      raise TypeError("Invalid format of input shape!")

  #function input -> output:
  #case 1 [[T]]->[[T]]
  #case 2 [T] -> [T]
  #case 3  T ->  [T]
  #case 3  int/float ->  int/float
  def func_input_sample_process(self, input_sample_collection):
      if isinstance(input_sample_collection, dict):
        return  list(input_sample_collection.values())
      elif isinstance(input_sample_collection, list):
        return  input_sample_collection
      elif isinstance(input_sample_collection, float) or isinstance(input_sample_collection, int):
         return  input_sample_collection
      elif torch.is_tensor(input_sample_collection):
         #same as [tensor a]
         return  [input_sample_collection]
      raise TypeError("Invalid  format of input sample only {tensor, dict,list,int,float} is supported!")


  def Torch_Test_Execution_Function_flag_allowed(self, module_native, input_sample_collection):

      input_sample_processed = self.func_input_sample_process(input_sample_collection)

      #level 3 global info: print basic seed
      self.print_global_info()

      current_correct_num = 0
      static_corret_dtype_case = dict.fromkeys(self.naive_dtype_list, 0)
      #Notice: if input is tensor_a ,then collection is [tensor_a], processed is [[tensor_a]]
      for idx, input_sample in enumerate(input_sample_processed):
          print("--------------------------------------------------------")
          print("Case {} Sample {} is started".format(self.case_name, idx))
          print("--------------------------------------------------------")

          self.print_each_input_batch_shape(input_sample)
          temp_flag,is_correct_dtype = self.Torch_Test_Execution_Function_Per(idx,  module_native, input_sample)
          current_correct_num += temp_flag == 1
          for dtype_per in self.naive_dtype_list:
              if (is_correct_dtype[dtype_per]):
                  static_corret_dtype_case[dtype_per] +=1
      #level 2 static info: print correct for all samples
      print("*****************Basic Stats CMP INFO*************************")
      print("[INFO]Tested_Dtype includes: ",self.tested_dtype ,["[INFO_END]"])
      correct_dtype_list = []
      for i in self.naive_dtype_list:
        if static_corret_dtype_case[i] == len(input_sample_processed):
              print("[CMD]Case {}: dtype {} is all corrected".format(self.case_name, i))
              correct_dtype_list+=[i]
      wrong_dtype = list(set(self.naive_dtype_list) - set(correct_dtype_list))
      if (len(wrong_dtype)>0):
          for idx, per_dtype in enumerate(wrong_dtype):
            print("[CMD]Case {}: dtype {} exist errors".format(self.case_name, per_dtype))
      print("Case {}: {}/{} Samples is completely corrected".format(self.case_name, current_correct_num, len(input_sample_processed)))
      print(("*****************Test {} Completely End*************************").format(self.case_name))

  def Torch_Test_Execution_Function(self, module_native, input_sample_collection):
    if(self.flag_is_such_arch_ready_test):
      self.Torch_Test_Execution_Function_flag_allowed(module_native, input_sample_collection)
    else:
      print("[INFO]Test skikped for this arch!")

class TensorComparator:
    def __init__(self, delta=1e-1, max_error_count=128):
        self.delta = delta
        self.max_error_count = max_error_count

    @staticmethod
    def cosine_similarity(x, y):
        numerator = torch.sum(x * y)
        sqrt_x = torch.sqrt(torch.sum(torch.pow(x, 2)))
        sqrt_y = torch.sqrt(torch.sum(torch.pow(y, 2)))
        denominator = sqrt_x * sqrt_y
        if denominator.item() == 0.0:
            if sqrt_x.item() == 0.0 and sqrt_y.item() == 0.0:
                return 1.0
            else:
                return 0.0
        return numerator / denominator
    
    @staticmethod
    def euclidean_similarity(x, y):
        ed = torch.sqrt(torch.sum(torch.pow(x - y, 2)))
        sr = torch.sqrt(torch.sum(torch.pow((x + y) / 2, 2))) + 1e-7
        if (torch.isinf(ed) or torch.isinf(sr)):
            res = 0.0
        else:
            res = 1 - ed / sr
        return res

    def compare_float(self, exp_tensor, got_tensor, only_warning):

        total = 0
        max_error_count = 128
        delta = 1e-1

        exp_tensor = np.array(exp_tensor)
        got_tensor = np.array(got_tensor)

        # Vectorized computation of absolute differences and relative differences
        abs_diff = np.abs(exp_tensor - got_tensor)
        max_abs_vals = np.maximum(np.abs(exp_tensor), np.abs(got_tensor))
        min_abs_vals = np.minimum(np.abs(exp_tensor), np.abs(got_tensor))
        with np.errstate(divide='ignore', invalid='ignore'):  # Ignore division by zero
            rel_diff = np.where(min_abs_vals < 1e-20, np.inf, abs_diff / min_abs_vals)

        # Mask for values with max absolute value > 1.0
        mask_large_values = max_abs_vals > 1.0
        # Mask for significant relative differences
        mask_rel_diff = rel_diff > delta
        # Mask for significant absolute differences
        mask_abs_diff = abs_diff > delta

        # Combine masks for warnings
        warning_mask = mask_large_values & (mask_rel_diff | (min_abs_vals < 1e-20))
        abs_warning_mask = ~mask_large_values & mask_abs_diff

        # Count warnings and print messages
        for idx in np.where(warning_mask | abs_warning_mask)[0]:
            if warning_mask[idx]:
                print(f"rel warning at index {idx} exp {exp_tensor[idx]:.20f} got {got_tensor[idx]:.20f}")
            elif abs_warning_mask[idx]:
                print(f"abs warning at index {idx} exp {exp_tensor[idx]:.20f} got {got_tensor[idx]:.20f}")
            total += 1
            if total > max_error_count and not only_warning:
                return -1, total

        return 0, total

    def cmp_result(self, tensor_target, tensor2_result):
            compare_status = self.compare_float(tensor_target.view(-1), tensor2_result.view(-1), False)
            if compare_status == -1:
                print("Error: Too many warnings detected.")
            cos_my = self.cosine_similarity(tensor_target.view(-1), tensor2_result.view(-1))
            euclidean_dist = self.euclidean_similarity(tensor_target.view(-1), tensor2_result.view(-1))
            print("Result : cosine similarity:", cos_my)
            print("Result : euclidean similarity:", euclidean_dist)
            if(cos_my < 0.9 or euclidean_dist < 0.8 or compare_status == -1):
                return False
            return True