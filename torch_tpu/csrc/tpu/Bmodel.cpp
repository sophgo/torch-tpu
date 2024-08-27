#if defined BACKEND_SG2260
#include "Bmodel.h"
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/tensor/python_tensor.h>

#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include <torch/extension.h>
#include "torch_tpu/csrc/core/TPUCtypeApi.h"
#include "torch_tpu/csrc/utils/LazyInit.h"
#include "torch_tpu/csrc/aten/TPUFormatCastHelper.h"

PythonTensor::PythonTensor(tpuRtDataType_t dtype_, const char *name_, float scale_,
                                 int zero_point_, tpuRtShape_t shape) {
  name = std::string(name_);
  qscale = scale_;
  qzero_point = zero_point_;
  std::vector<size_t> s(shape.dims, shape.dims + shape.num_dims);
  fixDtype(dtype_);
}

void PythonTensor::fixDtype(tpuRtDataType_t fmt) {
    switch (fmt) {
    case TPU_FLOAT32:
      pytype = py::dtype("single");
      dtype = "f32";
      break;
    case TPU_INT8:
      pytype = py::dtype("int8");
      dtype = "i8";
      break;
    case TPU_UINT8:
      pytype = py::dtype("uint8");
      dtype = "u8";
      break;
    case TPU_INT4:
      pytype = py::dtype("int8");
      dtype = "i4";
      break;
    case TPU_UINT4:
      pytype = py::dtype("uint8");
      dtype = "u4";
      break;
    case TPU_INT16:
      pytype = py::dtype("int16");
      dtype = "i16";
      break;
    case TPU_UINT16:
      pytype = py::dtype("uint16");
      dtype = "u16";
      break;
    case TPU_INT32:
      pytype = py::dtype("int32");
      dtype = "i32";
      break;
    case TPU_UINT32:
      pytype = py::dtype("uint32");
      dtype = "u32";
      break;
    case TPU_BFLOAT16:
      // numpy has no bf16 type, use uint16 instread of bf16.
      pytype = py::dtype("uint16");
      dtype = "bf16";
      break;
    case TPU_FLOAT16:
      pytype = py::dtype("float16");
      dtype = "f16";
      break;
    default:
      printf("error, tpuRtDataType_t : %d\n", fmt);
      assert(0);
    }
  }

uint64_t shape_count(tpuRtShape_t shape) {
    uint64_t count = 1;
    for (int i = 0; i < shape.num_dims; i++) {
      count *= shape.dims[i];
    }
    return count;
}

uint64_t PythonNet::shapeCount(tpuRtShape_t shape) {
  return shape_count(shape);
}

int data_type_size(tpuRtDataType_t dtype) {
    switch (dtype) {
      case TPU_FLOAT32:
      case TPU_INT32:
      case TPU_UINT32:
        return 4;
      case TPU_FLOAT16:
      case TPU_BFLOAT16:
      case TPU_INT16:
        return 2;
      case TPU_INT8:
      case TPU_UINT8:
        return 1;
      case TPU_INT4:
      case TPU_UINT4:
        return 1;  // need modify ?  TODO
      default:
        return 4;
    }
}

int PythonNet::dataTypeSize(tpuRtDataType_t dtype) {
    return data_type_size(dtype);
}

static void ReportAndDelete(void *ptr)
{
  // do nothing
  #ifdef DEBUG
  // std::cout << "ReportAndDelete for prt : " << ptr  << std::endl;
  // std::cout << "PLEASE BE CAREFUL, THIS FUNCTION SHOULD NOT CALLED" << std::endl;
  #endif
}

at::DataPtr make_tensor_ptr(void *ptr)
{
  return {ptr, ptr, &ReportAndDelete, tpu::TPUGetCurrentDevice()};
}

// dtype convert into torch dtype

auto convert_dtype_to_torch_dtype(tpuRtDataType_t dtype)
{
  switch (dtype) {
    case TPU_FLOAT32:
      return at::kFloat;
    case TPU_INT32:
      return at::kInt;
    case TPU_UINT32:
      return at::kInt;
    case TPU_FLOAT16:
      return at::kHalf;
    case TPU_BFLOAT16:
      return at::kBFloat16;
    case TPU_INT16:
      return at::kShort;
    case TPU_UINT16:
      return at::kShort;
    case TPU_INT8:
      return at::kChar;
    case TPU_UINT8:
      return at::kByte;
    case TPU_INT4:
      return at::kChar;
    case TPU_UINT4:
      return at::kByte;
    default:
      return at::kFloat;
  }
}

auto make_tensor_from_ptr(void *ptr, tpuRtShape_t shape, tpuRtDataType_t dtype)
{
  auto size = shape_count(shape) * data_type_size(dtype);
  auto meta_dtype = convert_dtype_to_torch_dtype(dtype);
  auto tensor_dtype = at::scalarTypeToTypeMeta(meta_dtype);
  at::detail::check_size_nonnegative ( size );
  #ifdef DEBUG
  std::cout << "ptr : " << ptr << "  size : " << size << std::endl;
  #endif
  auto ptr_ = make_tensor_ptr(ptr);
  auto allocator = c10::GetTPUAllocator();
  c10::intrusive_ptr<c10::StorageImpl> storage_impl = c10::make_intrusive<torch_tpu::TPUStorageImpl> (
                      c10::StorageImpl::use_byte_size_t(),
                      size,
                      std::move(ptr_),
                      allocator,
                      /*resizeable=*/false );
  auto tensor = at::detail::make_tensor<torch_tpu::TPUTensorImpl> ( storage_impl, tensor_dtype );
  std::vector <int64_t> sizes;
  for (int i = 0; i < shape.num_dims; i++) {
    sizes.push_back(shape.dims[i]);
  }
  tensor.unsafeGetTensorImpl()->set_sizes_contiguous ( sizes );
  return tensor;
}

PythonNet::PythonNet(tpuRtNet_t *net, const char *netname, int stage){
    m_net = *net;
    name = std::string(netname);
    m_info = tpuRtGetNetInfo(m_net, netname);
    num_input = m_info.input.num;
    num_output = m_info.output.num;
    for (int i = 0; i < num_input; i++) {
      auto &shape = m_info.stages[stage].input_shapes[i];
    //   auto &dtype = m_info.input.dtypes[i];
      input_shapes.push_back(shape);
    }
    input_mems = m_info.stages[stage].input_mems;
    for (int i = 0; i < num_output; i++) {
      auto &shape = m_info.stages[stage].output_shapes[i];
    //   auto &dtype = m_info.output.dtypes[i];
      output_shapes.push_back(shape);
    }
    output_mems = m_info.stages[stage].output_mems;
}

void PythonNet::forward(std::vector<py::object>& inputs,
               std::vector<py::object>& outputs){
    // py::object more comfortable to use than torch::tensor
    int input_size = (int) inputs.size();
    // assert(input_size == num_input && "input size not match");
    if (input_size != num_input) {
      printf("input size not match. cur input size %d, hope to %d\n", input_size, num_input);
      fflush(stdout);
      exit(1);
    }
    int output_size = (int) outputs.size();
    if (output_size != num_output) {
      printf("output size not match. cur output size %d, hope to %d\n", output_size, num_output);
      fflush(stdout);
      exit(1);
    }
    // assert(output_size == num_output && "output size not match");

    std::vector<tpuRtTensor_t> input_tensors;
    std::vector<tpuRtTensor_t> output_tensors;
    // more check can be added here
    for (int i = 0; i < num_input; i++) {
      auto dtype    = m_info.input.dtypes[i];
      auto shape    = m_info.stages[0].input_shapes[i];// only support one stage, todo later
      auto data_ptr = inputs[i].attr("data_ptr")().cast<uintptr_t>();
      tpuRtTensor_t input_tensor;
      input_tensor.shape = shape;
      input_tensor.dtype = dtype;
      input_tensor.data = (void*) data_ptr;
      input_tensors.push_back(input_tensor);
    }
    for (int i = 0; i < num_output; i++) {
      auto dtype    = m_info.output.dtypes[i];
      auto shape    = m_info.stages[0].output_shapes[i];// only support one stage, todo later
      auto data_ptr = outputs[i].attr("data_ptr")().cast<uintptr_t>();
      tpuRtTensor_t output_tensor;
      output_tensor.shape = shape;
      output_tensor.dtype = dtype;
      output_tensor.data = (void*) data_ptr;
      output_tensors.push_back(output_tensor);
    }
    auto stream = c10_tpu::getCurrentTPUStream();
    m_stream = stream;
    auto status = tpuRtLaunchNetAsync(m_net, input_tensors.data(), output_tensors.data(), name.c_str(), stream);
    if (status != tpuRtSuccess) {
      printf("launch net failed\n");
    }
}

void PythonNet::forward_sync(std::vector<py::object>& inputs,
               std::vector<py::object>& outputs){
    forward(inputs, outputs);
    tpuRtStreamSynchronize(m_stream);
}

void PythonNet::printNetworkInfo(tpuRtNetInfo_t *info){
    printf("++++++++++++++ net info ++++++++++++++\n");
    printf("net name: %s\n", info->name);
    printf("is dynamic:%d\n", info->is_dynamic);
    printf("input num:%d\n", info->input.num);
    for (int i = 0; i < info->input.num; i++) {
      printf("input:[%s], type:[%d], scale:[%f], zero_point:[%d]\n",
             info->input.names[i], info->input.dtypes[i], info->input.scales[i],
             info->input.zero_points[i]);
    }
    printf("output num:%d\n", info->output.num);
    for (int i = 0; i < info->output.num; i++) {
      printf("output:[%s], type:[%d], scale:[%f], zero_point:[%d]\n",
             info->output.names[i], info->output.dtypes[i],
             info->output.scales[i], info->output.zero_points[i]);
    }

    printf("stage num:%d\n", info->stage_num);
    for (int i = 0; i < info->stage_num; i++) {
      printf("-----------------stage[%d]-------------------\n", i);
      for (int j = 0; j < info->input.num; j++) {
        printf("input[%s], shape:[ ", info->input.names[j]);
        for (int k = 0; k < info->stages[i].input_shapes[j].num_dims; k++) {
          printf("%d ", info->stages[i].input_shapes[j].dims[k]);
        }
        printf("]\n");
      }
      for (int j = 0; j < info->output.num; j++) {
        printf("output[%s], shape:[ ", info->output.names[j]);
        for (int k = 0; k < info->stages[i].output_shapes[j].num_dims; k++) {
          printf("%d ", info->stages[i].output_shapes[j].dims[k]);
        }
        printf("]\n");
      }
    }
    printf("================ net info ===============\n");
}

PythonModel::PythonModel(const std::string &model_file, int dev_id,
                                const std::string &decrypt_lib) {
    // tpuRtStatus_t ret = tpuRtSuccess;
    m_dev_id = dev_id;
    // if(tpu::IsTPUMgrInited() != tpu::INIT_SUCCESS){
    //   printf(">>>>>>>  tpuMgr not inited, init it\n");
    //   tpu::InitTPUMgr();
    // }
    tpuRtSetDevice(dev_id);
    tpuRtCreateNetContext(&m_context);
    tpuRtLoadNet(model_file.c_str(), m_context, &m_net);
    char **net_names = NULL;
    m_net_num = tpuRtGetNetNames(m_net, &net_names);
    for (int i = 0; i < m_net_num; i++) {
      networks.push_back(net_names[i]);
    }
    tpuRtFreeNetNames(net_names);
}

void PythonModel_deleter(PyObject* capsule) {
  printf("PythonModel_deleter\n");
  auto model = static_cast<PythonModel*>(PyCapsule_GetPointer(capsule, "PythonModel"));
  delete model; // free
}

void PythonNet_deleter(PyObject* capsule) {
  printf("PythonNet_deleter\n");
  auto net = static_cast<PythonNet*>(PyCapsule_GetPointer(capsule, "PythonNet"));
  delete net; // free
}

PyObject* BmodelTensor_wrap(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* net_capsule;
  int input_idx;
  int is_input;
  if (!PyArg_ParseTuple(args, "Oii", &net_capsule, &input_idx, &is_input)) {
    return NULL;
  }
  // check net_capsule
  if (!PyCapsule_IsValid(net_capsule, "PythonNet")) {
    PyErr_SetString(PyExc_RuntimeError, "Invalid PythonNet capsule");
    return NULL;
  }
  auto net = (PythonNet*) PyCapsule_GetPointer(net_capsule, "PythonNet");
  at::Tensor tensor;
  if(is_input == 0)
  {
    tensor = make_tensor_from_ptr(net->output_mems[input_idx], net->output_shapes[input_idx], net->m_info.output.dtypes[input_idx]);
  }else{
    tensor = make_tensor_from_ptr(net->input_mems[input_idx], net->input_shapes[input_idx], net->m_info.input.dtypes[input_idx]);
  }
  PyObject* tensor_py = THPVariable_Wrap(tensor);
  if (!tensor_py) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to wrap tensor as Python object.");
    return NULL;
  }
  return tensor_py;
  END_HANDLE_TH_ERRORS
}


PyObject* THPTPythonModel_wrap(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  const char* model_file;
  int dev_id;
  const char* decrypt_lib;
  if (!PyArg_ParseTuple(args, "sis", &model_file, &dev_id, &decrypt_lib)) {
    return NULL;
  }
  auto model = new PythonModel(model_file, dev_id, decrypt_lib);
  return PyCapsule_New(model, "PythonModel", PythonModel_deleter);
  END_HANDLE_TH_ERRORS
}

PyObject* THPTPythonModel_info(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* dict = PyDict_New();
  PyObject* model_capsule;
  if (!PyArg_ParseTuple(args, "O", &model_capsule)) {
    return NULL;
  }
  auto model = (PythonModel*) PyCapsule_GetPointer(model_capsule, "PythonModel");
  auto networks = PyList_New(model->networks.size());
  for (size_t i = 0; i < model->networks.size(); i++) {
    PyList_SetItem(networks, i, PyUnicode_FromString(model->networks[i]));
  }
  PyDict_SetItemString(dict, "networks", networks);
  PyDict_SetItemString(dict, "net_num", PyLong_FromLong(model->m_net_num));
  return dict;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTPythonModel_Net(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* model_capsule;
  const char* net_name;
  if (!PyArg_ParseTuple(args, "Os", &model_capsule, &net_name)) {
    return NULL;
  }
  auto model = (PythonModel*) PyCapsule_GetPointer(model_capsule, "PythonModel");
  auto net = new PythonNet(&model->m_net, net_name);
  return PyCapsule_New(net, "PythonNet", PythonNet_deleter);
  END_HANDLE_TH_ERRORS
}

PyObject* THPTPythonNet_forward(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* net_capsule;
  PyObject* inputs;
  PyObject* outputs;
  if (!PyArg_ParseTuple(args, "OOO", &net_capsule, &inputs, &outputs)) {
    return NULL;
  }
  // inputs and outputs are list of torch tensor
  // printf(">>>>>>>>>>>>>> input size %d, output size %d\n", (int)PyList_Size(inputs), (int)PyList_Size(outputs));
  auto net = (PythonNet*) PyCapsule_GetPointer(net_capsule, "PythonNet");
  std::vector<py::object> input_tensors;
  std::vector<py::object> output_tensors;
    for (int i = 0; i < PyList_Size(inputs); i++) {
        PyObject* item = PyList_GetItem(inputs, i);
        input_tensors.push_back(py::reinterpret_borrow<py::object>(item));  // 转换并添加
    }
    for (int i = 0; i < PyList_Size(outputs); i++) {
        PyObject* item = PyList_GetItem(outputs, i);
        output_tensors.push_back(py::reinterpret_borrow<py::object>(item));  // 同样处理输出
    }
  // printf("\n\n\n\n>>>>>>>>>>>>>> input size %d, output size %d\n", (int)input_tensors.size(), (int)output_tensors.size());
  net->forward(input_tensors, output_tensors);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTPythonNet_forward_sync(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* net_capsule;
  PyObject* inputs;
  PyObject* outputs;
  if (!PyArg_ParseTuple(args, "OOO", &net_capsule, &inputs, &outputs)) {
    return NULL;
  }
  auto net = (PythonNet*) PyCapsule_GetPointer(net_capsule, "PythonNet");
  std::vector<py::object> input_tensors;
  std::vector<py::object> output_tensors;
    for (int i = 0; i < PyList_Size(inputs); i++) {
        PyObject* item = PyList_GetItem(inputs, i);
        input_tensors.push_back(py::reinterpret_borrow<py::object>(item));  // 转换并添加
    }
    for (int i = 0; i < PyList_Size(outputs); i++) {
        PyObject* item = PyList_GetItem(outputs, i);
        output_tensors.push_back(py::reinterpret_borrow<py::object>(item));  // 同样处理输出
    }
  net->forward_sync(input_tensors, output_tensors);
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTPythonNet_dump(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* net_capsule;
  if (!PyArg_ParseTuple(args, "O", &net_capsule)) {
    return NULL;
  }
  auto net = (PythonNet*) PyCapsule_GetPointer(net_capsule, "PythonNet");
  net->dump();
  Py_RETURN_NONE;
  END_HANDLE_TH_ERRORS
}

PyObject* THPTPythonNet_getNetworkInfo(PyObject* self, PyObject* args) {
  HANDLE_TH_ERRORS
  PyObject* dict = PyDict_New();
  PyObject* net_capsule;
  if (!PyArg_ParseTuple(args, "O", &net_capsule)) {
    return NULL;
  }
  auto net = (PythonNet*) PyCapsule_GetPointer(net_capsule, "PythonNet");
  auto info = net->m_info;
    PyDict_SetItemString(dict, "name", PyUnicode_FromString(info.name));
    // is_dynamic
    PyDict_SetItemString(dict, "is_dynamic", PyLong_FromLong(info.is_dynamic));
    PyDict_SetItemString(dict, "stage_num",  PyLong_FromLong(info.stage_num));
    PyDict_SetItemString(dict, "num_input",  PyLong_FromLong(info.input.num));
    PyDict_SetItemString(dict, "num_output", PyLong_FromLong(info.output.num));
    PyObject* inputs = PyList_New(info.input.num);
    for (int i = 0; i < info.input.num; i++) {
      PyObject* input = PyDict_New();
      PyDict_SetItemString(input, "name",  PyUnicode_FromString(info.input.names[i]));
      PyDict_SetItemString(input, "dtype", PyLong_FromLong(info.input.dtypes[i]));
      auto num_dims = info.stages[0].input_shapes[i].num_dims;
      PyDict_SetItemString(input, "shape", PyList_New(num_dims));
      for (int j = 0; j < num_dims; j++) {
        PyList_SetItem(PyDict_GetItemString(input, "shape"), j, PyLong_FromLong(info.stages[0].input_shapes[i].dims[j]));
      }
      PyList_SetItem(inputs, i, input);
      // address
      PyDict_SetItemString(input, "address", PyLong_FromLong((long)net->input_mems[i]));
    }
    PyDict_SetItemString(dict, "inputs", inputs);

    PyObject* outputs = PyList_New(info.output.num);
    for (int i = 0; i < info.output.num; i++) {
      PyObject* output = PyDict_New();
      PyDict_SetItemString(output, "name",  PyUnicode_FromString(info.output.names[i]));
      PyDict_SetItemString(output, "dtype", PyLong_FromLong(info.output.dtypes[i]));
      auto num_dims = info.stages[0].output_shapes[i].num_dims;
      PyDict_SetItemString(output, "shape", PyList_New(num_dims));
      for (int j = 0; j < num_dims; j++) {
        PyList_SetItem(PyDict_GetItemString(output, "shape"), j, PyLong_FromLong(info.stages[0].output_shapes[i].dims[j]));
      }
      PyList_SetItem(outputs, i, output);
      // address
      PyDict_SetItemString(output, "address", PyLong_FromLong((long)net->output_mems[i]));
    }
    PyDict_SetItemString(dict, "outputs", outputs);
    return dict;
  END_HANDLE_TH_ERRORS
}

static PyMethodDef THPTPythonBModel_methods[] = {
    // model
    {"Model" , (PyCFunction)THPTPythonModel_wrap, METH_VARARGS, nullptr},
    {"Net"   , (PyCFunction)THPTPythonModel_Net,  METH_VARARGS, nullptr},
    // net
    {"forward"       , (PyCFunction)THPTPythonNet_forward,        METH_VARARGS, nullptr},
    {"forward_sync"  , (PyCFunction)THPTPythonNet_forward_sync,   METH_VARARGS, nullptr},
    {"dump"          , (PyCFunction)THPTPythonNet_dump,           METH_VARARGS, nullptr},
    {"getNetworkInfo", (PyCFunction)THPTPythonNet_getNetworkInfo, METH_VARARGS, nullptr},
    {"getModelInfo"  , (PyCFunction)THPTPythonModel_info,         METH_VARARGS, nullptr},
    {"getTensor"     , (PyCFunction)BmodelTensor_wrap,            METH_VARARGS, nullptr},
    // TODO: add a more general way to get tensor ( such as with address, size, and dtype )
    // nullptr
    {nullptr}
};

PyMethodDef* THPTBmodel_get_methods(){
    return THPTPythonBModel_methods;
}

#endif // BACKEND_SG2260