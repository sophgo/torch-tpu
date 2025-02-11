#pragma once
#include <Python.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/python_headers.h>
#include "torch_tpu/csrc/core/TPUStream.h"


#if defined BACKEND_SG2260
#include "tpuv7_modelrt.h"
namespace py = pybind11;

struct PythonTensor {
  PythonTensor(tpuRtDataType_t dtype_, const char *name_, float scale_,
               int zero_point_, tpuRtShape_t shape);
  std::string name;
  std::string dtype; // f32/f16/bf16/i8/i16/i32/u8/u16/u32
  float qscale;
  int qzero_point;

private:
  py::dtype pytype;
  void fixDtype(tpuRtDataType_t fmt);
};

struct PythonNet {
  PythonNet(tpuRtNet_t *net, const char *netname, int stage = 0);

  uint64_t shapeCount(tpuRtShape_t shape);

  int dataTypeSize(tpuRtDataType_t dtype);

  void printNetworkInfo(tpuRtNetInfo_t *info);

  void dump() { printNetworkInfo(&m_info); }

  // input and output vector is torch tensors
  void forward(std::vector<py::object>& inputs,
               std::vector<py::object>& outputs);

  void forward_sync(std::vector<py::object>& inputs,
               std::vector<py::object>& outputs);

  std::string name;
  int num_input;
  int num_output;
  tpuRtNet_t m_net;
  tpuRtNetInfo_t m_info;
  void **input_mems;
  void **output_mems;
  std::vector<tpuRtShape_t> input_shapes;
  std::vector<tpuRtShape_t> output_shapes;
private:
  PythonNet() {}
  tpuRtStream_t m_stream;
};

struct PythonModel {
  PythonModel(const std::string &model_file, int dev_id,
	      const std::string &decrypt_lib);

  ~PythonModel() {
    tpuRtUnloadNet(m_net);
    tpuRtDestroyNetContext(m_context);
    //tpuRtFreeDevice(m_dev_id);
  }

  std::vector<const char *> networks;
  int m_net_num;
  tpuRtNet_t m_net;

private:
  PythonModel() {}
  uint32_t chip_id;
  tpuRtNetContext_t m_context;
  int m_dev_id = 0;
};

#elif defined BACKEND_1684X
#include "bmruntime_interface.h"
namespace py = pybind11;
struct PythonTensor {
  PythonTensor(bm_data_type_t dtype_, const char *name_, float scale_,
               int zero_point_, bm_shape_t shape);

  std::string name;
  std::string dtype; // f32/f16/bf16/i8/i16/i32/u8/u16/u32
  float qscale;
  int qzero_point;
  void fixDtype(bm_data_type_t fmt);
};

struct PythonNet{
  PythonNet(void* bmrt_, const char* netname, bm_handle_t handle_, int stage = 0);
  void dump() { bmrt_print_network_info(m_info); }
  void forward(std::vector<py::object> &inputs, std::vector<py::object> &outputs);
  // void forward_dynamic(std::vector<> &inputs, std::vector<> &outputs);
  void forward_sync(std::vector<py::object> &inputs, std::vector<py::object> &outputs);
  std::string name;
  int num_input;
  int num_output;
  void* p_bmrt;
  bm_handle_t bm_handle;
  const bm_net_info_t* m_info;
  std::vector<bm_shape_t> input_shapes;
  std::vector<bm_shape_t> output_shapes;
  bm_device_mem_t *input_mems;
  bm_device_mem_t *output_mems;
};

struct PythonModel{
  PythonModel(const std::string &model_file, int dev_id, const std::string &decrypt_lib);
  ~PythonModel(){
    bmrt_destroy(p_bmrt);
  }

  std::vector<const char *> networks;
  int m_net_num;
  void* p_bmrt;
  bm_handle_t bm_handle;

  int m_dev_id = 0;
};

#else
//
#endif // BACKEND_SG2260

PyMethodDef* THPTBmodel_get_methods();