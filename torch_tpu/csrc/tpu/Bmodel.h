#pragma once
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <torch/csrc/Stream.h>
#include <torch/csrc/python_headers.h>
#if defined BACKEND_SG2260
#include "torch_tpu/csrc/core/TPUStream.h"
#include "tpuv7_modelrt.h"
#include "torch_tpu/csrc/core/TPUDeviceManager.h"
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

PyMethodDef* THPTBmodel_get_methods();

#endif // BACKEND_SG2260