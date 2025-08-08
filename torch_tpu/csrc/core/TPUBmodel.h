#ifdef NO_PYTHON_API
#pragma once

namespace tpu{

namespace bmodel{

#if defined BACKEND_SG2260
#include "tpuv7_modelrt.h"
typedef tpuRtDataType_t        tpu_dtype_t;
typedef tpuRtNetInfo_t         tpu_net_info_t;
typedef tpuRtStream_t          tpu_stream_t;
typedef tpuRtTensor_t          tpu_tensor_t;
typedef tpuRtShape_t           tpu_shape_t;
#elif defined BACKEND_1684X or defined BACKEND_1686
#include "bmruntime_interface.h"
typedef bm_data_type_t         tpu_dtype_t;
typedef const bm_net_info_t*   tpu_net_info_t;
typedef bm_handle_t            tpu_stream_t;
typedef bm_tensor_t            tpu_tensor_t;
typedef bm_shape_t             tpu_shape_t;
#endif

typedef unsigned long long u64;

typedef void*  tpu_context_t;
typedef void*  tpu_net_t;

class TPUBmodel{

public:
    //
    TPUBmodel(const std::string cur_model_file, int dev_id, const std::string cur_decrypt_lib);
    void getTensor(int idx, bool is_input);
    void handle_each_net();
    bool check_each_net();
    void forward();
    void forward_sync();
    void forward(std::vector<at::Tensor> input_tensors, std::vector<at::Tensor> output_tensors);
    // void forward(std::vector<at::Tensor> &inputs, std::vector<at::Tensor> &outputs);
    void forward_sync(std::vector<at::Tensor> input_tensors, std::vector<at::Tensor> output_tensors);
    ~TPUBmodel();

    std::string                   model_file;
    int                           dev_id;
    std::string                   decrypt_lib;
    int                           cur_net_id;
    int                           m_net_num;
    std::vector<int>              input_num;
    std::vector<int>              output_num;
    std::vector<std::string>      networks;
    std::vector< std::vector<const char*> > input_names;
    std::vector< std::vector<const char*> > output_names;
    std::vector< std::vector<u64> > input_mems;
    std::vector< std::vector<u64> > output_mems;
    std::vector< std::vector<at::Tensor> >  input_tensors;
    std::vector< std::vector<at::Tensor> >  output_tensors;
    std::vector< std::vector<tpu_dtype_t> > input_dtypes;
    std::vector< std::vector<tpu_dtype_t> > output_dtypes;
    std::vector< tpu_net_info_t >   net_infos;
    // net_id; idx; shape 3 dimension
    std::vector< std::vector< std::vector<int> > > input_shapes;
    std::vector< std::vector< std::vector<int> > > output_shapes;
private:
    tpu_stream_t            m_stream;// bm_handle
    tpu_context_t           m_context;
    tpu_net_t               m_net;// p_bmrt

};

} // namespace bmodel

} // namespace tpu
#endif