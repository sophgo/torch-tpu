#include "torch_tpu/csrc/core/TPUGuard.h"
#include "torch_tpu/csrc/core/TPUStream.h"
#include "torch_tpu/csrc/core/TPUTorchUtils.h"
#include "torch_tpu/csrc/aten/TPUNativeFunctions.h"
#include "torch_tpu/csrc/aten/TPUBmodelRuntime.h"


namespace at_tpu{
namespace modelrt{

#if defined(BACKEND_SG2260) || defined(BACKEND_SG2260E)
#include <tpuv7_modelrt.h>
void CHECKDTYPE(tpuRtDataType_t model_iodtype, at::ScalarType tensor_dtype )
{
    if (model_iodtype == TPU_FLOAT32) {
        TORCH_CHECK(tensor_dtype ==  c10::ScalarType::Float,
            "[ERROR] detect model-io is f32, tensor is ", tensor_dtype );
    } else if (model_iodtype == TPU_FLOAT16) {
        TORCH_CHECK(tensor_dtype ==  c10::ScalarType::Half,
            "[ERROR] detect model-io is f16, tensor is ", tensor_dtype );
    } else if (model_iodtype == TPU_INT8) {
        TORCH_CHECK(tensor_dtype ==  c10::ScalarType::Char,
            "[ERROR] detect model-io is char, tensor is ", tensor_dtype );
    } else if (model_iodtype == TPU_UINT8) {
        TORCH_CHECK(tensor_dtype ==  c10::ScalarType::Byte,
            "[ERROR] detect model-io is u8, tensor is ", tensor_dtype );
    } else if (model_iodtype == TPU_INT16) {
        TORCH_CHECK(tensor_dtype ==  c10::ScalarType::Short,
            "[ERROR] detect model-io is f32, tensor is ", tensor_dtype );
    } else if (model_iodtype == TPU_UINT16) {
        TORCH_CHECK(tensor_dtype ==  c10::ScalarType::Bits16,
            "[ERROR] detect model-io is f16, tensor is ", tensor_dtype );
    } else if (model_iodtype == TPU_INT32) {
        TORCH_CHECK(tensor_dtype ==  c10::ScalarType::Int,
            "[ERROR] detect model-io is i32, tensor is ", tensor_dtype );
    } else if (model_iodtype == TPU_UINT32) {
        TORCH_CHECK(tensor_dtype ==  c10::ScalarType::Int,
            "[ERROR] detect model-io is u32, tensor is ", tensor_dtype );
    } else if (model_iodtype == TPU_BFLOAT16) {
        TORCH_CHECK(tensor_dtype ==  c10::ScalarType::BFloat16,
            "[ERROR] detect model-io is bf16, tensor is ", tensor_dtype );
    } else if (model_iodtype == TPU_INT4) {
        TORCH_CHECK(tensor_dtype ==  c10::ScalarType::Bits4x2,
            "[ERROR] detect model-io is i4, tensor is ", tensor_dtype );
    } else if (model_iodtype == TPU_UINT4) {
        TORCH_CHECK(tensor_dtype ==  c10::ScalarType::Bits4x2,
            "[ERROR] detect model-io is u4, tensor is ", tensor_dtype );
    } else {
        TORCH_CHECK(false, "should not arrive here, seems like a error bmodel.")
    }
}

at::ScalarType tpuRTDtype2TorchDtype(tpuRtDataType_t fmt)
{
    at::ScalarType dtype;
    switch (fmt) {
    case TPU_FLOAT32:   dtype = c10::ScalarType::Float; break;
    case TPU_INT8:      dtype = c10::ScalarType::Char; break;
    case TPU_UINT8:     dtype = c10::ScalarType::Byte; break;
    case TPU_INT32:     dtype = c10::ScalarType::Int; break;
    case TPU_BFLOAT16:  dtype = c10::ScalarType::BFloat16; break;
    case TPU_FLOAT16:   dtype = c10::ScalarType::Half; break;
    default:
        TORCH_CHECK( false, "error, tpuRtDataType_t : %d\n", fmt);
    }
    return dtype;
}

struct BModelRunnerImpl
{
    BModelRunnerImpl(const std::string &model_file)
    {
        tpuRtCreateNetContext(&m_context);
        tpuRtLoadNet(model_file.c_str(), m_context, &m_net);
        char **net_names = NULL;
        m_net_num = tpuRtGetNetNames(m_net, &net_names);
        for (int i = 0; i < m_net_num; i++) {
            m_network_names.push_back(net_names[i]);
        }
        tpuRtFreeNetNames(net_names);
        set_running_net(m_network_names[0]);
    }
    ~BModelRunnerImpl()
    {
        tpuRtUnloadNet(m_net);
        tpuRtDestroyNetContext(m_context);
    }

    void set_running_net(const char* net_name)
    {
        auto it = std::find(m_network_names.begin(), m_network_names.end(), net_name);
        if (it == m_network_names.end()) {
            TORCH_CHECK( false, "[ERROR] not found %s in bmodels.",net_name);
        }
        m_running_net_info = tpuRtGetNetInfo(m_net, net_name);
        TORCH_CHECK(m_running_net_info.stage_num == 1, "current only support stage_num == 1");
        TORCH_CHECK(m_running_net_info.is_dynamic == false, "current not support dynamic model");
    }

    void forward(at::TensorList input, at::TensorList output,  bool non_blocking)
    {
        std::vector<tpuRtTensor_t> rt_inputs, rt_outputs;
        int num_input  = m_running_net_info.input.num;
        int num_output = m_running_net_info.output.num;
        TORCH_CHECK( (size_t)num_input == input.size() ,   "bmodel's num of input is not same with input");
        TORCH_CHECK( (size_t)num_output == output.size() , "bmodel's num of output is not same with output");

        for (int i = 0; i < num_input; i++){
            tpuRtTensor_t t;
            t.dtype = m_running_net_info.input.dtypes[i];
            t.shape = m_running_net_info.stages[0].input_shapes[i];
            t.data  = (void*)GetAddrByUnifiedAddr( (unsigned long long)input[i].data_ptr() );
            CHECKDTYPE(t.dtype, input[i].scalar_type());
            rt_inputs.push_back(t);
        }
        for (int i = 0; i < num_output; i++)
        {
            tpuRtTensor_t t;
            t.dtype = m_running_net_info.output.dtypes[i];
            t.shape = m_running_net_info.stages[0].output_shapes[i];
            t.data  = (void*)GetAddrByUnifiedAddr( (unsigned long long)output[i].data_ptr() );
            CHECKDTYPE(t.dtype, output[i].scalar_type());
            rt_outputs.push_back(t);
        }

        auto stream = (tpuRtStream_t)c10_tpu::getCurrentTPUStream().stream();
        if ( non_blocking ) {
            auto status = tpuRtLaunchNetAsync(m_net, rt_inputs.data(), rt_outputs.data(), m_running_net_info.name, stream);
            TORCH_CHECK(tpuRtSuccess == status, "tpuRtLaunchNetAsync ERROR!!!");
        } else {
            auto status = tpuRtLaunchNet(m_net, rt_inputs.data(), rt_outputs.data(), m_running_net_info.name, stream);
            TORCH_CHECK(tpuRtSuccess == status, "tpuRtLaunchNet ERROR!!!");
        }
    }

    std::tuple<BModelRunner::tensor_vec, BModelRunner::tensor_vec> get_inplace_io() {
        int num_input  = m_running_net_info.input.num;
        int num_output = m_running_net_info.output.num;

        BModelRunner::tensor_vec inputs, outputs;
        for (int i = 0; i < num_input; i++){
            std::vector<int64_t> sizes;
            tpuRtShape_t rt_shape = m_running_net_info.stages[0].input_shapes[i];
            void* data_ptr = m_running_net_info.stages[0].input_mems[i];
            auto scalar_dtype = tpuRTDtype2TorchDtype(m_running_net_info.input.dtypes[i]);
            for (int s = 0; s < rt_shape.num_dims; s++) { sizes.push_back(rt_shape.dims[s]); }
            if ( sizes.size() == 1 && sizes[0] == 1 ) sizes = {};

            auto tensor = at_tpu::TPUNativeFunctions::make_tensor_from_ptr(
                                (void*)UnifiedAddr((uint64_t)data_ptr, m_dev_id), sizes, scalar_dtype);
            inputs.push_back(tensor);
        }
        for (int i = 0; i < num_output; i++){
            std::vector<int64_t> sizes;
            tpuRtShape_t rt_shape = m_running_net_info.stages[0].output_shapes[i];
            void* data_ptr = m_running_net_info.stages[0].output_mems[i];
            auto scalar_dtype = tpuRTDtype2TorchDtype(m_running_net_info.output.dtypes[i]);
            for (int s = 0; s < rt_shape.num_dims; s++) { sizes.push_back(rt_shape.dims[s]); }
            if ( sizes.size() == 1 && sizes[0] == 1 ) sizes = {};

            auto tensor = at_tpu::TPUNativeFunctions::make_tensor_from_ptr(
                                (void*)UnifiedAddr((uint64_t)data_ptr, m_dev_id), sizes, scalar_dtype);
            outputs.push_back(tensor);
        }
        return {inputs, outputs};
    }

    std::tuple<BModelRunner::name_vec, BModelRunner::name_vec> get_io_names() {
        int num_input  = m_running_net_info.input.num;
        int num_output = m_running_net_info.output.num;
        BModelRunner::name_vec inputs, outputs;
        for (int i = 0; i < num_input; i++) {
            inputs.push_back(std::string(m_running_net_info.input.names[i]));
        }
        for (int i = 0; i < num_output; i++) {
            outputs.push_back(std::string(m_running_net_info.output.names[i]));
        }
        return {inputs, outputs};
    }

    tpuRtNet_t                  m_net;
    tpuRtNetContext_t           m_context;
    tpuRtNetInfo_t              m_running_net_info;
    int                         m_dev_id = 0;
    std::vector<const char *>   m_network_names;
    int                         m_net_num;
    const char*                 m_running_net_name;
};
#elif defined BACKEND_1684X or defined BACKEND_1686
struct BModelRunnerImpl
{
    BModelRunnerImpl(const std::string &model_file)
    {
        TORCH_CHECK( false);
    }
    ~BModelRunnerImpl()
    {
    }

    void set_running_net(const char* net_name)
    {
        TORCH_CHECK( false);
    }

    void forward(at::TensorList input, at::TensorList output,  bool non_blocking)
    {
        TORCH_CHECK( false);
    }

    std::tuple<BModelRunner::tensor_vec, BModelRunner::tensor_vec> get_inplace_io() {
        TORCH_CHECK( false);
    }

    std::tuple<BModelRunner::name_vec, BModelRunner::name_vec> get_io_names() {
        TORCH_CHECK( false);
    }

};
#endif

BModelRunner::BModelRunner( const std::string &model_file, int dev_id,
                            const std::string &decrypt_lib ) : m_dev_id(dev_id) {
    c10_tpu::TPUGuard guard(dev_id);
    m_pimpl = new BModelRunnerImpl(model_file);
}

BModelRunner::~BModelRunner() {
    delete m_pimpl;
    m_pimpl = nullptr;
}

void BModelRunner::set_running_net(const char* net_name) {
    m_pimpl->set_running_net(net_name);
    return;
}

std::tuple<BModelRunner::name_vec, BModelRunner::name_vec> BModelRunner::GetIONames() {
    return  m_pimpl->get_io_names();
}

std::tuple<BModelRunner::tensor_vec, BModelRunner::tensor_vec> BModelRunner::GenInplaceTensor() {
    return  m_pimpl->get_inplace_io();
}

void BModelRunner::forward(at::TensorList input, at::TensorList output,  bool non_blocking)
{
    return m_pimpl->forward(input, output, non_blocking);
}
void BModelRunner::forward(at::Tensor& input, at::Tensor& output, bool non_blocking) {
    at::TensorList input_list({ input }), output_list({ output });
    forward(input_list, output_list, non_blocking);
}
void BModelRunner::forward(at::Tensor& input, at::TensorList output, bool non_blocking) {
    at::TensorList input_list({ input });
    forward(input_list, output, non_blocking);
}
void BModelRunner::forward(at::TensorList input, at::Tensor& output, bool non_blocking) {
    at::TensorList output_list({ output });
    forward(input, output_list, non_blocking);
}

} // namespace modelrt
} // namespace at_tpu
