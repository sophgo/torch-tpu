#ifdef NO_PYTHON_API
#include <vector>
#include <string>
#include <ATen/ATen.h>
#include <ATen/NativeFunctions.h>
#include "TPUDeviceUtils.h"
#include "TPUBmodel.h"

namespace tpu{

namespace bmodel{

static void ReportAndDelete(void *ptr)
{
  // do nothing
  #ifdef DEBUG
//   std::cout << "ReportAndDelete for prt : " << ptr  << std::endl;
//   std::cout << "PLEASE BE CAREFUL, THIS FUNCTION SHOULD NOT WORK" << std::endl;
  #endif
}

static at::DataPtr make_tensor_ptr(void *ptr)
{
  return {ptr, ptr, &ReportAndDelete, tpu::TPUGetCurrentDevice()};
}

static int data_type_size(tpu_dtype_t dtype){
    #ifdef BACKEND_SG2260
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
    #elif defined BACKEND_1684X or defined BACKEND_1686
        switch (dtype) {
        case BM_FLOAT32:
        case BM_INT32:
        case BM_UINT32:
            return 4;
        case BM_FLOAT16:
        case BM_BFLOAT16:
        case BM_INT16:
            return 2;
        case BM_INT8:
        case BM_UINT8:
            return 1;
        case BM_INT4:
        case BM_UINT4:
            return 1;  // need modify ?  TODO
        default:
            return 4;
        }
    #endif
}

static auto convert_dtype_to_torch_dtype(tpu_dtype_t dtype)
{
    #ifdef BACKEND_SG2260
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
    #elif defined BACKEND_1684X or defined BACKEND_1686
        switch (dtype) {
            case BM_FLOAT32:
            return at::kFloat;
            case BM_INT32:
            return at::kInt;
            case BM_UINT32:
            return at::kInt;
            case BM_FLOAT16:
            return at::kHalf;
            case BM_BFLOAT16:
            return at::kBFloat16;
            case BM_INT16:
            return at::kShort;
            case BM_UINT16:
            return at::kShort;
            case BM_INT8:
            return at::kChar;
            case BM_UINT8:
            return at::kByte;
            case BM_INT4:
            return at::kChar;
            case BM_UINT4:
            return at::kByte;
            default:
            return at::kFloat;
        }
    #endif
}

static auto make_tensor_from_ptr(u64 ptr, std::vector<int> shape, tpu_dtype_t dtype)
{
    size_t size = 1;
    for(auto s : shape){
        size *= s;
    }
    size *= data_type_size(dtype);
    auto meta_dtype = convert_dtype_to_torch_dtype(dtype);
    auto tensor_dtype = at::scalarTypeToTypeMeta(meta_dtype);
    at::detail::check_size_nonnegative( size );
    #ifdef DEBUG
    // std::cout << "ptr : " << ptr << "  size : " << size << std::endl;
    #endif
    auto ptr_ = make_tensor_ptr((void*) ptr);
    auto allocator = c10::GetTPUAllocator();
    c10::intrusive_ptr<c10::StorageImpl> storage_impl = c10::make_intrusive<torch_tpu::TPUStorageImpl>(
                        c10::StorageImpl::use_byte_size_t(),
                        size,
                        std::move(ptr_),
                        allocator,
                        /*resizeable=*/false );
    auto tensor = at::detail::make_tensor<torch_tpu::TPUTensorImpl>(storage_impl, tensor_dtype);
    std::vector<int64_t> sizes;
    for(auto s : shape){
        sizes.push_back(s);
    }
    tensor.unsafeGetTensorImpl()->set_sizes_contiguous(sizes);
    return tensor;
}

TPUBmodel::TPUBmodel(const std::string cur_model_file, int dev_id, const std::string cur_decrypt_lib){
    dev_id       = dev_id;
    model_file   = cur_model_file;
    decrypt_lib  = cur_decrypt_lib;

    at::Device device(at::DeviceType::TPU, dev_id);
    torch_tpu::utils::maybe_initialize_tpu(device);

    #if defined BACKEND_SG2260
        tpuRtSetDevice(dev_id);
        tpuRtCreateNetContext(&m_context);
        tpuRtLoadNet(model_file.c_str(), m_context, &m_net);
    #elif defined BACKEND_1684X or defined BACKEND_1686
        bm_handle_t bm_handle;
        bm_dev_request(&bm_handle, dev_id);
        m_stream = bm_handle;
        void* p_bmrt = bmrt_create(bm_handle);
        m_net = p_bmrt;
        bool flag = true;
        if(decrypt_lib.empty()){
            flag = bmrt_load_bmodel(m_net, model_file.c_str());
        }else{
            assert(0 && "decrypt lib not supported");
        }
        assert(flag == true);
    #endif

    #if defined BACKEND_SG2260
        char** net_names = NULL;
        m_net_num = tpuRtGetNetNames(m_net, &net_names);
    #elif defined BACKEND_1684X or defined BACKEND_1686
        const char **net_names = NULL;
        bmrt_get_network_names(m_net, &net_names);
        m_net_num = bmrt_get_network_number(m_net);
    #endif
    for (int i = 0; i < m_net_num; i++) {
        networks.push_back(net_names[i]);
    }
    cur_net_id = 0;
    delete net_names;

    // handle each net
    net_infos.resize(m_net_num);
    input_tensors.resize(m_net_num);
    output_tensors.resize(m_net_num);
    input_dtypes.resize(m_net_num);
    output_dtypes.resize(m_net_num);
    input_shapes.resize(m_net_num);
    output_shapes.resize(m_net_num);
    input_num.resize(m_net_num);
    output_num.resize(m_net_num);
    input_mems.resize(m_net_num);
    output_mems.resize(m_net_num);
    input_names.resize(m_net_num);
    output_names.resize(m_net_num);
    for(int i = 0; i < m_net_num; i++){
        cur_net_id = i;
        handle_each_net();
    }
    cur_net_id = 0;
}

TPUBmodel::~TPUBmodel(){
    #if defined BACKEND_SG2260
        tpuRtUnloadNet(m_net);
        tpuRtDestroyNetContext(m_context);
    #elif defined BACKEND_1684X or defined BACKEND_1686
        void* p_bmrt = m_net;
        bmrt_destroy(p_bmrt);
    #endif
}

void TPUBmodel::handle_each_net(){
    const char* net_name = networks[cur_net_id].c_str();
    // check each net and its stage
    #if defined BACKEND_SG2260
        tpu_net_info_t info    = tpuRtGetNetInfo(m_net, net_name);
        input_num[cur_net_id]  = info.input.num;
        output_num[cur_net_id] = info.output.num;
        input_tensors[cur_net_id].resize(info.input.num);
        output_tensors[cur_net_id].resize(info.output.num);
        input_shapes[cur_net_id].resize(info.input.num);
        output_shapes[cur_net_id].resize(info.output.num);
        input_dtypes[cur_net_id].resize(info.input.num);
        output_dtypes[cur_net_id].resize(info.output.num);
        input_mems[cur_net_id].resize(info.input.num);
        output_mems[cur_net_id].resize(info.output.num);
        input_names[cur_net_id].resize(info.input.num);
        output_names[cur_net_id].resize(info.output.num);
        net_infos[cur_net_id]  = info;
        assert(info.stage_num == 1 && "only support one stage");
        for(int i = 0; i < info.input.num; i++){
            // fix with 2260
            auto shape = info.stages[0].input_shapes[i];
            input_shapes[cur_net_id][i].resize(shape.num_dims);
            for (int j = 0; j < shape.num_dims; j++) {
                input_shapes[cur_net_id][i][j] = shape.dims[j];
            }
            input_dtypes[cur_net_id][i] = info.input.dtypes[i];
            input_mems[cur_net_id][i]   = (u64) info.stages[0].input_mems[i];
            input_names[cur_net_id][i]  = std::string(info.input.names[i]);
            getTensor(i, true);
        }
        for(int i = 0; i < info.output.num; i++){
            auto shape = info.stages[0].output_shapes[i];
            output_shapes[cur_net_id][i].resize(shape.num_dims);
            for (int j = 0; j < shape.num_dims; j++) {
                output_shapes[cur_net_id][i][j] = shape.dims[j];
            }
            output_dtypes[cur_net_id][i] = info.output.dtypes[i];
            output_mems[cur_net_id][i]   = (u64) info.stages[0].output_mems[i];
            output_names[cur_net_id][i]  = std::string(info.output.names[i]);
            getTensor(i, false);
        }

    #elif defined BACKEND_1684X or defined BACKEND_1686
        void* p_bmrt           = m_net;
        tpu_net_info_t info    = bmrt_get_network_info(p_bmrt, net_name);
        input_num[cur_net_id]  = info->input_num;
        output_num[cur_net_id] = info->output_num;
        net_infos[cur_net_id]  = info;
        input_tensors[cur_net_id].resize(info->input_num);
        output_tensors[cur_net_id].resize(info->output_num);
        input_shapes[cur_net_id].resize(info->input_num);
        output_shapes[cur_net_id].resize(info->output_num);
        input_dtypes[cur_net_id].resize(info->input_num);
        output_dtypes[cur_net_id].resize(info->output_num);
        input_mems[cur_net_id].resize(info->input_num);
        output_mems[cur_net_id].resize(info->output_num);
        input_names[cur_net_id].resize(info->input_num);
        output_names[cur_net_id].resize(info->output_num);
        assert(info->stage_num == 1 && "only support one stage");
        for(int i = 0; i < info->input_num; i++){
            bm_shape_t shape = info->stages[0].input_shapes[i];
            input_shapes[cur_net_id][i].resize(shape.num_dims);
            for (int j = 0; j < shape.num_dims; j++) {
                input_shapes[cur_net_id][i][j] = shape.dims[j];
            }
            input_dtypes[cur_net_id][i] = info->input_dtypes[i];
            bm_device_mem_t tensor      = info->stages[0].input_mems[i];
            input_mems[cur_net_id][i]   = (u64) tensor.u.device.device_addr;
            input_names[cur_net_id][i]  = info->input_names[i];
            getTensor(i, true);
        }
        for(int i = 0; i < info->output_num; i++){
            bm_shape_t shape = info->stages[0].output_shapes[i];
            output_shapes[cur_net_id][i].resize(shape.num_dims);
            for (int j = 0; j < shape.num_dims; j++) {
                output_shapes[cur_net_id][i][j] = shape.dims[j];
            }
            output_dtypes[cur_net_id][i] = info->output_dtypes[i];
            output_mems[cur_net_id][i]   = (u64) info->stages[0].output_mems[i].u.device.device_addr;
            output_names[cur_net_id][i]  = info->output_names[i];
            getTensor(i, false);
        }
    #endif
}

void TPUBmodel::getTensor(int idx, bool is_input){
    auto ptr   = is_input ? input_mems[cur_net_id][idx]   : output_mems[cur_net_id][idx];
    auto shape = is_input ? input_shapes[cur_net_id][idx] : output_shapes[cur_net_id][idx];
    auto dtype = is_input ? input_dtypes[cur_net_id][idx] : output_dtypes[cur_net_id][idx];
    auto tensor = make_tensor_from_ptr(ptr, shape, dtype);
    if(is_input){
        input_tensors[cur_net_id][idx] = tensor;
    }else{
        output_tensors[cur_net_id][idx] = tensor;
    }
}

void TPUBmodel::forward(){
    std::vector<at::Tensor> cur_input_tensors  = input_tensors[cur_net_id];
    std::vector<at::Tensor> cur_output_tensors = output_tensors[cur_net_id];
    forward(cur_input_tensors, cur_output_tensors);
}

void TPUBmodel::forward_sync(){
    forward();
    #if defined BACKEND_SG2260
        auto stream = c10_tpu::getCurrentTPUStream();
        tpuRtStreamSynchronize(stream);
    #elif defined BACKEND_1684X or defined BACKEND_1686
        auto status = bm_thread_sync(m_stream);
        if(status != BM_SUCCESS){
            std::cout << "bm_thread_sync failed" << std::endl;
            exit(1);
        }
    #endif
}

void TPUBmodel::forward(std::vector<at::Tensor> input_tensors, std::vector<at::Tensor> output_tensors){

    const char* net_name = networks[cur_net_id].c_str();
    int inner_num_input  = input_tensors.size();
    int inner_num_output = output_tensors.size();

    int num_input        = input_num[cur_net_id];
    int num_output       = output_num[cur_net_id];

    assert(inner_num_input  == num_input  && "input size not match");
    assert(inner_num_output == num_output && "output size not match");

    std::vector<tpu_tensor_t> inputs;
    std::vector<tpu_tensor_t> outputs;
    inputs.resize(num_input);
    outputs.resize(num_output);

    for(int i = 0; i < num_input; i++){
        tpu_tensor_t tensor;
        tpu_shape_t  shape;
        size_t       size = 1;
        shape.num_dims = input_shapes[cur_net_id][i].size();
        for(int j = 0; j < shape.num_dims; j++){
            shape.dims[j] = input_shapes[cur_net_id][i][j];
            size *= shape.dims[j];
        }
        size *= data_type_size((tpu_dtype_t) input_dtypes[cur_net_id][i]);
        #if defined BACKEND_SG2260
            tensor.data    = (void*) input_tensors[i].data_ptr();
        #elif defined BACKEND_1684X or defined BACKEND_1686
            tensor.device_mem.size = size;
            tensor.st_mode = BM_STORE_1N;
            tensor.device_mem.u.device.device_addr = (u64) input_tensors[i].data_ptr();
        #endif
        tensor.shape   = shape;
        tensor.dtype   = (tpu_dtype_t) input_dtypes[cur_net_id][i];
        inputs[i]      = tensor;
    }

    for(int i = 0; i < num_output; i++){
        tpu_tensor_t tensor;
        tpu_shape_t  shape;
        size_t      size = 1;
        shape.num_dims = output_shapes[cur_net_id][i].size();
        for(int j = 0; j < shape.num_dims; j++){
            shape.dims[j] = output_shapes[cur_net_id][i][j];
            size *= shape.dims[j];
        }
        size *= data_type_size((tpu_dtype_t) output_dtypes[cur_net_id][i]);
        #if defined BACKEND_SG2260
            tensor.data    = (void*) output_tensors[i].data_ptr();
        #elif defined BACKEND_1684X or defined BACKEND_1686
            tensor.st_mode = BM_STORE_1N;
            tensor.device_mem.size = size;
            tensor.device_mem.u.device.device_addr = (u64) output_tensors[i].data_ptr();
        #endif
        tensor.shape   = shape;
        tensor.dtype   = (tpu_dtype_t) output_dtypes[cur_net_id][i];
        outputs[i]     = tensor;
    }

    #if defined BACKEND_SG2260
        auto stream = c10_tpu::getCurrentTPUStream();
        auto status = tpuRtLaunchNetAsync(m_net, inputs.data(), outputs.data(), net_name, stream);
        if(status != tpuRtSuccess){
            std::cout << "tpuRtLaunchNet failed" << std::endl;
            exit(1);
        }
    #elif defined BACKEND_1684X or defined BACKEND_1686
        void* p_bmrt = m_net;
        auto status = bmrt_launch_tensor_ex(p_bmrt, net_name, inputs.data(), num_input, outputs.data(), num_output, true, false);
        if(status != true){
            std::cout << "bmrt_launch_network failed" << std::endl;
            exit(1);
        }
    #endif
}

void TPUBmodel::forward_sync(std::vector<at::Tensor> input_tensors, std::vector<at::Tensor> output_tensors){
    forward(input_tensors, output_tensors);
    #if defined BACKEND_SG2260
        auto stream = c10_tpu::getCurrentTPUStream();
        tpuRtStreamSynchronize(stream);
    #elif defined BACKEND_1684X or defined BACKEND_1686
        auto status = bm_thread_sync(m_stream);
        if(status != BM_SUCCESS){
            std::cout << "bm_thread_sync failed" << std::endl;
            exit(1);
        }
    #endif
}

}
}
#endif