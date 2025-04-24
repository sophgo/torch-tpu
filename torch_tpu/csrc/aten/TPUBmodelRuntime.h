
#pragma once
#include <vector>
#include <string>

#include <ATen/Tensor.h>
#include <ATen/ATen.h>

namespace at_tpu{
namespace modelrt{

struct BModelRunnerImpl;

struct BModelRunner {
    using tensor_vec = std::vector<at::Tensor>;
    using name_vec   = std::vector<std::string>;
    
    BModelRunner(const std::string &model_file, int dev_id,
            const std::string &decrypt_lib);
    ~BModelRunner();

    void set_running_net(const char* net_name);
    void forward(at::Tensor& input,     at::Tensor& output,     bool non_blocking = 0);
    void forward(at::TensorList input,  at::Tensor& output,     bool non_blocking = 0);
    void forward(at::Tensor& input,     at::TensorList output,  bool non_blocking = 0);
    void forward(at::TensorList input,  at::TensorList output,  bool non_blocking = 0);
    std::tuple<tensor_vec, tensor_vec>  GenInplaceTensor();
    std::tuple<name_vec, name_vec>      GetIONames();
private:
    int                         m_dev_id = 0;
    BModelRunnerImpl*           m_pimpl;
};
} // namespace modelrt
} // namespaca at_tpu