#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>

typedef struct {
    torch::jit::script::Module* module;
    std::vector<torch::Tensor> input_params;
    std::vector<torch::Tensor> input_buffers;
    std::vector<std::string>   input_param_names;
    std::vector<std::string>   input_buffer_names;
} param_buffers_t;

static inline param_buffers_t read_from_pt(const std::string& pt_path) {
    auto module = new torch::jit::script::Module(torch::jit::load(pt_path));
    param_buffers_t pb;
    pb.module = module;
    for (const auto& pair : module->named_parameters()) {
        pb.input_param_names.push_back(pair.name);
        pb.input_params.push_back(pair.value);
    }
    for (const auto& pair : module->named_buffers()) {
        pb.input_buffer_names.push_back(pair.name);
        pb.input_buffers.push_back(pair.value);
    }
    return pb;
}