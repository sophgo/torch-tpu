#pragma once

#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include "json_help.h"
#include "pt_help.h"
#include "aten/TPUGeneratorImpl.h"
#include "TPUDeviceManager.h"
#include "TPUGuard.h"
#include "TPUBmodel.h"

class TPUCompile {
public:
    // current only support one net and one stage
    TPUCompile(std::string bmodel_path, std::string info_path, std::string pt_path, int device_id=0, std::string chip="bm1684x"){
        this->bmodel_path      = bmodel_path;
        this->pt_path          = pt_path;// load pt model
        this->info_path        = info_path;
        this->device_id        = device_id;
        this->chip             = chip;
        this->model            = new tpu::bmodel::TPUBmodel(bmodel_path, device_id, "");
        this->model_info       = read_from_config(info_path);
        this->pt_param_buffers = read_from_pt(pt_path);
        this->total_input_num  = this->model->input_num[0];
        this->total_output_num = this->model->output_num[0];
        this->user_input_num   = this->model_info.user_inputs.size();
        this->user_output_num  = this->model_info.user_outputs.size();
        this->param_num        = this->pt_param_buffers.input_params.size();
        this->buffer_num       = this->pt_param_buffers.input_buffers.size();

        for(int idx = 0; idx < this->param_num; idx++){
            this->model_parameters.push_back( this->model->input_tensors[0][idx] );
        }
        // get parameters
    };

    ~TPUCompile(){
        delete this->pt_param_buffers.module;
        delete this->model;
    };

    void handle_input_params_buffers(){
        // if you want to modify the input tensor(with init),
        //    such as do 32ic and 32oc
        //  you should change this->pt_param_buffers
        //  we only cpy data from pt_param_buffers to model.input_tensors
        //  if you want to cpy more times, you should change have_init_params_buffers to false
        if(this->have_init_params_buffers){
            return;
        }
        for(int idx = 0; idx < this->param_num; idx++){
            auto& bmodel_tensor = this->model->input_tensors[0][idx];
            auto& pt_tensor     = this->pt_param_buffers.input_params[idx];
            // 打印 device
            bmodel_tensor.copy_( pt_tensor.to( bmodel_tensor.dtype() ));
        }
        for(int idx = 0; idx < this->buffer_num; idx++){
            auto& bmodel_tensor = this->model->input_tensors[0][idx + this->param_num];
            auto& pt_tensor     = this->pt_param_buffers.input_buffers[idx];
            bmodel_tensor.copy_( pt_tensor.to( bmodel_tensor.dtype() ));
        }
        this->have_init_params_buffers = true;
    };

    void handle_input_user(){
        // we **always** cpy user data into bmodel tensors
        //  so you do not need to make sure the user input data is on device
        //  if you want to modify, change this function

        assert( this->user_input_tensors.size() == this->model_info.user_inputs.size() );

        for(int idx = 0; idx < this->user_input_num; idx++){
            auto& bmodel_tensor = this->model->input_tensors[0][idx + this->param_num + this->buffer_num];
            auto& user_tensor   = this->user_input_tensors[idx];
            bmodel_tensor.copy_( user_tensor.to( bmodel_tensor.dtype() ));
        }
    };

    void handle_input_extra(){
        if( this->user_input_num + this->param_num + this->buffer_num >= this->total_input_num ){
            return;
        }
        int idx = this->user_input_num + this->param_num + this->buffer_num;
        for(; idx < this->total_input_num; idx++){
            std::string name = this->model->input_names[0][idx];
            if(isSubstring("dropout", name.c_str())){
                // random tensor
                auto size = this->model->input_tensors[0][idx].sizes();
                torch::Tensor random_tensor = torch::rand( size , this->model->input_tensors[0][idx].dtype() );
                this->model->input_tensors[0][idx].copy_( random_tensor );
            }else{
                std::cerr << "this kind of name '" << name << "' is not supported" << std::endl;
                exit(1);
            }
        }
    };

    void handle_tpu_input(){
        this->handle_input_params_buffers();
        this->handle_input_user();
        this->handle_input_extra();
    };

    void handle_normalize_output(){
        // this function is used to normalize the output tensor
        //   (such as 32ic->normal | other -> normal)

    };

    void handle_tpu_output_buffers_params(){
        // output: buffers    | user_outputs   | gradients
        user_output_tensors.clear();
        // for finetune some layers, you should take care of this function
        //   you can not use buffer_num which is forward buffer_num, not buffer_num that used updated
        //   i fix it with model_info
        int output_idx = 0;
        // buffers
        for(auto idx: this->model_info.output_buffers){
            model->input_tensors[0][idx].copy_( model->output_tensors[0][output_idx].to( model->input_tensors[0][idx].dtype() ));
            output_idx += 1;
        }
        // user_outputs
        for(int idx = 0; idx < user_output_num; idx++){
            user_output_tensors.push_back( model->output_tensors[0][output_idx] );
            output_idx += 1;
        }
        user_output_tensors[0].set_requires_grad(true);
        // gradients
        for(auto idx: this->model_info.output_params){
            model->input_tensors[0][idx].set_requires_grad(true);
            model->input_tensors[0][idx].mutable_grad() = model->output_tensors[0][output_idx];
            output_idx += 1;
        }
        assert( output_idx == this->total_output_num );
    };

    void forward(){
        TIME_START;
        handle_input_params_buffers();
        handle_input_user();
        handle_input_extra();
        TIME_START_NAME(forward_sync);
        model->forward_sync();
        TIME_END_NAME(forward_sync);
        handle_normalize_output();
        handle_tpu_output_buffers_params();
        TIME_END;
    };

    template <typename... Args>
    std::vector<torch::Tensor> forward(Args&&... args){
        user_input_tensors.clear();
        user_input_tensors = {std::forward<Args>(args)...};
        forward();
        return user_output_tensors;
    };

    void check_params();

    std::vector<torch::Tensor> parameters(){
        return model_parameters;
    };

    TPUCompile* cpu(){
        for(auto& tensor: model->input_tensors[0]){
            tensor.to(torch::kCPU);
        }
        return this;
    };

    void save2pt(std::string save_path){
        // update pt module with model.input_tensors and save into pt

        auto module = this->pt_param_buffers.module;

        int idx = 0;
        for(const auto& pair : module->named_parameters()){
            torch::Tensor tensor = pair.value;
            tensor.data() = this->model->input_tensors[0][idx].to( torch::kCPU );
            idx += 1;
        }

        for(const auto& pair : module->named_buffers()){
            torch::Tensor tensor = pair.value;
            tensor.data() = this->model->input_tensors[0][idx].to( torch::kCPU );
            idx += 1;
        }

        this->pt_param_buffers.module->save(save_path);
    };

    void save(std::string save_path){
        // may need to normalize back to normal tensor & shape
        // save model.input_tensors into bmodel with parameters names (model_info.input_params)
        // please make use the tensor have been in cpu
        SaveTensors save_tensors;
        // save params
        for(int idx = 0; idx < this->param_num; idx++){
            save_tensors.write( this->model_info.input_params[idx], this->model->input_tensors[0][idx] );
        }
        // save buffers
        for(int idx = 0; idx < this->buffer_num; idx++){
            save_tensors.write( this->model_info.input_buffers[idx], this->model->input_tensors[0][idx + this->param_num] );
        }
        save_tensors.save(save_path);
    };

    int device_id;
    std::string chip;
    std::string bmodel_path;
    std::string info_path;
    std::string pt_path;
    bool have_init_params_buffers = false;

    tpu::bmodel::TPUBmodel* model;
    JitModelInfo_t model_info;
    param_buffers_t pt_param_buffers;

    int total_input_num;
    int total_output_num;
    int param_num;
    int buffer_num;
    int user_input_num;
    int user_output_num;

    std::vector<torch::Tensor> user_input_tensors;
    std::vector<torch::Tensor> user_output_tensors;

    std::vector<torch::Tensor> norm_output_tensors;
    std::vector<torch::Tensor> model_parameters;
};