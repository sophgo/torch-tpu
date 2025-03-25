#pragma once
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include <optional>
#include <functional>
#include <cstring>

#include <torch/torch.h>

#include "aten/TPUGeneratorImpl.h"
#include "TPUDeviceManager.h"
#include "TPUGuard.h"

// macro
#define F16  torch::kHalf
#define F32  torch::kFloat
#define Long torch::kLong

class SaveTensors {

  public:
    torch::serialize::OutputArchive archive;

    SaveTensors() {
        archive = torch::serialize::OutputArchive();
    }

    void write(const std::string& name, torch::Tensor tensor) {
        archive.write(name, tensor.to(torch::kCPU));
    }

    void save(const std::string& file_path) {
        try {
            std::ofstream outfile(file_path, std::ios::binary);
            archive.save_to(outfile);
            outfile.close();
            std::cout << "Tensors saved successfully.\n";
        } catch (const c10::Error& e) {
            std::cerr << "Error saving tensors: " << e.what() << std::endl;
        }
    }
};

static inline torch::nn::Conv2dOptions conv_options(int64_t in_planes, int64_t out_planes, int64_t kerner_size,
	int64_t stride = 1, int64_t padding = 1, int groups = 1, bool with_bias = false) {

	torch::nn::Conv2dOptions conv_options = torch::nn::Conv2dOptions(in_planes, out_planes, kerner_size);
	conv_options.stride(stride).padding(padding).bias(with_bias).groups(groups);
	return conv_options;
}

static inline torch::nn::Conv2d conv2d(int64_t in_planes, int64_t out_planes, int64_t kernel_size,
                                         int64_t stride = 1, int64_t padding = 0, int groups = 1, bool with_bias = false) {
    return torch::nn::Conv2d(conv_options(in_planes, out_planes, kernel_size, stride, padding, groups, with_bias));
}

static inline torch::nn::MaxPool2d maxpool2d(int64_t kernel_size, int64_t stride = 1, int64_t padding = 0) {
    return torch::nn::MaxPool2d(torch::nn::MaxPool2dOptions(kernel_size).stride(stride).padding(padding));
}

struct ResidualBlockImpl : torch::nn::Module {
    torch::nn::Conv2d conv1, conv2;
    torch::nn::BatchNorm2d bn1, bn2;
    torch::nn::ReLU relu;
    torch::nn::Conv2d downsample_conv = nullptr;
	torch::nn::BatchNorm2d bn3 = nullptr;
    bool downsample;

    ResidualBlockImpl(int in_channels, int out_channels, bool downsample = false)
        : conv1(register_module("conv1", conv2d(in_channels, out_channels, 3, downsample ? 2 : 1, 1))),
          bn1(register_module("bn1", torch::nn::BatchNorm2d(out_channels))),
          conv2(register_module("conv2", conv2d(out_channels, out_channels, 3, 1, 1))),
          bn2(register_module("bn2", torch::nn::BatchNorm2d(out_channels))),
          relu(),
          downsample(downsample) {
			if (downsample) {
				downsample_conv = register_module("downsample_conv", conv2d(in_channels, out_channels, 1, 2));
				bn3             = register_module("bn3", torch::nn::BatchNorm2d(out_channels));
			}
    }

    torch::Tensor forward(torch::Tensor x) {
        auto identity = x;
        x = relu(bn1(conv1(x)));
        x = bn2(conv2(x));

        if (downsample) {
            identity = downsample_conv(identity);
			identity = bn3(identity);
        }
        x += identity;
        return relu(x);
    }
};

TORCH_MODULE(ResidualBlock);