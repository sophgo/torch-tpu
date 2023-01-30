#pragma once

#include <torch/torch.h>
#include <torch/script.h>

namespace tpu
{
void MoveModuleToTPUDevice ( torch::nn::Module & Module );
} // namespace tpu
