#include <torch/torch.h>
#include <torch/script.h>
#include <TPUModule.h>

struct Net : torch::nn::Module {
  Net() {
    // Construct and register two Linear submodules.
    fc1 = register_module ( "fc1", torch::nn::Linear ( 784, 64 ) );
    fc2 = register_module ( "fc2", torch::nn::Linear ( 64, 32 ) );
    fc3 = register_module ( "fc3", torch::nn::Linear ( 32, 10 ) );
  }

  // Implement the Net's algorithm.
  torch::Tensor forward ( torch::Tensor x ) {
    // Use one of many tensor manipulation functions.
    x = torch::relu ( fc1->forward ( x ) );
    x = torch::dropout ( x, /*p=*/0.5, /*train=*/is_training() );
    x = torch::relu ( fc2->forward ( x ) );
    x = torch::log_softmax ( fc3->forward ( x ), /*dim=*/1 );
    return x;
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

int main()
{
  auto net = std::make_shared<Net>();
  torch::Device Device ( "privateuseone" );
  auto A = torch::randn ( { 1, 784 } ).to ( Device );
  tpu::MoveModuleToTPUDevice ( *net );
  net->forward ( A );
  return 0;
}
