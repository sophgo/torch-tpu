#include <torch/torch.h>
#include <torch/script.h>

struct Net : torch::nn::Module {
  Net() {
    // Construct and register two Linear submodules.
    fc1 = register_module ( "fc1", torch::nn::Linear ( 784, 64 ) );
  }

  // Implement the Net's algorithm.
  torch::Tensor forward ( torch::Tensor x ) {
    // Use one of many tensor manipulation functions.
    x = torch::relu ( fc1->forward ( x.reshape ( {x.size ( 0 ), 784} ) ) );
    return x;
  }

  // Use one of many "standard library" modules.
  torch::nn::Linear fc1{nullptr};
};

int main()
{
  auto net = std::make_shared<Net>();
  torch::Device Device ( "privateuseone" );
  const int Num = 10;
  auto A = torch::randn ( { Num } );
  auto APtr = A.data_ptr<float>();
  for ( int i = 0; i < Num; ++i )
  {
    std::cout << APtr[i] << " ";
  }
  std::cout << std::endl;
  auto B = A.to ( Device );
  auto C = B.to ( torch::Device ( "cpu" ) );
  auto CPtr = C.data_ptr<float>();
  for ( int i = 0; i < Num; ++i )
  {
    std::cout << CPtr[i] << " ";
  }
  std::cout << std::endl;
  //net->to ( Device );
  return 0;
}
