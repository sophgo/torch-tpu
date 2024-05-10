#include "TPUTorchUtils.h"

namespace tpu
{

void TPUCompareResult ( const at::Tensor & Got, const at::Tensor & Exp,
                        double Threshold, double ErrScale )
{
  if ( Got.dtype() != Exp.dtype() )
  {
    LOG ( FATAL ) << "Tensor comparing failed: Got data type = "
                  << Got.dtype() << ", Exp data type = " << Exp.dtype();
  }
  if ( Got.sizes() != Exp.sizes() )
  {
    LOG ( FATAL ) << "Tensor comparing failed: Got shape = "
                  << Got.sizes() << ", Exp shape = " << Exp.sizes();
  }
  if ( Got.dtype() == caffe2::TypeMeta::Make<float>() )
  {
    int ErrCnt = 0;
    const auto MaxErrCnt = 100;//Got.numel();
    auto Err = torch::sub ( Got, Exp );
    auto AbsErr = torch::abs ( Err );
    auto AbsExp = torch::abs ( Exp );
    auto RltAbsErr = torch::div ( AbsErr, AbsExp ) * ErrScale;
    auto ErrPtr = Err.data_ptr<float>();
    auto GotPtr = Got.data_ptr<float>();
    auto ExpPtr = Exp.data_ptr<float>();
    auto AbsErrPtr = AbsErr.data_ptr<float>();
    auto RltAbsErrPtr = RltAbsErr.data_ptr<float>();
    for ( auto i = 0; i < Got.numel(); ++i )
    {
      if ( std::isnan ( ExpPtr[i] ) == true && std::isnan ( GotPtr[i] ) == true )
      {
        continue;
      }
      if ( AbsErrPtr[i] < Threshold || RltAbsErrPtr[i] <= 1e-5 )
      {
        continue;
      }
// FAILED:
      if ( ErrCnt < MaxErrCnt )
      {
        LOG ( WARNING ) << "Compare failed: Got = " << GotPtr[i]
                        << ", Exp = " << ExpPtr[i]
                        << ", Err = " << ErrPtr[i]
                        << ", index = " << i;
      }
      else
      {
        LOG ( WARNING ) << "<Skip the other compare errors>";
        return;
      }
      ++ErrCnt;
    }
  }
  else
  {
    LOG ( FATAL ) << "Unsupported data type " << Got.dtype();
  }
}



static const char * OpTypeStr[OP_NUM] =
{
  "CDMA D2S",
  "CDMA S2D",
  "CDMA C2C",
  "Copy",
  "Cpu Layer",
  "Convolution",
  "Convolution Backward",
  "BatchNorm",
  "BatchNorm Backward",
  "LayerNorm",
  "LayerNorm Backward",
  "Avg Pooling",
  "Max Pooling",
  "ReLU",
  "ReLU Backward",
  "GeLU",
  "GeLU Backward",
  "Leaky ReLU",
  "MatMul",
  "Add MatMul",
  "Batch MatMul",
  "Linear",
  "Softmax",
  "Softmax Backward",
  "Log Softmax",
  "Permute",
  "Transpose",
  "Add",
  "Sub",
  "Mul",
  "Div",
  "Add Bcast",
  "Index Select",
  "DType Convert",
  "Reduce Mean",
  "Reduce Sum",
  "REDUCE_PROD",
  "Reduce Max",
  "Reduce Min",
  "Reduce var",
  "Where",
  "Strided Copy",
  "Concat",
  "Const Fill",
  "Masked Fill",
  "Sqrt",
  "Rsqrt",
  "Sign",
  "Addcdiv",
  "Addcmul",
  "Embedding Backward",
  "Malloc",
  "Free",
  "Cross Entropy Loss",
  "Cross Entropy Loss Backward",
  "AddC",
  "MulC",
  "CSub",
  "CDiv",
  "Norm2",
  "Native Group Norm",
  "Groupnorm backward",
  "Bcast Add",
  "MLP Forward",
  "MLP Backward",
  "Attention Forward",
  "Attention Backward",
  "BITWISE_XOR",
  "BITWISE_XOR_BCAST",
  "BITWISE_XOR_C",
  "Abs Forward",
  "Cos Forward",
  "Sin Forward",
  "Tan Forward",
  "Log Forward",
  "ACosH Forward",
  "ASinH Forward",
  "ATanH Forward",
  "SinH Forward",
  "CosH Forward",
  "TanH Forward",
  "Exp Forward",
  "Asin",
  "Acos",
  "Atan",
  "Sinh",
  "Cosh",
  "Tanh",
  "Ceil",
  "Floor",
  "Round",
  "Neg",
  "Exp2",
  "EXPM1",
  "Expand",
  "Flip",
  "Squeeze",
  "Unsqueeze",
  "Isfinite",
  "Isinf",
  "Isnan",
  "Bitwise_Not",
  "Minimum",
  "Maximum",
  "Fmin",
  "Fmax",
  "Atan2",
  "Logical And",
  "Logical Or",
  "BITWISE_AND",
  "BITWISE_AND_BCAST",
  "BITWISE_AND_C",
  "BITWISE_OR",
  "BITWISE_OR_BCAST",
  "BITWISE_OR_C",
  "EQUAL",
  "EQUAL_BCAST",
  "EQUAL_C",
  "GREATER_OR_EQUAL",
  "GREATER_OR_EQUAL_BCAST",
  "GREATER_OR_EQUAL_C",
  "GREATER",
  "GREATER_BCAST",
  "GREATER_C",
  "LESS_THAN_OR_EQUAL",
  "LESS_THAN_OR_EQUAL_BCAST",
  "LESS_THAN_OR_EQUAL_C",
  "SHIFT_LEFT",
  "SHIFT_LEFT_BCAST",
  "SHIFT_LEFT_C",
  "SHIFT_RIGHT_ARITHMETIC",
  "SHIFT_RIGHT_ARITHMETIC_BCAST",
  "SHIFT_RIGHT_ARITHMETIC_C",
  "LESS_THAN",
  "LESS_THAN_BCAST",
  "LESS_THAN_C",
  "NOT_EQUAL",
  "NOT_EQUAL_BCAST",
  "NOT_EQUAL_C",
  "SIGNBIT",
  "FULL",
  "Logical Not",
  "Upsampl Bilinear2d",
  "Upsampl nearest",
  "Upsampl nearest2d Backward",
  "Arrange",
  "SiLU",
  "Sigmoid",
  "CLAMP",
  "layernorm Matmul",
  "ERF",
  "ERFC",
  "Pow",
  "Pow_bcast",
  "POW SCALAR",
  "SCALAR POW",
  "Add Layernorm Matmul",
  "RECIPROCAL",
  "TRUNC",
  "layernorm Matmul Backward",
  "Add layernorm Matmul Backward",
  "TOPK",
  "NonZero",
  "REPEAT",
  "Argmax",
  "Argmin",
  "Max_dim",
  "Min_dim",
  "HARDTANH",
  "HYPOT",
  "Nextafter",
  "Triu",
  "Cbrt",
  "Constant_pad",
  "Reflection_pad2d",
  "Replication_pad2d",
  "Replication_pad3d",
  "GATHER",
  "BADDBMM",
  "MSE Loss",
  "MSE Loss Backward",
  "Slice_scatter",
  "Inf Check And Unscale",
  "LLAMA_ATTENTION",
  "LLAMA_MLP_FORWARD",
  "RMSNORM_FORWARD",
  "binary_op",
  "binary_op_c",
  "binary_op_bcast",
  "real",
  "Conj",
  "Dummy"
};


OpTimer * OpTimer::instance_ = nullptr;

OpTimer & OpTimer::Clear()
{
  mutex_.lock();
  for ( auto i = 0; i < OP_NUM; ++i )
  {
    elapsed_time_us_[i] = 0;
  }
  mutex_.unlock();
  return *this;
}

OpTimer & OpTimer::Start()
{
  mutex_.lock();
  is_paused_ = false;
  is_start_  = true;
  mutex_.unlock();
  return *this;
}

OpTimer & OpTimer::Pause()
{
  mutex_.lock();
  is_paused_ = true;
  is_start_ = false;
  mutex_.unlock();
  return *this;
}

OpTimer & OpTimer::AddTime ( OpType type, unsigned long time_us )
{
  mutex_.lock();
  if ( is_paused_ == false )
  {
    elapsed_time_us_[type] += time_us;
    #ifdef SHOW_EACH_OP_TIME
    if (is_start_)
    {
      std::cout << std::setw ( 42 ) << OpTypeStr[type] << " Elapsed: " << std::setw ( 12 ) << time_us << "us" << "\n";
    }
    #endif
  }
  mutex_.unlock();
  return *this;
}

void OpTimer::Dump() const
{
  unsigned long long ElapsedAll = 0;
  for ( auto i = 0; i < OP_NUM; ++i )
  {
    if ( elapsed_time_us_[i] > 0 )
    {
      ElapsedAll += elapsed_time_us_[i];
    }
  }
  for ( auto i = 0; i < OP_NUM; ++i )
  {
    if ( elapsed_time_us_[i] > 0 )
    {
      std::cout << std::setw ( 42 ) << OpTypeStr[i] << ": " << std::setw ( 12 ) << elapsed_time_us_[i] << "us, ";
      std::cout << std::setw ( 8 ) << std::setprecision ( 3 ) << elapsed_time_us_[i] * 100. / ElapsedAll << "%" << std::endl;
    }
  }
  std::cout << "TPU Elapsed All: " << ElapsedAll << "us" << std::endl;
}

OpTimer & OpTimer::Instance()
{
  if ( instance_ == nullptr )
  {
    instance_ = new OpTimer;
  }
  return *instance_;
}

GlobalTimer * GlobalTimer::instance_ = nullptr;

GlobalTimer & GlobalTimer::Reset()
{
  timer_.Start();
  return *this;
}

void GlobalTimer::Dump() const
{
  std::cout << "TPU Elpased: " << timer_.ElapsedUS() << "us" << std::endl;
}

GlobalTimer & GlobalTimer::Instance()
{
  if ( instance_ == nullptr )
  {
    instance_ = new GlobalTimer;
  }
  return *instance_;
}

#ifdef TPU_OP_TIMING

TensorWatcher * TensorWatcher::instance_ = nullptr;

void TensorWatcher::AddTensor ( const at::Tensor & Tensor )
{
  tensors_.push_back ( Tensor );
  tensors_cpu_.push_back ( TENSOR_TO_CPU ( Tensor ) );
}

bool TensorWatcher::Watch() const
{
  for ( auto I = 0; I < (int)tensors_.size(); ++I )
  {
    if ( tensors_[I].defined() )
    {
      auto tensor_cpu = TENSOR_TO_CPU ( tensors_[I] );
      auto ptr_saved = ( unsigned char * ) tensors_cpu_[I].data_ptr();
      auto ptr_current = ( unsigned char * ) tensor_cpu.data_ptr();
      for ( auto i = 0; i < (int)tensors_[I].nbytes(); ++i )
      {
        if ( ptr_saved[i] != ptr_current[i] )
        {
          std::cout << "Exp[" << i << "] = " << ( int ) ptr_saved[i]
                    << " Got[" << i << "] = " << ( int ) ptr_current[i] << std::endl;
          return false;
        }
      }
    }
  }
  return true;
}

TensorWatcher & TensorWatcher::Instance()
{
  if ( instance_ == nullptr )
  {
    instance_ = new TensorWatcher;
  }
  return *instance_;
}
#endif // TPU_OP_TIMING
} // namespace tpu
