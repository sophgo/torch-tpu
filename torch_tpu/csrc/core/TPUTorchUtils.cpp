#include "TPUTorchUtils.h"

namespace tpu
{

void print_python_code() {
    PyGILState_STATE gstate = PyGILState_Ensure();
    
    PyThreadState *tstate = PyThreadState_Get();
    if (tstate && tstate->frame) {
        PyFrameObject *frame = tstate->frame;
        while (frame) {
            PyCodeObject *code = frame->f_code;
            PyObject *filename = code->co_filename;
            PyObject *name = code->co_name;
            int lineno = PyFrame_GetLineNumber(frame);
            if (filename && PyUnicode_Check(filename) && name && PyUnicode_Check(name)) {
                const char *filename_str = PyUnicode_AsUTF8(filename);
                const char *name_str = PyUnicode_AsUTF8(name);
                printf("File: \"%s\", line %d, in %s\n", filename_str, lineno, name_str);
                std::ifstream file(filename_str);
                std::string line;
                for (int i = 1; std::getline(file, line); i++) {
                  if (i == lineno) {
                    std::cout << line << std::endl;
                    break;
                  }
                }
            }
            // Clean up references
            frame = frame->f_back;
        }
    }

    PyGILState_Release(gstate);
}

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


static inline const char *OpTypeStr(int v)
{
  switch (v)
  {
#define __case(k)   \
    case k:         \
      return #k;

    __case(CDMA_D2S)
    __case(CDMA_S2D)
    __case(CDMA_C2C)
    __case(COPY)
    __case(CPU_LAYER)
    __case(CONVOLUTION)
    __case(CONVOLUTION_BACKWARD)
    __case(BATCHNORM)
    __case(BATCHNORM_BACKWARD)
    __case(LAYERNORM)
    __case(LAYERNORM_BACKWARD)
    __case(AVG_POOLING)
    __case(MAX_POOLING)
    __case(RELU)
    __case(RELU_BACKWARD)
    __case(GELU)
    __case(GELU_BACKWARD)
    __case(LEAKY_RELU)
    __case(MM)
    __case(LLama2A16MATMUL)
    __case(ADDMM)
    __case(BMM)
    __case(LINEAR)
    __case(SOFTMAX)
    __case(SOFTMAX_BACKWARD)
    __case(LOGSOFTMAX)
    __case(PERMUTE)
    __case(TRANSPOSE)
    __case(ADD)
    __case(SUB)
    __case(MUL)
    __case(DIV)
    __case(ADDBCAST)
    __case(INDEX_SELECT)
    __case(DTYPE_CONVERT)
    __case(REDUCE_MEAN)
    __case(REDUCE_SUM)
    __case(REDUCE_PROD)
    __case(REDUCE_MAX)
    __case(REDUCE_MIN)
    __case(REDUCE_VAR)
    __case(WHERE)
    __case(STRIDED_COPY)
    __case(CONCAT)
    __case(CONST_FILL)
    __case(MASKED_FILL)
    __case(SQRT)
    __case(RSQRT)
    __case(SIGN)
    __case(ADDCDIV)
    __case(ADDCMUL)
    __case(EMBEDDING_BACKWARD)
    __case(MALLOC)
    __case(FREE)
    __case(CROSS_ENTROPY_LOSS)
    __case(CROSS_ENTROPY_LOSS_BACKWARD)
    __case(ADD_C)
    __case(MUL_C)
    __case(C_SUB)
    __case(C_DIV)
    __case(NORM2)
    __case(NATIVE_GROUP_NORM)
    __case(GROUPNORM_BACKWARD)
    __case(BCAST_ADD)
    __case(MLP_FORWARD)
    __case(MLP_BACKWARD)
    __case(ATTN_FORWARD)
    __case(ATTN_BACKWARD)
    __case(BITWISE_XOR)
    __case(BITWISE_XOR_BCAST)
    __case(BITWISE_XOR_C)
    __case(ABS_FORWARD)
    __case(COS_FORWARD)
    __case(SIN_FORWARD)
    __case(TAN_FORWARD)
    __case(LOG_FORWARD)
    __case(ACOSH_FORWARD)
    __case(ASINH_FORWARD)
    __case(ATANH_FORWARD)
    __case(SINH_FORWARD)
    __case(COSH_FORWARD)
    __case(TANH_FORWARD)
    __case(EXP_FORWARD)
    __case(ASIN)
    __case(ACOS)
    __case(ATAN)
    __case(SINH)
    __case(COSH)
    __case(TANH)
    __case(TANH_BACKWARD)
    __case(SIGMOID_BACKWARD)
    __case(SILU_BACKWARD)
    __case(CEIL)
    __case(FLOOR)
    __case(ROUND)
    __case(NEG)
    __case(EXP2)
    __case(EXPM1)
    __case(EXPAND)
    __case(FLIP)
    __case(SQUEEZE)
    __case(UNSQUEEZE)
    __case(ISFINITE)
    __case(ISINF)
    __case(ISNAN)
    __case(BITWISE_NOT)
    __case(MINIMUM)
    __case(MAXIMUM)
    __case(FMIN)
    __case(FMAX)
    __case(ATAN2)
    __case(LOGICAL_AND)
    __case(LOGICAL_OR)
    __case(BITWISE_AND)
    __case(BITWISE_AND_BCAST)
    __case(BITWISE_AND_C)
    __case(BITWISE_OR)
    __case(BITWISE_OR_BCAST)
    __case(BITWISE_OR_C)
    __case(EQUAL)
    __case(EQUAL_BCAST)
    __case(EQUAL_C)
    __case(GREATER_OR_EQUAL)
    __case(GREATER_OR_EQUAL_BCAST)
    __case(GREATER_OR_EQUAL_C)
    __case(GREATER)
    __case(GREATER_BCAST)
    __case(GREATER_C)
    __case(LESS_THAN_OR_EQUAL)
    __case(LESS_THAN_OR_EQUAL_BCAST)
    __case(LESS_THAN_OR_EQUAL_C)
    __case(SHIFT_LEFT)
    __case(SHIFT_LEFT_BCAST)
    __case(SHIFT_LEFT_C)
    __case(SHIFT_RIGHT_ARITHMETIC)
    __case(SHIFT_RIGHT_ARITHMETIC_BCAST)
    __case(SHIFT_RIGHT_ARITHMETIC_C)
    __case(LESS_THAN)
    __case(LESS_THAN_BCAST)
    __case(LESS_THAN_C)
    __case(NOT_EQUAL)
    __case(NOT_EQUAL_BCAST)
    __case(NOT_EQUAL_C)
    __case(SIGNBIT)
    __case(FULL)
    __case(LOGICAL_NOT)
    __case(UPSAMPLING_BILINEAR)
    __case(UPSAMPLING_NEAREST)
    __case(UPSAMPLING_NEAREST_BACKWARD)
    __case(ARANGE)
    __case(SILU)
    __case(SIGMOID)
    __case(CLAMP)
    __case(LN_MM_FORWARD)
    __case(ERF)
    __case(ERFC)
    __case(POW)
    __case(POW_BCAST)
    __case(POWC)
    __case(CPOW)
    __case(ADD_LN_MM_FORWARD)
    __case(RECIPROCAL)
    __case(TRUNC)
    __case(LN_MM_BACKWARD)
    __case(ADD_LN_MM_BACKWARD)
    __case(TOPK)
    __case(NONZERO)
    __case(REPEAT)
    __case(ARGMAX)
    __case(ARGMIN)
    __case(MAX_DIM)
    __case(MIN_DIM)
    __case(HARDTANH)
    __case(HYPOT)
    __case(NEXTAFTER)
    __case(TRIU)
    __case(CBRT)
    __case(CONSTANT_PAD)
    __case(REFLECTION_PAD2D)
    __case(REPLICATION_PAD2D)
    __case(REPLICATION_PAD3D)
    __case(GATHER)
    __case(BADDBMM)
    __case(MSE_LOSS)
    __case(MSE_LOSS_BACKWARD)
    __case(SLICE_SCATTER)
    __case(InfCheckAndUnscale)
    __case(LLAMA_ATTENTION)
    __case(LLAMA_ATTENTION_BACKWARD)
    __case(LLAMA_MLP_FORWARD)
    __case(LLAMA_A16_MLP_FORWARD)
    __case(RMSNORM_FORWARD)
    __case(RMSNORM_BACKWARD)
    __case(BINARYOP)
    __case(BINARYOP_C)
    __case(BINARYOP_BCAST)
    __case(REAL)
    __case(CONJ)
    __case(DUMMY)
    __case(ENABLE_PMU)
    __case(DISABLE_PMU)
    __case(ADAM)
    __case(ADAM_BACKWARD)
    __case(DROPOUT)
    __case(OP_NUM)

#undef __case
    default:
      return "UNKNOWN";
  }
}

OpTimer * OpTimer::instance_ = nullptr;

OpTimer & OpTimer::Clear()
{
  mutex_.lock();
  for ( auto i = 0; i < OP_NUM; ++i )
  {
    elapsed_time_us_[i] = 0;
    count_[i] = 0;
  }
  is_paused_ = false;
  is_start_  = true;
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
    ++count_[type];
    if (is_start_)
    {
      std::cout << std::setw ( 42 ) << OpTypeStr(type) << " Elapsed: " << std::setw ( 12 ) << time_us << "us" << "\n";
    }
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
      std::cout << std::setw ( 42 ) << OpTypeStr(i) << ": " << std::setw ( 12 ) << (elapsed_time_us_[i] / count_[i]) << "us avg, " << std::setw ( 12 ) << elapsed_time_us_[i] << "us total, ";
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
