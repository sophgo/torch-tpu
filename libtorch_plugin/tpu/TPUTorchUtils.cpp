#include "TPUTorchUtils.h"

namespace tpu
{

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
  mutex_.unlock();
  return *this;
}

OpTimer & OpTimer::Pause()
{
  mutex_.lock();
  is_paused_ = true;
  mutex_.unlock();
  return *this;
}

OpTimer & OpTimer::AddTime ( OpType type, unsigned long time_us )
{
  mutex_.lock();
  if ( is_paused_ == false )
  {
    elapsed_time_us_[type] += time_us;
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


TensorWatcher * TensorWatcher::instance_ = nullptr;

void TensorWatcher::AddTensor ( const at::Tensor & Tensor )
{
  tensors_.push_back ( Tensor );
  tensors_cpu_.push_back ( TENSOR_TO_CPU ( Tensor ) );
}

bool TensorWatcher::Watch() const
{
  for ( auto I = 0; I < tensors_.size(); ++I )
  {
    if ( tensors_[I].defined() )
    {
      auto tensor_cpu = TENSOR_TO_CPU ( tensors_[I] );
      auto ptr_saved = ( unsigned char * ) tensors_cpu_[I].data_ptr();
      auto ptr_current = ( unsigned char * ) tensor_cpu.data_ptr();
      for ( auto i = 0; i < tensors_[I].nbytes(); ++i )
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

} // namespace tpu
