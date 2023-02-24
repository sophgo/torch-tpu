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

OpTimer & OpTimer::AddTime ( OpType type, unsigned long time_us )
{
  mutex_.lock();
  elapsed_time_us_[type] += time_us;
  mutex_.unlock();
  return *this;
}

void OpTimer::Dump() const
{
  for ( auto i = 0; i < OP_NUM; ++i )
  {
    std::cout << OpTypeStr[i] << ": " << elapsed_time_us_[i] << "us" << std::endl;
  }
}

OpTimer & OpTimer::Instance()
{
  if ( instance_ == nullptr )
  {
    instance_ = new OpTimer;
  }
  return *instance_;
}

} // namespace tpu
