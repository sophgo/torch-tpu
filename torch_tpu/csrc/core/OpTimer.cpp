#include "OpTimer.h"
namespace tpu {

OpTimer * OpTimer::instance_         = nullptr;

OpTimer & OpTimer::Clear()
{
  mutex_.lock();
  func_time_map_.clear();
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

OpTimer & OpTimer::AddTime ( const char* func_name, unsigned long time_us )
{
  mutex_.lock();
  if ( is_paused_ == false )
  {
    // update elpased_time
    auto it = func_time_map_.find(func_name);
    if (it != func_time_map_.end())
    {
        auto & [total_time, count] = it->second;
        total_time += time_us;
        count++;
    }
    else
    {
        func_time_map_.emplace(func_name, std::make_pair(time_us, 1));
    }

    if (is_start_)
    {
      std::cout << std::setw ( 42 ) << func_name << " Elapsed: " << std::setw ( 12 ) << time_us << "us" << "\n";
    }
  }
  mutex_.unlock();
  return *this;
}

void OpTimer::Dump() const
{
  unsigned long long ElapsedAll = 0;
  for (const auto& [func_name, stats] : func_time_map_)
  {
      const auto& [total_time, call_count] = stats;
      ElapsedAll += total_time;
  }
  for (const auto& [func_name, stats] : func_time_map_)
  {
      const auto& [total_time, call_count] = stats;
      std::cout << std::setw ( 42 ) << func_name << ": " << std::setw ( 12 ) << (total_time / call_count) << "us avg, " << std::setw ( 12 ) << total_time << "us total, ";
      std::cout << std::setw ( 8 ) << std::setprecision ( 3 ) << total_time * 100. / ElapsedAll << "%" << std::endl;
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
} //namespace tpu