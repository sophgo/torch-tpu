#pragma once

#ifdef TPU_OP_TIMING
#define TIMING_START auto timer = tpu::Timer().Start();
#define TIMING_END  \
  tpu::OpTimer::Instance().AddTime(__func__, timer.ElapsedUS());
#else
#define TIMING_START
#define TIMING_END
#endif

namespace tpu{
struct OpTimer
{
  OpTimer & Clear();
  OpTimer & Start();
  OpTimer & Pause();
  OpTimer & AddTime ( const char* func_name, unsigned long time_us );
  void Dump() const;
  static OpTimer & Instance();
private:
  OpTimer() {}

  std::map<std::string, std::pair<unsigned long, unsigned>> func_time_map_;
  bool is_paused_ = false;
  bool is_start_ = false;
  std::mutex mutex_;
  static OpTimer * instance_;
};
struct Timer
{
  Timer & Start()
  {
    gettimeofday ( &timer, NULL );
    return *this;
  }
  unsigned long ElapsedUS() const
  {
    struct timeval end;
    gettimeofday ( &end, NULL );
    return ( end.tv_sec - timer.tv_sec ) * 1000000UL + ( end.tv_usec - timer.tv_usec );
  }
  unsigned long ElapsedMS() const
  {
    return ElapsedUS() / 1000;
  }
private:
  struct timeval timer;
};

} // namespace tpu