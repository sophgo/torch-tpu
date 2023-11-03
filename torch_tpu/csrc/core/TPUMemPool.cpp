// #include <map>
// #include <vector>
// #include <mutex>

// #include "config.h"
// namespace tpu
// {
// typedef unsigned long long pool_addr_t;
// typedef unsigned long long pool_size_t;
// typedef std::pair<pool_addr_t, pool_size_t> pool_pair_t;
// typedef std::map<pool_addr_t, pool_size_t> pool_map_t;

// struct pool_struct {
//     std::mutex  mem_pool_lock;
//     int         num_slots_in_use;
//     pool_size_t mem_in_use;
//     std::vector<pool_pair_t> slot_avail;
//     pool_map_t slot_in_use;
// };

// class MemPool {
// public:
//     MemPool(unsigned long long total_size = 0);
//     ~MemPool();
//     pool_addr_t MemPool_alloc(pool_size_t size);
//     void MemPool_free(pool_addr_t addr);

// private:
//    pool_size_t        _total_size;
//    struct pool_struct _mem_pool_list[MAX_POOL_COUNT];
//    int                _mem_pool_count;
// };

// }// namespace tpu;