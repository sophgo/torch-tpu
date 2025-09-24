#include "TPUCachingAllocator.h"
#include "TPUFunction.h"

//#include <c10/util/Gauge.h>
#include <c10/util/Logging.h>
// #include <c10/util/ScopeExit.h>
// #include <c10/util/UniqueVoidPtr.h>
// #include <c10/util/env.h>
// #include <c10/util/error.h>
// #include <c10/util/flat_hash_map.h>
// #include <c10/util/hash.h>
// #include <c10/util/llvmMathExtras.h>
// #include <c10/util/static_tracepoint.h>
#include <c10/util/Exception.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <new>
#include <regex>
#include <set>
#include <stack>
#include <thread>
#include <utility>
#include <vector>
#include <iterator>


namespace c10_tpu {

namespace TPUCachingAllocator {

// Included here as this is externally used in TPUAllocatorConfig
const size_t kLargeBuffer =
    20971520; // "large" allocations may be packed in 20 MiB blocks
constexpr size_t kMinBlockSize =
    512; // all sizes are rounded to at least 512 bytes
constexpr size_t kSmallSize = 1048576; // largest "small" allocation is 1 MiB
constexpr size_t kSmallBuffer =
    2097152; // "small" allocations are packed in 2 MiB blocks
constexpr size_t kMinLargeAlloc =
    10485760; // allocations between 1 and 10 MiB may use kLargeBuffer
constexpr size_t kRoundLarge = 2097152; // round up large allocations to 2 MiB

constexpr size_t kRoundUpPowerOfTwoIntervals = 16;

TPUAllocatorConfig::TPUAllocatorConfig()
    : m_max_split_size(std::numeric_limits<size_t>::max()),
      m_max_non_split_rounding_size(kLargeBuffer),
      m_release_lock_on_tpumalloc(false) {
  m_roundup_power2_divisions.assign(kRoundUpPowerOfTwoIntervals, 0);
}

size_t TPUAllocatorConfig::roundup_power2_divisions(size_t size) {
  size_t log_size = (63 - llvm::countLeadingZeros(size));

  // Our intervals start at 1MB and end at 64GB
  const size_t interval_start =
      63 - llvm::countLeadingZeros(static_cast<size_t>(1048576));
  const size_t interval_end =
      63 - llvm::countLeadingZeros(static_cast<size_t>(68719476736));
  TORCH_CHECK(
      (interval_end - interval_start == kRoundUpPowerOfTwoIntervals),
      "kRoundUpPowerOfTwoIntervals mismatch");

  int index = static_cast<int>(log_size) - static_cast<int>(interval_start);

  index = std::max(0, index);
  index = std::min(index, static_cast<int>(kRoundUpPowerOfTwoIntervals) - 1);
  return instance().m_roundup_power2_divisions[index];
}

void TPUAllocatorConfig::lexArgs(
    const std::string& env,
    std::vector<std::string>& config) {
  std::vector<char> buf;

  for (char ch : env) {
    if (ch == ',' || ch == ':' || ch == '[' || ch == ']') {
      if (!buf.empty()) {
        config.emplace_back(buf.begin(), buf.end());
        buf.clear();
      }
      config.emplace_back(1, ch);
    } else if (ch != ' ') {
      buf.emplace_back(ch);
    }
  }
  if (!buf.empty()) {
    config.emplace_back(buf.begin(), buf.end());
  }
}

void TPUAllocatorConfig::consumeToken(
    const std::vector<std::string>& config,
    size_t i,
    const char c) {
  TORCH_CHECK(
      i < config.size() && config[i] == std::string(1, c),
      "Error parsing CachingAllocator settings, expected ",
      c,
      "");
}

size_t TPUAllocatorConfig::parseMaxSplitSize(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  constexpr int mb = 1024 * 1024;
  if (++i < config.size()) {
    size_t val1 = stoi(config[i]);
    TORCH_CHECK(
        val1 > kLargeBuffer / mb,
        "CachingAllocator option max_split_size_mb too small, must be > ",
        kLargeBuffer / mb,
        "");
    val1 = std::max(val1, kLargeBuffer / mb);
    val1 = std::min(val1, (std::numeric_limits<size_t>::max() / mb));
    m_max_split_size = val1 * 1024 * 1024;
  } else {
    TORCH_CHECK(false, "Error, expecting max_split_size_mb value", "");
  }
  return i;
}

size_t TPUAllocatorConfig::parseMaxNonSplitRoundingSize(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  constexpr int mb = 1024 * 1024;
  if (++i < config.size()) {
    size_t val1 = stoi(config[i]);
    TORCH_CHECK(
        val1 > kLargeBuffer / mb,
        "CachingAllocator option max_non_split_rounding_mb too small, must be > ",
        kLargeBuffer / mb,
        "");
    val1 = std::max(val1, kLargeBuffer / mb);
    val1 = std::min(val1, (std::numeric_limits<size_t>::max() / mb));
    m_max_non_split_rounding_size = val1 * 1024 * 1024;
  } else {
    TORCH_CHECK(false, "Error, expecting max_non_split_rounding_mb value", "");
  }
  return i;
}

size_t TPUAllocatorConfig::parseRoundUpPower2Divisions(
    const std::vector<std::string>& config,
    size_t i) {
  consumeToken(config, ++i, ':');
  bool first_value = true;

  if (++i < config.size()) {
    if (std::string_view(config[i]) == "[") {
      size_t last_index = 0;
      // NOLINTNEXTLINE(bugprone-inc-dec-in-conditions)
      while (++i < config.size() && std::string_view(config[i]) != "]") {
        const std::string& val1 = config[i];
        size_t val2 = 0;

        consumeToken(config, ++i, ':');
        if (++i < config.size()) {
          val2 = stoi(config[i]);
        } else {
          TORCH_CHECK(
              false, "Error parsing roundup_power2_divisions value", "");
        }
        TORCH_CHECK(
            val2 == 0 || llvm::isPowerOf2_64(val2),
            "For roundups, the divisons has to be power of 2 or 0 to disable roundup ",
            "");

        if (std::string_view(val1) == ">") {
          std::fill(
              std::next(
                  m_roundup_power2_divisions.begin(),
                  static_cast<std::vector<unsigned long>::difference_type>(
                      last_index)),
              m_roundup_power2_divisions.end(),
              val2);
        } else {
          size_t val1_long = stoul(val1);
          TORCH_CHECK(
              llvm::isPowerOf2_64(val1_long),
              "For roundups, the intervals have to be power of 2 ",
              "");

          size_t index = 63 - llvm::countLeadingZeros(val1_long);
          index = std::max((size_t)0, index);
          index = std::min(index, m_roundup_power2_divisions.size() - 1);

          if (first_value) {
            std::fill(
                m_roundup_power2_divisions.begin(),
                std::next(
                    m_roundup_power2_divisions.begin(),
                    static_cast<std::vector<unsigned long>::difference_type>(
                        index)),
                val2);
            first_value = false;
          }
          if (index < m_roundup_power2_divisions.size()) {
            m_roundup_power2_divisions[index] = val2;
          }
          last_index = index;
        }

        if (std::string_view(config[i + 1]) != "]") {
          consumeToken(config, ++i, ',');
        }
      }
    } else { // Keep this for backwards compatibility
      size_t val1 = stoi(config[i]);
      TORCH_CHECK(
          llvm::isPowerOf2_64(val1),
          "For roundups, the divisons has to be power of 2 ",
          "");
      std::fill(
          m_roundup_power2_divisions.begin(),
          m_roundup_power2_divisions.end(),
          val1);
    }
  } else {
    TORCH_CHECK(false, "Error, expecting roundup_power2_divisions value", "");
  }
  return i;
}

void TPUAllocatorConfig::parseArgs(const std::optional<std::string>& env) {
  // If empty, set the default values
  m_max_split_size = std::numeric_limits<size_t>::max();
  m_roundup_power2_divisions.assign(kRoundUpPowerOfTwoIntervals, 0);

  if (!env.has_value()) {
    return;
  }

  std::vector<std::string> config;
  lexArgs(env.value(), config);

  for (size_t i = 0; i < config.size(); i++) {
    std::string_view config_item_view(config[i]);
    if (config_item_view == "max_split_size_mb") {
      i = parseMaxSplitSize(config, i);
    } else if (config_item_view == "max_non_split_rounding_mb") {
      i = parseMaxNonSplitRoundingSize(config, i);
    } else if (config_item_view == "roundup_power2_divisions") {
      i = parseRoundUpPower2Divisions(config, i);
    } else {
      TORCH_CHECK(
          false, "Unrecognized CachingAllocator option: ", config_item_view);
    }

    if (i + 1 < config.size()) {
      consumeToken(config, ++i, ',');
    }
  }
}

// General caching allocator utilities
void setAllocatorSettings(const std::string& env) {
  TPUCachingAllocator::TPUAllocatorConfig::instance().parseArgs(env.c_str());
}


//namespace {

using stream_set = ska::flat_hash_set<TPUStream>;

struct PrivatePool;
typedef bool (*Comparison)(const Block*, const Block*);
static bool BlockComparatorSize(const Block* a, const Block* b);
struct BlockPool {
  BlockPool(bool small, PrivatePool* private_pool = nullptr)
      : blocks(BlockComparatorSize),
        is_small(small),
        owner_PrivatePool(private_pool) {}

  // Do not insert a Block to blocks directly; use insert_into_blocks(),
  // instead.
  std::set<Block*, Comparison> blocks;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const bool is_small;
  PrivatePool* owner_PrivatePool;

  // Add a Block into blocks set with updating gc counter.
  std::pair<std::set<Block*, Comparison>::iterator, bool> insert_into_blocks(
      Block* block);
};

struct Block {
  c10::DeviceIndex device; // tpu
  tpuStream_t stream; // allocation stream
  stream_set stream_uses; // streams on which the block was used
  size_t size; // block size in bytes
  size_t requested_size; // memory originally requested
  BlockPool* pool{nullptr}; // owning memory pool
  void* ptr{nullptr}; // memory address
  bool allocated{false}; // in-use flag
  Block* prev{nullptr}; // prev block if split from a larger allocation
  Block* next{nullptr}; // next block if split from a larger allocation
  int event_count{0}; // number of outstanding TPU events

  Block(
      c10::DeviceIndex device,
      tpuStream_t stream,
      size_t size,
      BlockPool* pool,
      void* ptr)
      : device(device),
        stream(stream),
        size(size),
        requested_size(0),
        pool(pool),
        ptr(ptr) {}

  // constructor for search key
  Block(c10::DeviceIndex device, tpuStream_t stream, size_t size)
      : device(device), stream(stream), size(size), requested_size(0) {}

  bool is_split() const {
    return (prev != nullptr) || (next != nullptr);
  }
};

std::pair<std::set<Block*, Comparison>::iterator, bool> BlockPool::
    insert_into_blocks(Block* block) {
  return blocks.insert(block);
}

static bool BlockComparatorSize(const Block* a, const Block* b) {
  if (a->stream != b->stream) {
    return (uintptr_t)a->stream < (uintptr_t)b->stream;
  }
  if (a->size != b->size) {
    return a->size < b->size;
  }
  return (uintptr_t)a->ptr < (uintptr_t)b->ptr;
}

struct AllocParams {
  AllocParams(
      c10::DeviceIndex device,
      size_t size,
      tpuStream_t stream,
      BlockPool* pool,
      size_t alloc_size)
      : search_key(device, stream, size), pool(pool), alloc_size(alloc_size) {}

  c10::DeviceIndex device() const {
    return search_key.device;
  }
  tpuStream_t stream() const {
    return search_key.stream;
  }
  size_t size() const {
    return search_key.size;
  }

  Block search_key;
  BlockPool* pool;
  size_t alloc_size;
  Block* block{nullptr};
  tpuError_t err{tpuSuccess};
};

// Note: tpuEventCreate when concurrently invoked from multiple threads can be
// very expensive (at least on certain device/driver combinations). Thus, we a)
// serialize event creation at a per-device level, and b) pool the events to
// avoid constantly calling tpuEventCreate/tpuEventDestroy. This results in
// significant improvements in multithreaded workloads with high allocation
// rates.
class EventPool {
 public:
  using Event = std::unique_ptr<tpuEvent_t, std::function<void(tpuEvent_t*)>>;
  // TODO: Explicit device count
  EventPool() : pools_(c10_tpu::device_count()) {}

  Event get(c10::DeviceIndex device) {
    TORCH_INTERNAL_ASSERT(0 <= device);
    TORCH_INTERNAL_ASSERT(device < static_cast<int>(pools_.size()));
    auto& pool = pools_[device];
    auto destructor = [&pool](tpuEvent_t* event) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.push_back(std::unique_ptr<tpuEvent_t>(event));
    };

    // Try to acquire an event from the per-device pool.
    {
      std::lock_guard<std::mutex> g(pool.mutex_);
      if (!pool.event_pool_.empty()) {
        auto* event = pool.event_pool_.back().release();
        pool.event_pool_.pop_back();
        return Event(event, destructor);
      }
    }
    // otherwise, allocate a new event that will be returned to the pool on
    // destruction.
    auto new_ptr = std::make_unique<tpuEvent_t>();
    C10_TPU_CHECK(
        tpuEventCreate(new_ptr.get()));

    return Event(new_ptr.release(), destructor);
  }

  void empty_cache() {
    for (auto& pool : pools_) {
      std::lock_guard<std::mutex> g(pool.mutex_);
      pool.event_pool_.clear();
    }
  }

 private:
  struct PerDevicePool {
    alignas(64) std::mutex mutex_;
    std::vector<std::unique_ptr<tpuEvent_t>> event_pool_;
  };
  std::vector<PerDevicePool> pools_;
};

// TPU graphs helper
struct PrivatePool {
  PrivatePool(MempoolId_t id, TPUAllocator* allocator = nullptr)
      : id(std::move(id)),
        allocator_(allocator),
        large_blocks(/*small=*/false, this),
        small_blocks(/*small=*/true, this) {}
  PrivatePool(const PrivatePool&) = delete;
  PrivatePool(PrivatePool&&) = delete;
  PrivatePool& operator=(const PrivatePool&) = delete;
  PrivatePool& operator=(PrivatePool&&) = delete;
  ~PrivatePool() = default;

  MempoolId_t id{0, 0};
  // Number of live graphs using this pool
  int use_count{1};
  // Number of unfreed tpuMallocs made for this pool. When use_count and
  // tpuMalloc_count drop to zero, we can delete this PrivatePool from
  // graph_pools.
  int tpuMalloc_count{0};
  // Instead of maintaining private BlockPools here, I could stuff all blocks
  // (private or no) into the top-level large_blocks and small_blocks, and
  // distinguish private blocks by adding a "pool id" check above the stream
  // check in BlockComparator. BlockComparator is performance- critical though,
  // I'd rather not add more logic to it.
  TPUAllocator* allocator_;
  BlockPool large_blocks;
  BlockPool small_blocks;

 public:
 TPUAllocator* allocator() {
    return allocator_;
  }
};

struct MempoolIdHash {
  std::size_t operator()(const MempoolId_t& mempool_id) const noexcept {
    return mempool_id.first != 0 ? mempool_id.first : mempool_id.second;
  }
};

tpuError_t allocPrimitive(void** ptr, size_t size, AllocParams& p) {
  if (p.pool->owner_PrivatePool && p.pool->owner_PrivatePool->allocator()) {
    *ptr = p.pool->owner_PrivatePool->allocator()->raw_alloc(size);
    return *ptr ? tpuSuccess : tpuErrorMemoryAllocation;
  } else {
    return tpuMalloc(ptr, size);
  }
}

tpuError_t tpuMallocMaybeCapturing(void** ptr, size_t size, AllocParams& p) {
  return allocPrimitive(ptr, size, p);
}

//} // anonymous namespace

class DeviceCachingAllocator {
 private:
  // lock around all operations
  mutable std::recursive_mutex mutex;

  // unallocated cached blocks larger than 1 MB
  BlockPool large_blocks;

  // unallocated cached blocks 1 MB or smaller
  BlockPool small_blocks;

  // allocated or in use by a stream. Holds all active allocations,
  // whether they came from graph_pools or one of the BlockPools above.
  ska::flat_hash_set<Block*> active_blocks;

  // captures_underway tracks if we are diverting some
  // allocations to a specific pool.
  // Most of the time it's empty
  std::vector<std::pair<MempoolId_t, std::function<bool(tpuStream_t)>>>
      captures_underway;

  // tracks which pools we can use as a last resort before ooming
  ska::flat_hash_set<MempoolId_t, MempoolIdHash> use_on_oom_pools;

  // See free() for this thing's purpose
  std::vector<Block*> needs_events_deferred_until_no_capture;
  // outstanding tpu events
  ska::flat_hash_map<
      TPUStream,
      std::deque<std::pair<EventPool::Event, Block*>>>
      tpu_events;

  // record used memory.
  size_t total_allocated_memory = 0;

  // Members specific to TPU graphs

  // Private pools for TPU graphs
  ska::flat_hash_map<MempoolId_t, std::unique_ptr<PrivatePool>, MempoolIdHash>
      graph_pools;
  // Pools no longer referenced by any graph. Their BlockPools are eligible for
  // free_blocks. Can't be a vector or deque because we might erase entries in
  // any order. Could be an std::list, but we don't care much, access and
  // insert/erase are rare.
  ska::flat_hash_map<MempoolId_t, PrivatePool*, MempoolIdHash>
      graph_pools_freeable;

  // mapping from block to a stream_set, containing streams on which the block
  // was used while tpugraph capturing
  std::unordered_map<Block*, stream_set> block_to_tpugraph_stream_uses;

 public:
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
  DeviceCachingAllocator()
      : large_blocks(/*small=*/false), small_blocks(/*small=*/true) {
  }

  bool checkPoolLiveAllocations(
      MempoolId_t mempool_id,
      const std::unordered_set<void*>& expected_live_allocations) {
    std::unique_lock<std::recursive_mutex> lock(mutex);

    PrivatePool* pool = nullptr;
    auto pool_it = graph_pools.find(mempool_id);
    TORCH_CHECK(pool_it != graph_pools.end(), "Could not find pool of id");
    pool = pool_it->second.get();

    TORCH_INTERNAL_ASSERT(pool != nullptr);

    size_t allocated_pool_blocks = 0;

    for (Block* b : active_blocks) {
      TORCH_INTERNAL_ASSERT(b != nullptr);
      TORCH_INTERNAL_ASSERT(b->pool != nullptr);
      if (b->allocated && b->pool->owner_PrivatePool == pool) {
        if (!expected_live_allocations.count(b->ptr)) {
          return false;
        }

        allocated_pool_blocks += 1;
      }
    }

    return allocated_pool_blocks == expected_live_allocations.size();
  }

  // All public methods (except the above) acquire the allocator mutex.
  // Thus, do not call a public method from another public method.

  Block* malloc(
      c10::DeviceIndex device,
      size_t orig_size,
      tpuStream_t stream) {

    std::unique_lock<std::recursive_mutex> lock(mutex);

    if (C10_LIKELY(captures_underway.empty())) {
      // Processes end-of-life events for outstanding allocations used on
      // multiple streams (checks if their TPU-side uses are complete and
      // recycles their memory if so)
      //
      // Q. Why skip process_events if a capture might be underway?
      // A. process_events involves tpuEventQueries, illegal during TPU graph
      //    capture.
      //    Dumb simple solution: defer reclaiming these allocations until after
      //    capture. Cross-stream memory use is uncommon, so the deferral's
      //    effect on memory use during capture should be small.
      process_events();
    }
    size_t size = round_size(orig_size);
    auto& pool = get_pool(size, stream);
    const size_t alloc_size = get_allocation_size(size);
    AllocParams params(device, size, stream, &pool, alloc_size);

    // First, try to get a block from the existing pool.
    bool block_found =
        // Search pool
        get_free_block(params);
        // Trigger callbacks and retry search
        // || (trigger_free_memory_callbacks(params) && get_free_block(params));

    // Can't reuse an existing block; try to get a new one.
    if (!block_found) {
      // Attempt allocate
      // WARNING: alloc_block may release the allocator lock when calling
      // tpuMalloc. So far this function has not modified allocator state, but
      // keep in mind that any observed allocator state may change across calls
      // to alloc_block since it may release the lock.
      block_found = alloc_block(params, false, lock)
          // Free enough available cached blocks to satisfy alloc and retry
          // alloc.
          || (release_available_cached_blocks(params) &&
              alloc_block(params, false, lock))
          // Free all non-split cached blocks and retry alloc.
          || (C10_LIKELY(captures_underway.empty()) &&
              release_cached_blocks({0, 0}) &&
              alloc_block(params, true, lock));
    }

    // we are about to oom, try to use existing mempools as a last resort
    if (!block_found && params.err == tpuErrorMemoryAllocation) {
      // if already trying to use a mempool, then just oom
      bool active_pool = params.pool->owner_PrivatePool;
      if (!active_pool) {
        for (MempoolId_t mempool_id : use_on_oom_pools) {
          auto tid = std::this_thread::get_id();
          auto filter = [tid](tpuStream_t) {
            return std::this_thread::get_id() == tid;
          };
          beginAllocateToPool(mempool_id, filter);
          auto& mempool = get_pool(size, stream);
          AllocParams mempool_params(
              device, size, stream, &mempool, alloc_size);
          block_found = get_free_block(mempool_params);
          endAllocateToPool(mempool_id);
          releasePool(mempool_id);
          if (block_found) {
            params = mempool_params;
            break;
          }
        }
      }
    }

    if (!block_found) {
      // Make sure we do not have the device lock before calling our
      // observers which might need hold the GIL
      // It is safe to release at this point because will no longer
      // be reading any allocator state.

      lock.unlock();
    }

    bool split_remainder = should_split(params.block, params.size());
    return alloc_found_block(
        params, orig_size, split_remainder);
  }

  Block* alloc_found_block(
      const AllocParams& params,
      size_t orig_size,
      bool split_remainder) {
    auto size = params.size();
    auto device = params.device();
    auto pool = params.pool;
    auto stream = params.stream();

    TORCH_INTERNAL_ASSERT(
        params.err == tpuSuccess && params.block != nullptr &&
        params.block->ptr != nullptr);
    Block* block = params.block;
    Block* remaining = nullptr;

    if (split_remainder) {
      remaining = block;

      block = new Block(device, stream, size, pool, block->ptr);
      block->prev = remaining->prev;
      if (block->prev) {
        block->prev->next = block;
      }
      block->next = remaining;

      remaining->prev = block;
      remaining->ptr = static_cast<char*>(remaining->ptr) + size;
      remaining->size -= size;
      // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
      bool inserted = pool->insert_into_blocks(remaining).second;
      TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);
    }

    block->allocated = true;
    block->requested_size = orig_size;

    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
    bool inserted = active_blocks.insert(block).second;
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(inserted);

    return block;
  }

  void free(Block* block) {
    std::lock_guard<std::recursive_mutex> lock(mutex);

    block->allocated = false;

    if (!block->stream_uses.empty()) {
      if (C10_UNLIKELY(!captures_underway.empty())) {
        // It's forbidden to tpuEventQuery an event recorded during TPU graph
        // capture. We conservatively defer recording end-of-life events until
        // the next call to process_events() (which won't happen until no
        // captures are underway)
        needs_events_deferred_until_no_capture.push_back(block);
      } else {
        insert_events(block);
      }
    } else {
      free_block(block);
    }
  }

  void recordStream(Block* block, TPUStream stream) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    if (stream.stream() == block->stream) {
      // ignore uses on the allocation stream, since those don't require any
      // special synchronization
      return;
    }
    block->stream_uses.insert(stream);
    if (C10_UNLIKELY(!captures_underway.empty())) {
      block_to_tpugraph_stream_uses[block].insert(stream);
    }
  }

  /** returns cached blocks to the system allocator **/
  void emptyCache(MempoolId_t mempool_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    release_cached_blocks(mempool_id);
  }

  /** Retrieves size of largest unused block held by the memory cache **/
  void cacheInfo(size_t* largest) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    cache_info_aux(large_blocks, largest);
    cache_info_aux(small_blocks, largest);
    for (const auto& gp : graph_pools) {
      cache_info_aux(gp.second->large_blocks, largest);
      cache_info_aux(gp.second->small_blocks, largest);
    }
  }

  // This function takes the size and number of divisions argument and rounds
  // up the size argument for the nearest power-of-2 division.
  // For example, if we need to round-up 1200 and number of divisions is 4,
  // the size 1200 lies between 1024 and 2048 and if we do 4 divisions between
  // them, the values are 1024, 1280, 1536, and 1792. So the function will
  // return 1280 as the nearest ceiling of power-2 divison.
  static size_t roundup_power2_next_division(size_t size, size_t divisions) {
    if (llvm::isPowerOf2_64(size)) {
      return size;
    }

    TORCH_CHECK(divisions >= 2, "Only 2 or more divisions are supported");

    // divide the space between these 2's power into equal divisions
    // If division is zero, return the power-of-2 ceiling.
    size_t power2_floor = llvm::PowerOf2Floor(size);
    size_t power2_divison =
        power2_floor >> (63 - llvm::countLeadingZeros(divisions));
    if (C10_UNLIKELY(power2_divison == 0)) {
      return (power2_floor << 1);
    }
    size_t round_size_floor = size & (~(power2_divison - 1));
    return (round_size_floor == size) ? size
                                      : round_size_floor + power2_divison;
  }

  static size_t round_size(size_t size) {
    if (size < kMinBlockSize) {
      return kMinBlockSize;
    } else {
      auto divisions = TPUAllocatorConfig::roundup_power2_divisions(size);
      if (divisions > 1 && size > (kMinBlockSize * divisions)) {
        return roundup_power2_next_division(size, divisions);
      } else {
        return kMinBlockSize * ((size + kMinBlockSize - 1) / kMinBlockSize);
      }
    }
  }

  void createOrIncrefPool(MempoolId_t mempool_id, TPUAllocator* allocator) {
    // Create a PrivatePool object if it does not exist yet
    // and increment its use_count
    std::lock_guard<std::recursive_mutex> lock(mutex);
    create_or_incref_pool(mempool_id, allocator);
  }

  void setUseOnOOM(MempoolId_t mempool_id) {
    // Choose if this pool should be used as a last resort before ooming
    std::lock_guard<std::recursive_mutex> lock(mutex);
    use_on_oom_pools.insert(mempool_id);
  }

  // See Note [Interaction with TPU graph capture]

  // Called by TPUGraph::capture_begin
  void beginAllocateToPool(
      MempoolId_t mempool_id,
      std::function<bool(tpuStream_t)> filter) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    create_or_incref_pool(mempool_id);
    for (auto it2 = captures_underway.begin(); it2 != captures_underway.end();
         ++it2) {
      TORCH_CHECK(
          it2->first != mempool_id,
          "beginAllocateToPool: already recording to mempool_id");
    }
    captures_underway.emplace_back(mempool_id, std::move(filter));
  }

  // Called by TPUGraph::capture_end
  void endAllocateToPool(MempoolId_t mempool_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    for (auto it = captures_underway.begin(); it != captures_underway.end();
         ++it) {
      if (it->first == mempool_id) {
        captures_underway.erase(it);
        return;
      }
    }
    TORCH_CHECK(
        false, "endAllocatePool: not currently recording to mempool_id");
  }

  // Called by TPUGraph::reset and MemPool::~MemPool()
  void releasePool(MempoolId_t mempool_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    // The instantiated tpuGraphExec_t has been destroyed. We can't blindly
    // delete and tpuFree the mempool its capture used, because
    //  1. other graph(s) might share the same pool
    //  2. the user might still hold references to output tensors allocated
    //  during capture.
    // To handle 1 and 2, we track the number of graphs using this particular
    // mempool. When the count reaches 0, we tell free_cached_blocks it may now
    // tpuFree blocks from this graph's pool when it discovers they're unused
    // (unsplit).
    auto pp = get_private_pool(mempool_id);
    auto uc = --(pp->use_count);
    TORCH_INTERNAL_ASSERT(uc >= 0);
    if (uc == 0) {
      // Allows free_cached_blocks to begin tpuFreeing this pool's memory,
      // and makes sure this pool wasn't somehow made freeable already.
      // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
      bool inserted = graph_pools_freeable.insert({mempool_id, pp}).second;
      TORCH_INTERNAL_ASSERT(inserted);
    }
  }

  int getPoolUseCount(MempoolId_t mempool_id) {
    std::lock_guard<std::recursive_mutex> lock(mutex);
    auto pp = get_private_pool(mempool_id);
    return pp->use_count;
  }

 private:
  // All private methods do not acquire the allocator mutex.

  std::vector<Block*> get_all_blocks() const {
    std::vector<Block*> blocks;
    blocks.insert(
        blocks.end(), small_blocks.blocks.begin(), small_blocks.blocks.end());
    blocks.insert(
        blocks.end(), large_blocks.blocks.begin(), large_blocks.blocks.end());
    for (const auto& gp : graph_pools) {
      blocks.insert(
          blocks.end(),
          gp.second->small_blocks.blocks.begin(),
          gp.second->small_blocks.blocks.end());
      blocks.insert(
          blocks.end(),
          gp.second->large_blocks.blocks.begin(),
          gp.second->large_blocks.blocks.end());
    }
    blocks.insert(blocks.end(), active_blocks.begin(), active_blocks.end());
    return blocks;
  }

  std::vector<Block*> get_private_pool_head_blocks(PrivatePool* pool) const {
    std::vector<Block*> blocks;
    for (Block* b : active_blocks) {
      if ((b->pool == &pool->small_blocks || b->pool == &pool->large_blocks) &&
          b->prev == nullptr) {
        blocks.push_back(b);
      }
    }

    for (Block* b : pool->small_blocks.blocks) {
      if (b->prev == nullptr) {
        blocks.push_back(b);
      }
    }
    for (Block* b : pool->large_blocks.blocks) {
      if (b->prev == nullptr) {
        blocks.push_back(b);
      }
    }

    return blocks;
  }

  void create_or_incref_pool(
      MempoolId_t mempool_id,
      TPUAllocator* allocator = nullptr) {
    auto it = graph_pools.find(mempool_id);
    if (it == graph_pools.end()) {
      // mempool_id does not reference an existing pool.
      // Make a new pool for TPUGraph capture
      // usage. use_count is initially 1, which means the pool is
      // being used since somebody called createOrIncrefPool.
      graph_pools.emplace(
          mempool_id, std::make_unique<PrivatePool>(mempool_id, allocator));
    } else {
      // mempool_id references an existing pool, which the current TPUGraph
      // capture will share. Check this pool is live (at least one other capture already
      // references it). Increment it to establish the usage.
      TORCH_INTERNAL_ASSERT(it->second->use_count > 0);
      TORCH_INTERNAL_ASSERT(allocator == nullptr);
      it->second->use_count++;
    }
  }

  PrivatePool* get_private_pool(MempoolId_t mempool_id) {
    auto it = graph_pools.find(mempool_id);
    TORCH_INTERNAL_ASSERT(it != graph_pools.end());
    return it->second.get();
  }

  /** moves a block into a pool of cached free blocks */
  void free_block(Block* block) {
    TORCH_INTERNAL_ASSERT(
        !block->allocated && block->event_count == 0 &&
        block->stream_uses.empty());

    auto& pool = *block->pool;

    const std::array<Block*, 2> merge_candidates = {block->prev, block->next};
    for (Block* merge_candidate : merge_candidates) {
      try_merge_blocks(block, merge_candidate, pool);
    }

    active_blocks.erase(block);
    // Makes sure the Block* isn't already present in the pool we're freeing it
    // back into.
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
    bool inserted = pool.insert_into_blocks(block).second;
    TORCH_INTERNAL_ASSERT(inserted);
  }

  /** combine previously split blocks. returns the size of the subsumed block,
   * or 0 on failure. */
  size_t try_merge_blocks(Block* dst, Block* src, BlockPool& pool) {
    if (!src || src->allocated || src->event_count > 0 ||
        !src->stream_uses.empty()) {
      return 0;
    }

    AT_ASSERT(dst->is_split() && src->is_split());

    if (dst->prev == src) { // [src dst]
      dst->ptr = src->ptr;
      dst->prev = src->prev;
      if (dst->prev) {
        dst->prev->next = dst;
      }
    } else { // [dest src]
      dst->next = src->next;
      if (dst->next) {
        dst->next->prev = dst;
      }
    }
    const size_t subsumed_size = src->size;
    dst->size += subsumed_size;
    // NOLINTNEXTLINE(clang-analyzer-deadcode.DeadStores)
    auto erased = pool.blocks.erase(src);
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(erased == 1);
    delete src;

    return subsumed_size;
  }

  BlockPool& get_pool(size_t size, tpuStream_t stream) {
    // captures_underway is a conservative guess that the current stream may be
    // capturing. It's only non-empty if some thread has begun and not yet ended
    // a capture, so it's usually 0, and we can short-circuit
    // tpuStreamCaptureStatus (which does a TLS lookup).
    if (C10_UNLIKELY(!captures_underway.empty())) {
      for (auto& entry : captures_underway) {
        if (entry.second(stream)) {
          auto it1 = graph_pools.find(entry.first);
          TORCH_INTERNAL_ASSERT(it1 != graph_pools.end());
          if (size <= kSmallSize) {
            return it1->second->small_blocks;
          } else {
            return it1->second->large_blocks;
          }
        }
      }
    }
    if (size <= kSmallSize) {
      return small_blocks;
    } else {
      return large_blocks;
    }
  }

  bool should_split(const Block* block, size_t size) {
    size_t remaining = block->size - size;
    if (block->pool->is_small) {
      return remaining >= kMinBlockSize;
    } else {
      return (size < TPUAllocatorConfig::max_split_size()) &&
          (remaining > kSmallSize);
    }
  }

  static size_t get_allocation_size(size_t size) {
    if (size <= kSmallSize) {
      return kSmallBuffer;
    } else if (size < kMinLargeAlloc) {
      return kLargeBuffer;
    } else {
      return kRoundLarge * ((size + kRoundLarge - 1) / kRoundLarge);
    }
  }

  bool get_free_block(AllocParams& p) {
    BlockPool& pool = *p.pool;

    auto it = pool.blocks.lower_bound(&p.search_key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream())
      return false;

    // Do not return an oversized block for a large request
    if ((p.size() < TPUAllocatorConfig::max_split_size()) &&
        ((*it)->size >= TPUAllocatorConfig::max_split_size()))
      return false;
    // Allow oversized block size to be rounded up but within a limit
    if ((p.size() >= TPUAllocatorConfig::max_split_size()) &&
        ((*it)->size >=
         p.size() + TPUAllocatorConfig::max_non_split_rounding_size()))
      return false;
    p.block = *it;
    pool.blocks.erase(it);
    return true;
  }

  // This function assumes that global lock has been taken whle calling into
  // this function. We do tpuMalloc sync call in this function which
  // can be expensive while holding the lock. Hence, we pass-in the lock to the
  // function to temporarily release the lock before tpuMalloc call and acquire
  // it back again after the call so that other threads dont get blocked.
  bool alloc_block(
      AllocParams& p,
      bool isRetry,
      std::unique_lock<std::recursive_mutex>& lock) {

    size_t size = p.alloc_size;
    void* ptr = nullptr;

    if (TPUAllocatorConfig::release_lock_on_tpumalloc()) {
      lock.unlock();
      std::unique_ptr<void, std::function<void(void*)>> guard(
          (void*)1,
          [&](void*){
            lock.lock();
          });
      p.err = tpuMallocMaybeCapturing(&ptr, size, p);
    } else {
      p.err = tpuMallocMaybeCapturing(&ptr, size, p);
    }
    if (TPUAllocatorConfig::release_lock_on_tpumalloc()) {
      TORCH_CHECK(
          lock.owns_lock(), "Failed to acquire lock after tpuMalloc");
    }
    if (p.err != tpuSuccess) {
      if (p.err == tpuErrorMemoryAllocation) {
        // If this is the first attempt (!isRetry), we can forgive and clear
        // TPu's internal error state.
        //
        // If this is the second attempt (isRetry), malloc's TORCH_CHECK_WITH
        // will take over to throw a helpful exception. The user can choose
        // to catch the exception, free some stuff in their script, and
        // attempt the allocation again. In this case, we can also forgive and
        // clear TPU's internal error state.
      } else {
        // If the error's unrelated to memory allocation, we should throw
        // immediately.
        C10_TPU_CHECK(p.err);
      }
      return false;
    }

    if (p.pool->owner_PrivatePool) {
      // The block is for a TPU graph's PrivatePool.
      p.pool->owner_PrivatePool->tpuMalloc_count++;
    }

    total_allocated_memory += size;
    p.block = new Block(p.device(), p.stream(), size, p.pool, (char*)ptr);

    return true;
  }

  /** Free one or more oversize blocks to the system allocator.  But only enough
   * **/
  /** to satisfy the target size **/
  bool release_available_cached_blocks(const AllocParams& p) {
    if (TPUAllocatorConfig::max_split_size() ==
        std::numeric_limits<size_t>::max())
      return false;
    BlockPool& pool = *p.pool;

    // because of std::unique_ptr, block cannot be trivially copied
    // Use constructor for search key.
    Block key(p.search_key.device, p.search_key.stream, p.search_key.size);
    key.size = (key.size < TPUAllocatorConfig::max_split_size())
        ? TPUAllocatorConfig::max_split_size()
        : key.size;
    auto it = pool.blocks.lower_bound(&key);
    if (it == pool.blocks.end() || (*it)->stream != p.stream()) {
      // No single block is large enough; free multiple oversize blocks,
      // starting with the largest
      if (it == pool.blocks.begin())
        return false;
      size_t totalReleased = 0;
      --it; // Back up one item.  Now on the largest block for the correct
            // stream
      while ((totalReleased < key.size) &&
             ((*it)->size >= TPUAllocatorConfig::max_split_size()) &&
             ((*it)->stream == p.stream())) {
        auto cur = it;
        bool is_first = cur == pool.blocks.begin();
        if (!is_first) {
          --it;
        }

        release_block(*cur);
        totalReleased += (*cur)->size;
        if (is_first) {
          break;
        }
      }
      if (totalReleased < key.size)
        return false;
    } else {
      release_block(*it);
    }
    return true;
  }

  bool release_cached_blocks(MempoolId_t mempool_id) {
    if (mempool_id.first == 0 && mempool_id.second == 0) {
      // If there is no active mempool, we work on releasing *all* blocks.

      // First ensure that all blocks that can't currently be allocated due to
      // outstanding events are returned to the pool.
      synchronize_and_free_events();

      // Free all non-split cached blocks to system allocator
      release_blocks(large_blocks);
      release_blocks(small_blocks);
    }

    for (auto it = graph_pools_freeable.begin();
         it != graph_pools_freeable.end();) {
      if (mempool_id.first != 0 || mempool_id.second != 0) {
        if (it->first == mempool_id) {
          // If there is an active mempool, we sync only the events
          // associated with the pool
          synchronize_and_free_events(it->second);
        } else {
          // otherwise we move on
          ++it;
          continue;
        }
      }
      // See notifyCaptureDestroy for the strategy here.
      TORCH_INTERNAL_ASSERT(it->second->use_count == 0);
      release_blocks(it->second->small_blocks);
      release_blocks(it->second->large_blocks);
      if (it->second->tpuMalloc_count == 0) {
        auto erase_count = graph_pools.erase(it->first);
        TORCH_INTERNAL_ASSERT(erase_count == 1);
        it = graph_pools_freeable.erase(it);
      } else {
        ++it;
      }
    }

    return true;
  }

  void release_block(Block* block) {

    auto* pool = block->pool;
    if (pool->owner_PrivatePool && pool->owner_PrivatePool->allocator()) {
      // If there is an active mempool with a given allocator,
      // we use the given allocator's delete function.
      pool->owner_PrivatePool->allocator()->free((void*)block->ptr);
    } else {
      auto stream = c10_tpu::getCurrentTPUStream();
      tpudnnFlush(stream);
      AT_TPU_CHECK(tpuStreamSynchronize(stream.stream()));
      C10_TPU_CHECK(tpuFree(block->ptr));
    }
    total_allocated_memory -= block->size;

    if (pool->owner_PrivatePool) {
      // The tpuFreed block belonged to a TPU graph's PrivatePool.
      TORCH_INTERNAL_ASSERT(pool->owner_PrivatePool->tpuMalloc_count > 0);
      pool->owner_PrivatePool->tpuMalloc_count--;
    }

    pool->blocks.erase(block);
    delete block;
  }

  void release_blocks(BlockPool& pool) {
    std::vector<Block*> to_unmap;
    // Frees all non-split blocks
    auto it = pool.blocks.begin();
    while (it != pool.blocks.end()) {
      Block* block = *it;
      ++it;
      if (!block->prev && !block->next) {
        release_block(block);
      }
    }
  }

  EventPool::Event create_event_internal(c10::DeviceIndex idx) {
    // Leak the event pool to avoid shutdown issues.
    static auto* event_pool = new EventPool();
    return event_pool->get(idx);
  }

  void synchronize_and_free_events(PrivatePool* pool = nullptr) {
    // This function syncs, so capture should not be underway. Might as well
    // make sure capture-deferred end of life events get processed too.
    TORCH_INTERNAL_ASSERT(captures_underway.empty());
    insert_events_deferred_until_no_capture();

    for (auto it = tpu_events.begin(); it != tpu_events.end();) {
      for (auto e = it->second.begin(); e != it->second.end();) {
        Block* block = e->second;

        // If a pool was passed, only synchronize the events
        // that are associated with the pool, otherwise move on
        if (pool && block->pool->owner_PrivatePool != pool) {
          ++e;
          continue;
        }

        EventPool::Event event = std::move(e->first);

        C10_TPU_CHECK(tpuEventSynchronize(*event));

        block->event_count--;
        if (block->event_count == 0) {
          free_block(block);
        }
        // We are done with the event, so erase it from the deque
        e = it->second.erase(e);
      }

      // If the events deque is empty, only then erase the
      // tpu event from the events map
      if (it->second.empty()) {
        it = tpu_events.erase(it);
      } else {
        it++;
      }
    }
  }

  void remove_tpugraph_stream_uses(Block* block) {
    // remove stream uses added during tpugraph capture
    // (i.e., block->stream_uses - block->tpugraph_stream_uses)
    if (C10_UNLIKELY(
            block_to_tpugraph_stream_uses.find(block) !=
            block_to_tpugraph_stream_uses.end())) {
      stream_set streams(std::move(block->stream_uses));
      AT_ASSERT(block->stream_uses.empty());
      for (auto& stream : streams) {
        if (block_to_tpugraph_stream_uses[block].find(stream) ==
            block_to_tpugraph_stream_uses[block].end()) {
          block->stream_uses.insert(stream);
        }
      }
      block_to_tpugraph_stream_uses.erase(block);
    }
  }

  void insert_events(Block* block) {
    int prev_device = 0;
    C10_TPU_CHECK(tpuGetDevice(&prev_device));

    stream_set streams(std::move(block->stream_uses));
    AT_ASSERT(block->stream_uses.empty());
    for (auto& stream : streams) {
      C10_TPU_CHECK(tpuSetDevice(stream.device_index()));

      EventPool::Event event = create_event_internal(stream.device_index());
      C10_TPU_CHECK(tpuEventRecord(*event, stream.stream()));

      block->event_count++;
      tpu_events[stream].emplace_back(std::move(event), block);
    }

    C10_TPU_CHECK(tpuSetDevice(prev_device));
  }

  void insert_events_deferred_until_no_capture() {
    if (C10_UNLIKELY(!needs_events_deferred_until_no_capture.empty())) {
      for (auto* block : needs_events_deferred_until_no_capture) {
        TORCH_INTERNAL_ASSERT(!block->stream_uses.empty());
        // only streams recorded before tpugraph will be used to insert events
        // since we know all streams recorded during tpugraph must have completed
        remove_tpugraph_stream_uses(block);
        insert_events(block);
        if (block->event_count == 0) {
          free_block(block);
        }
      }
      needs_events_deferred_until_no_capture.clear();
    }
  }

  void process_events() {
    insert_events_deferred_until_no_capture();

    // Process outstanding tpuEvents. Events that are completed are
    // removed from the queue, and the 'event_count' for the
    // corresponding allocation is decremented. We maintain a separate
    // list of events per stream to avoid head-of-line delays if one
    // or more streams has long-running operations.

    // Iterate over different streams.
    for (auto it = tpu_events.begin(); it != tpu_events.end();) {
      // Iterate over this stream's (event, block) pairs.
      while (!it->second.empty()) {
        auto& e = it->second.front();
        EventPool::Event event = std::move(e.first);
        Block* block = e.second;

        tpuError_t err = tpuEventQuery(*event);
        if (err == tpuErrorNoDevice) {
          // Return the ownership of the Event (unique ptr)
          e.first = std::move(event);
          break;
        } else if (err != tpuSuccess) {
          C10_TPU_CHECK(err);
        }

        block->event_count--;
        if (block->event_count == 0) {
          free_block(block);
        }
        it->second.pop_front();
      }

      if (it->second.empty()) {
        it = tpu_events.erase(it);
      } else {
        it++;
      }
    }
  }

  // Iterates over sizes of all memory blocks for given device in given pool
  void cache_info_aux(const BlockPool& pool, size_t* largest) {
    for (const auto& block : pool.blocks) {
      const auto blocksize = block->size;
      if (blocksize > *largest) {
        *largest = blocksize;
      }
    }
  }

};

static void local_raw_delete(void* ptr);

void CachingTPUAllocator::add_allocated_block(Block* block) const {
  std::lock_guard<std::mutex> lock(mutex);
  allocated_blocks[block->ptr] = block;
}

Block* CachingTPUAllocator::get_allocated_block(void* ptr, bool remove) {
  std::scoped_lock<std::mutex> lock(mutex);
  auto it = allocated_blocks.find(ptr);
  if (it == allocated_blocks.end()) {
    return nullptr;
  }
  Block* block = it->second;
  if (remove) {
    allocated_blocks.erase(it);
  }
  return block;
}

void CachingTPUAllocator::init() {
  int device_count = 0;
  tpuGetDeviceCount(&device_count);
  const auto size = static_cast<DeviceIndex>(device_allocators.size());
  if (size < device_count) {
    device_allocators.resize(device_count);
    for (const auto i : c10::irange(size, device_count)) {
      device_allocators[i] = std::make_unique<DeviceCachingAllocator>();
    }
  }
}

void CachingTPUAllocator::malloc(
    void** devPtr,
    DeviceIndex device,
    size_t size) const {
  TORCH_INTERNAL_ASSERT(
      0 <= device && static_cast<size_t>(device) < device_allocators.size(),
      "Allocator not initialized for device ",
      static_cast<int16_t>(device),
      ": did you call init?");
  Block* block = device_allocators[device]->malloc(device, size, c10_tpu::getCurrentTPUStream().stream());
  add_allocated_block(block);
  *devPtr = block->ptr;
}

void CachingTPUAllocator::free(void* ptr) {
  if (!ptr) {
    return;
  }
  Block* block = this->get_allocated_block(ptr, /* remove */ true);
  TORCH_CHECK(block, "invalid device pointer: ", ptr);
  device_allocators[block->device]->free(block);
}

void CachingTPUAllocator::emptyCache(MempoolId_t mempool_id) {
  for (auto& da : device_allocators) {
    da->emptyCache(mempool_id);
  }
}

at::DataPtr CachingTPUAllocator::allocate(size_t size) {
  auto device = c10_tpu::current_device();
  void* r = nullptr;
  if (size != 0) {
    this->malloc(&r, device, size);
  }
  return {r, r, &local_raw_delete, Device(DeviceType::TPU, device)};
}

at::DeleterFnPtr CachingTPUAllocator::raw_deleter() const {
  return &local_raw_delete;
}

void* CachingTPUAllocator::raw_alloc(size_t size) {
  if (size == 0) {
    return nullptr;
  }
  auto device = c10_tpu::current_device();
  void* r = nullptr;
  malloc(&r, device, size);
  return r;
}
void CachingTPUAllocator::createOrIncrefPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    TPUAllocator* allocator) {
  device_allocators[device]->createOrIncrefPool(
      std::move(mempool_id), allocator);
}

void CachingTPUAllocator::setUseOnOOM(c10::DeviceIndex device, MempoolId_t mempool_id) {
  device_allocators[device]->setUseOnOOM(std::move(mempool_id));
}

void CachingTPUAllocator::beginAllocateToPool(
    c10::DeviceIndex device,
    MempoolId_t mempool_id,
    std::function<bool(tpuStream_t)> filter) {
  device_allocators[device]->beginAllocateToPool(
      std::move(mempool_id), std::move(filter));
}

void CachingTPUAllocator::endAllocateToPool(c10::DeviceIndex device, MempoolId_t mempool_id) {
  device_allocators[device]->endAllocateToPool(mempool_id);
}

void CachingTPUAllocator::releasePool(c10::DeviceIndex device, MempoolId_t mempool_id) {
  device_allocators[device]->releasePool(std::move(mempool_id));
}

int CachingTPUAllocator::getPoolUseCount(c10::DeviceIndex device, MempoolId_t mempool_id) {
  return device_allocators[device]->getPoolUseCount(std::move(mempool_id));
}

static CachingTPUAllocator g_caching_tpu_allocator;

void local_raw_delete(void* ptr) {
    g_caching_tpu_allocator.free(ptr);
}

TPUAllocator* GetCachingTPUAllocator() {
  return &g_caching_tpu_allocator;
}

} // namespace TPUCachingAllocator

// uid_ is incremented when a user creates a MemPool,
// for example: using graph_pool_handle() or c10_tpu::MemPool().
//
// uuid_ is incremented when TPUGraph creates a MemPool
// as a result of a user not providing a pool.
//
// MempoolId_t of {0, 0} is used to denote when no MemPool has been
// passed to a function, either by user or TPUGraphs. For example,
// default value of MempoolId_t for capture_begin function is {0, 0}.
// That's why uid_ and uuid_ start at 1.
std::atomic<CaptureId_t> MemPool::uid_{1};
std::atomic<CaptureId_t> MemPool::uuid_{1};

MemPool::MemPool(
    TPUAllocator* allocator,
    bool is_user_created,
    bool use_on_oom)
    : allocator_(allocator), is_user_created_(is_user_created) {
  if (is_user_created_) {
    id_ = {0, uid_++};
  } else {
    id_ = {uuid_++, 0};
  }
  device_ = c10_tpu::current_device();
  TPUCachingAllocator::createOrIncrefPool(device_, id_, allocator);
  if (use_on_oom) {
    TPUCachingAllocator::setUseOnOOM(device_, id_);
  }
}

MemPool::~MemPool() {
  TORCH_INTERNAL_ASSERT(use_count() == 1);
  TPUCachingAllocator::releasePool(device_, id_);
  TPUCachingAllocator::emptyCache(id_);
}

MempoolId_t MemPool::id() {
  return id_;
}

TPUAllocator* MemPool::allocator() {
  return allocator_;
}

int MemPool::use_count() {
  return TPUCachingAllocator::getPoolUseCount(device_, id_);
}

c10::DeviceIndex MemPool::device() {
  return device_;
}

MempoolId_t MemPool::graph_pool_handle(bool is_user_created) {
  if (is_user_created) {
    return {0, uid_++};
  }
  return {uuid_++, 0};
}

} // namespace c10_tpu
