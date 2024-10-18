
# https://pytorch.org/docs/stable/cuda.html

TORCH.CUDA

This package adds support for CUDA tensor types.

It implements the same function as CPU tensors, but they utilize GPUs for computation.

It is lazily initialized, so you can always import it, and use is_available() to determine if your system supports CUDA.

CUDA semantics has more details about working with CUDA.

# status: to_validate, to_impl, support, not_support 

StreamContext                #support, to test
can_device_access_peer       #done, to test
current_blas_handle          #no support
current_device               #support 
current_stream               #support
default_stream               #support
device                       #done, to test
device_count                 #support
device_of                    #support
get_arch_list                #no support
get_device_capability        #no support
get_device_name              #done
get_device_properties        #done, to impl driver func.
get_gencode_flags            #no support
get_sync_debug_mode          #done, need to validate how work
init                         #support
ipc_collect                  #no support
is_available                 #support
is_initialized               #support
memory_usage                 #no support
set_device                   #support
set_stream                   #to impl
set_sync_debug_mode          #done, need to validate how work
stream                       #support
synchronize                  #support
utilization                  #no support
temperature                  #no support
power_draw                   #no support
clock_rate                   #no support
OutOfMemoryError             #to impl



Random Number Generator
====
get_rng_state                #to check
get_rng_state_all            #to check
set_rng_state                #to check
set_rng_state_all            #to check
manual_seed                  #to check
manual_seed_all              #to check
seed                         #to check
seed_all                     #to check
initial_seed                 #to check

Communication collectives
====
comm.broadcast
comm.broadcast_coalesced
comm.reduce_add
comm.scatter
comm.gather

Streams and events
===
Stream                       # support
ExternalStream               # support
Event                        # support

Graphs (beta)
===
is_current_stream_capturing
graph_pool_handle
CUDAGraph
graph
make_graphed_callables

Memory management
====
empty_cache                          #to impl
list_gpu_process                     #no support
mem_get_info                         #no support
memory_stats                         #to impl
memory_summary                       #support
memory_snapshot                      #to impl
memory_allocated                     #support
max_memory_allocated                 #support
reset_max_memory_allocated           #support
memory_reserved                      #support
max_memory_reserved                  #support
set_per_process_memory_fraction      #to impl
memory_cached                        #support
max_memory_cached                    #support
reset_max_memory_cached              #to impl
reset_peak_memory_stats              #to impl
caching_allocator_alloc              #to impl
caching_allocator_delete             #to impl
CUDAPluggableAllocator               #no support
change_current_allocator             #no support


NVIDIA Tools Extension (NVTX)
====
nvtx.mark                 #no support
nvtx.range_push           #no support
nvtx.range_pop            #no support

Jiterator (beta)
====
jiterator._create_jit_fn               #
jiterator._create_multi_output_jit_fn  #

Stream Sanitizer (prototype)
====
#TODO