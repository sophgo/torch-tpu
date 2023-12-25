#ifndef DEVICE_MEM_ALLOCATOR_H
#define DEVICE_MEM_ALLOCATOR_H

#include "sgdnn_api.h"
#include "tpu_defs.h"
#include <cstring>
#include <functional>
#include <map>
#include <set>
#include <stdio.h>

#define ASSERT_INFO(_cond, fmt, ...)                                             \
  do {                                                                           \
      if (!(_cond)) {                                                            \
        printf("ASSERT %s: %s: %d: %s\n", __FILE__, __func__, __LINE__, #_cond); \
        printf("ASSERT info: " fmt "\n", ##__VA_ARGS__);                         \
      }                                                                          \
    } while (0)

static bool operator < (sg_device_mem_t m0, sg_device_mem_t m1){
    return memcmp(&m0, &m1, sizeof(m0))<0;
}

static sg_system_mem_t sg_mem_from_system(void *system_addr) {
  sg_system_mem_t mem;
  memset(&mem, 0x0, sizeof(sg_device_mem_t));
  mem.u.system.system_addr = system_addr;
  mem.flags.u.mem_type = BM_MEM_TYPE_SYSTEM;
  return mem;
}

class DeviceMemAllocator{
public:
    DeviceMemAllocator(bm_handle_t handle):
        _handle(handle) {}

    template<typename T=uint8_t>
    sg_device_mem_t alloc_on_device(size_t elem_size, std::function<size_t (T* ptr, size_t len)> init_func = nullptr){
        sg_device_mem_t mem;
        size_t byte_size = elem_size * sizeof(T);
        auto ret = sg_malloc_device_byte(_handle, &mem, byte_size);
        if(ret != BM_SUCCESS){
            destroy_mem();
            throw ret;
        }
        if(init_func){
            T* ptr = new T[elem_size];
            auto len = init_func(ptr, elem_size);
            auto ret = sg_memcpy_s2d_partial(_handle, mem, ptr, len*sizeof(T));
            delete [] ptr;
            if(ret != BM_SUCCESS){
                destroy_mem();
                throw ret;
            }
        }
        _mem_need_free.insert(mem);
        return mem;
    }

    void dealloc(const sg_device_mem_t& mem){
        if (sg_mem_get_type(mem) == BM_MEM_TYPE_SYSTEM){
            if(_mem_need_free.count(mem)){
                sg_free_device(_handle, mem);
                _mem_need_free.erase(mem);
            } else {
                destroy_mem();
                throw BM_ERR_FAILURE;
            }
        } else {
            auto ptr = (unsigned char*)sg_mem_get_system_addr(mem);
            if(_ptr_need_free.count(ptr)){
                delete [] ptr;
                _ptr_need_free.erase(ptr);
            } else {
                destroy_mem();
                throw BM_ERR_FAILURE;
            }
        }
    }

    template<typename T>
    sg_device_mem_t alloc_on_system(size_t elem_size, std::function<void (T* ptr, size_t len)> init_func = nullptr){
        size_t byte_size = elem_size*sizeof(T);
        unsigned char* ptr = new unsigned char[byte_size];
        if(ptr == nullptr){
            destroy_mem();
            throw BM_ERR_NOMEM;
        }
        if(init_func){
            init_func((T*)ptr, elem_size);
        }
        _ptr_need_free.insert(ptr);
        sg_device_mem_t mem;
        sg_mem_set_system_addr(&mem, ptr);
        return mem;
    }

    template<typename T>
    sg_device_mem_t map_input_to_device(const sg_device_mem_t& raw_mem, size_t elem_size){
        if (sg_mem_get_type(raw_mem) == BM_MEM_TYPE_SYSTEM){
            sg_device_mem_t new_mem = alloc_on_device<T>(elem_size);
            auto ret = sg_memcpy_s2d(_handle, new_mem, sg_mem_get_system_addr(raw_mem));
            if(ret != BM_SUCCESS){
                destroy_mem();
                throw ret;
            }
            return new_mem;
        }
        return raw_mem;
    }

    template<typename T>
    sg_device_mem_t map_output_to_device(const sg_device_mem_t& raw_mem, size_t elem_size,
                                         bool is_inplace = false){
        if (sg_mem_get_type(raw_mem) == BM_MEM_TYPE_SYSTEM){
            sg_device_mem_t new_mem = alloc_on_device<T>(elem_size);
            _post_copy_map[raw_mem] = new_mem;
            if (is_inplace) {
                auto ret = sg_memcpy_s2d(_handle, new_mem, sg_mem_get_system_addr(raw_mem));
                if(ret != BM_SUCCESS){
                    destroy_mem();
                    throw ret;
                }
            }
            return new_mem;
        }
        return raw_mem;
    }

    sg_device_mem_t map_output_to_device(const sg_device_mem_t& raw_mem, size_t elem_size, SgdnnDataType_t dtype,
                                         bool is_inplace = false){
        switch(dtype){
        case SGDNN_DTYPE_FP32:
            return map_output_to_device<float>(raw_mem, elem_size, is_inplace);
        case SGDNN_DTYPE_FP16:
            return map_output_to_device<float16>(raw_mem, elem_size, is_inplace);
        case SGDNN_DTYPE_BF16:
            return map_output_to_device<bfloat16>(raw_mem, elem_size, is_inplace);
        case SGDNN_DTYPE_INT8:
        // case SGDNN_DTYPE_INT4:
            return map_output_to_device<signed char>(raw_mem, elem_size, is_inplace);
        case SGDNN_DTYPE_UINT8:
        // case SGDNN_DTYPE_UINT4:
            return map_output_to_device<unsigned char>(raw_mem, elem_size, is_inplace);
        case SGDNN_DTYPE_INT16:
            return map_output_to_device<short>(raw_mem, elem_size, is_inplace);
        case SGDNN_DTYPE_UINT16:
            return map_output_to_device<unsigned short>(raw_mem, elem_size, is_inplace);
        case SGDNN_DTYPE_INT32:
            return map_output_to_device<int>(raw_mem, elem_size, is_inplace);
        case SGDNN_DTYPE_UINT32:
            return map_output_to_device<unsigned int>(raw_mem, elem_size, is_inplace);
        default:
            ASSERT_INFO(0, "unsupported dtype=%d", dtype);
        }
        return sg_device_mem_t{};
    }

    sg_device_mem_t map_input_to_device(const sg_device_mem_t& raw_mem, size_t elem_size, SgdnnDataType_t dtype){
        switch(dtype){
        case SGDNN_DTYPE_FP32:
            return map_input_to_device<float>(raw_mem, elem_size);
        case SGDNN_DTYPE_FP16:
            return map_input_to_device<float16>(raw_mem, elem_size);
        case SGDNN_DTYPE_BF16:
            return map_input_to_device<bfloat16>(raw_mem, elem_size);
        // case SGDNN_DTYPE_INT4:
        case SGDNN_DTYPE_INT8:
            return map_input_to_device<signed char>(raw_mem, elem_size);
        // case SGDNN_DTYPE_UINT4:
        case SGDNN_DTYPE_UINT8:
            return map_input_to_device<unsigned char>(raw_mem, elem_size);
        case SGDNN_DTYPE_INT16:
            return map_input_to_device<short>(raw_mem, elem_size);
        case SGDNN_DTYPE_UINT16:
            return map_input_to_device<unsigned short>(raw_mem, elem_size);
        case SGDNN_DTYPE_INT32:
            return map_input_to_device<int>(raw_mem, elem_size);
        case SGDNN_DTYPE_UINT32:
            return map_input_to_device<unsigned int>(raw_mem, elem_size);
        default:
            ASSERT_INFO(0, "unsupported dtype=%d", dtype);
        }
        return sg_device_mem_t{};
    }

    unsigned long long map_input_to_device_addr(void* raw_ptr, size_t elem_size, SgdnnDataType_t dtype){
        return map_input_to_device_addr(sg_mem_from_system(raw_ptr), elem_size, dtype);
    }
    unsigned long long map_input_to_device_addr(const sg_device_mem_t& raw_mem, size_t elem_size, SgdnnDataType_t dtype){

        auto mem = map_input_to_device(raw_mem, elem_size, dtype);
        return sg_mem_get_device_addr(mem);
    }

    unsigned long long map_output_to_device_addr(const sg_device_mem_t& raw_mem, size_t elem_size, SgdnnDataType_t dtype,
                                                 bool is_post_copy = false, bool is_inplace = false){
        auto mem = map_output_to_device(raw_mem, elem_size, dtype, is_inplace);
        return sg_mem_get_device_addr(mem);
    }

    unsigned long long map_output_to_device_addr(void* raw_ptr, size_t elem_size, SgdnnDataType_t dtype){
        return map_output_to_device_addr(sg_mem_from_system(raw_ptr), elem_size, dtype);
    }

    unsigned long long map_buffer_to_device_addr(size_t buffer_size){
        if (buffer_size == 0) return -1;
        auto mem = alloc_on_device<char>(buffer_size);
        return sg_mem_get_device_addr(mem);
    }

    void flush_output() {
        for(auto m: _post_copy_map){
            sg_memcpy_d2s(_handle, sg_mem_get_system_addr(m.first), m.second);
        }
        _post_copy_map.clear();
    }

    ~DeviceMemAllocator(){
        flush_output();
        destroy_mem();
    }

private:
    void destroy_mem(){
        for(auto p: _ptr_need_free){
            delete [] p;
        }
        _ptr_need_free.clear();
        for(auto m: _mem_need_free){
            sg_free_device(_handle, m);
        }
        _mem_need_free.clear();
    }
    bm_handle_t _handle;
    std::set<sg_device_mem_t> _mem_need_free;
    std::set<unsigned char*> _ptr_need_free;
    std::map<sg_device_mem_t, sg_device_mem_t> _post_copy_map;
};

#endif // BMCV_MEM_ALLOCATOR_H
