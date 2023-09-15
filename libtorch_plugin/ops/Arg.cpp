#include <torch/library.h>
#include <torch/torch.h>
#include <ATen/core/TensorBase.h>
#include <ATen/EmptyTensor.h>
#include <TPUDeviceManager.h>
#include <TPUTorchUtils.h>
#include <sgdnn_api.h>

#include "common/config.h"

namespace at 
{
Tensor & argmax_out_tpu( const Tensor & self, c10::optional<int64_t> dim, bool keepdim, Tensor & out)
{
    CHECK_TENSOR_IN_DEVICE ( out );
    CHECK_TENSOR_IN_DEVICE ( self );
#if 0
    LOG( WARNING ) << "argmax use cpu impl";
    auto out_cpu = argmax( self.cpu(), dim, keepdim );
    out = out_cpu.to(out.device()).to(out.dtype());
#else
    if (self.dim() == 0) {
        auto out_cpu = argmax(self.cpu());
        tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                               out.nbytes());
        return out;
    }

    if (dim.has_value()){
        if(dim.value() < 0){
            dim = dim.value() + self.dim();
        }
        TORCH_CHECK(dim.value() >= 0 || dim.value() < self.dim());
    }

    TIMING_START;
    bm_status_t status = sgdnnArg(  tpu::TPUGetDeviceHandle(), 
                                    tpu::TPUGenerateSgdnnTensor(self),
                                    dim.has_value() ? dim.value() :self.dim(),
                                    0,
                                    tpu::TPUGenerateSgdnnTensor(out),
                                    tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
    TIMING_END(tpu::ARGMAX);

#endif
    return out;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "argmax.out",  argmax_out_tpu);
}

Tensor & argmin_out_tpu( const Tensor & self, c10::optional<int64_t> dim, bool keepdim, Tensor & out)
{
    CHECK_TENSOR_IN_DEVICE ( out );
    CHECK_TENSOR_IN_DEVICE ( self );
#if 0
    LOG( WARNING ) << "argmin use cpu impl";
    auto out_cpu = argmin( self.cpu(), dim, keepdim );
    out = out_cpu.to(out.device()).to(out.dtype());
#else
    if (self.dim() == 0) {
        auto out_cpu = argmin(self.cpu());
        tpu::TPUCopyHostToDevice(out.data_ptr(), out_cpu.contiguous().data_ptr(),
                               out.nbytes());
        return out;
    }
    if (dim.has_value()){
        if(dim.value() < 0){
            dim = dim.value() + self.dim();
        }
        TORCH_CHECK(dim.value() >= 0 || dim.value() < self.dim());
    }
    
    TIMING_START;
    bm_status_t status = sgdnnArg(  tpu::TPUGetDeviceHandle(), 
                                    tpu::TPUGenerateSgdnnTensor(self),
                                    dim.has_value() ? dim.value() : self.dim(),
                                    1,
                                    tpu::TPUGenerateSgdnnTensor(out),
                                    tpu::TPUGenerateSgdnnTensor(out));
    TORCH_CHECK(status == BM_SUCCESS);
    TIMING_END(tpu::ARGMIN);
#endif
    return out;
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "argmin.out",  argmin_out_tpu);
}

std::tuple<Tensor&, Tensor&> max_dim_max_out_tpu( const Tensor & self, int64_t dim, bool keepdim, Tensor  &values, Tensor  &indices)
{
    CHECK_TENSOR_IN_DEVICE ( indices );
    CHECK_TENSOR_IN_DEVICE ( values );
    CHECK_TENSOR_IN_DEVICE ( self );
#if 0
    LOG( WARNING ) << "max.dim use cpu impl";
    auto out_cpu =  max(self.cpu(),dim,keepdim);
    values = TENSOR_TO_TPU(std::get<0>(out_cpu));
    indices = TENSOR_TO_TPU(std::get<1>(out_cpu));
#else
    if (self.dim() == 0) {
        tpu::TPUCopyHostToDevice(values.data_ptr(), self.contiguous().data_ptr(),self.nbytes());
        indices.zero_();
        return {values,indices};
    }
    if(dim < 0){
        dim = dim + self.dim();
    }
    TORCH_CHECK(dim >= 0 || dim < self.dim());
    TIMING_START;
    bm_status_t status = sgdnnArg(  tpu::TPUGetDeviceHandle(), 
                                    tpu::TPUGenerateSgdnnTensor(self),
                                    dim,
                                    2,
                                    tpu::TPUGenerateSgdnnTensor(values),
                                    tpu::TPUGenerateSgdnnTensor(indices));
    TORCH_CHECK(status == BM_SUCCESS);
    TIMING_END(tpu::MAX_DIM);
#endif
    return {values,indices};
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "max.dim_max",  max_dim_max_out_tpu);
}

std::tuple<Tensor&, Tensor&> min_dim_min_out_tpu( const Tensor & self, int64_t dim, bool keepdim, Tensor  &values, Tensor  &indices)
{
    CHECK_TENSOR_IN_DEVICE ( indices );
    CHECK_TENSOR_IN_DEVICE ( values );
    CHECK_TENSOR_IN_DEVICE ( self );
#if 0
    LOG( WARNING ) << "min.dim use cpu impl";
    auto out_cpu =  min(self.cpu(),dim,keepdim);
    values = TENSOR_TO_TPU(std::get<0>(out_cpu));
    indices = TENSOR_TO_TPU(std::get<1>(out_cpu));
#else
    if (self.dim() == 0) {
        tpu::TPUCopyHostToDevice(values.data_ptr(), self.contiguous().data_ptr(),self.nbytes());
        indices.zero_();
        return {values,indices};
    }
    if(dim < 0){
        dim = dim + self.dim();
    }
    TORCH_CHECK(dim >= 0 || dim < self.dim());
    TIMING_START;
    bm_status_t status = sgdnnArg(  tpu::TPUGetDeviceHandle(), 
                                    tpu::TPUGenerateSgdnnTensor(self),
                                    dim,
                                    3,
                                    tpu::TPUGenerateSgdnnTensor(values),
                                    tpu::TPUGenerateSgdnnTensor(indices));
    TORCH_CHECK(status == BM_SUCCESS);
    TIMING_END(tpu::MIN_DIM);
#endif
    return {values,indices};
}

TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
 m.impl ( "min.dim_min",  min_dim_min_out_tpu);
}

}