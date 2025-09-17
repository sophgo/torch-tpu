#include "TPUTorchUtils.h"

namespace tpu{

class OpCalDtype
{
const char* F16 = "F16";
const char* F32 = "F32";
const char* BF16= "BF16";

public:

  static OpCalDtype& Instance()
  {
    if ( instance_ == nullptr )
    { 
      instance_ = new OpCalDtype;
      instance_->_register_all();
    }
    return *instance_;
  }

  tpudnnDataType_t get_convDtype()    {return _conv;}
  tpudnnDataType_t get_convbwdDtype() {return _conv_bwd;}
  tpudnnDataType_t get_mmDtype()      {return _mm;}

private:
  OpCalDtype() = default;
  static OpCalDtype * instance_;


  /*** _register_all dtype ***/
  tpudnnDataType_t _conv     = TPUDNN_DTYPE_UNKNOWN;
  tpudnnDataType_t _conv_bwd = TPUDNN_DTYPE_UNKNOWN;
  tpudnnDataType_t _mm       = TPUDNN_DTYPE_UNKNOWN;
  void _register_all()
  {
    const char* conv_dtype =  std::getenv("TORCHTPU_CONV_OP_DTYPE");
    if (conv_dtype) 
    { 
      if      (strcmp(F16, conv_dtype) == 0)  _conv = TPUDNN_DTYPE_FP16;
      else if (strcmp(BF16, conv_dtype) == 0) _conv = TPUDNN_DTYPE_BF16;
      else if (strcmp(F32, conv_dtype) == 0)  _conv = TPUDNN_DTYPE_FP32;
      else TORCH_CHECK(false, "unsupport autocast dtype for TORCHTPU_CONV_OP_DTYPE \n");
    }
    const char* convbwd_dtype =  std::getenv("TORCHTPU_CONVBWD_OP_DTYPE");
    if (convbwd_dtype) 
    { 
      if      (strcmp(F16, convbwd_dtype) == 0)  _conv_bwd = TPUDNN_DTYPE_FP16;
      else if (strcmp(BF16, convbwd_dtype) == 0) _conv_bwd = TPUDNN_DTYPE_BF16;
      else if (strcmp(F32, convbwd_dtype) == 0)  _conv_bwd = TPUDNN_DTYPE_FP32;
      else TORCH_CHECK(false, "unsupport autocast dtype for TORCHTPU_CONVBWD_OP_DTYPE \n");
    }

    const char* mm_dtype =  std::getenv("TORCHTPU_MM_OP_DTYPE");
    if (mm_dtype) 
    { 
      if      (strcmp(F16, mm_dtype) == 0)  _mm = TPUDNN_DTYPE_FP16;
      else if (strcmp(BF16, mm_dtype) == 0) _mm = TPUDNN_DTYPE_BF16;
      else if (strcmp(F32, mm_dtype) == 0)  _mm = TPUDNN_DTYPE_FP32;
      else TORCH_CHECK(false, "unsupport autocast dtype for TORCHTPU_MM_OP_DTYPE \n");
    }
  }
};

OpCalDtype * OpCalDtype::instance_         = nullptr;
} // namespcae tpu