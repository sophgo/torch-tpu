#include "sg_api_struct.h"
#include "tpu_kernel.h"


int tpu_kernel_api_dummy ( const void * args )
{
    return 0;
}
TPUKERNEL_FUNC_REGISTER ( tpu_kernel_api_dummy );
