#pragma once

#define CHECK_TENSOR_IN_DEVICE(t) \
do                                                                 \
{                                                                  \
if ( c10::tpu::TPUPtrIsInCurrentDevice ( t.data_ptr() ) == false ) \
{                                                                  \
  LOG ( FATAL ) << #t << " is not in current device";              \
}                                                                  \
}                                                                  \
while ( 0 )
