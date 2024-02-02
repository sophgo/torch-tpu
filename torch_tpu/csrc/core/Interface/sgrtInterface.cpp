#ifdef BACKEND_SG2260
#include "sgrtInterface.h"

#define SGRT_CHECK(cmd) \
do \
{ \
  if ( !( ( cmd ) == tpuRtSuccess ) ) \
  { \
    printf ( "%s:%d:%s: %s failed\n", __FILE__, __LINE__, __func__, #cmd ); \
    throw; \
  } \
} \
while ( false )


namespace c10_tpu {
namespace sgrt {

const char *SgGetErrMsg() {
  return ""; //TODO
}

/****************************************************
 * * *             Streams Func                 * * *
 ****************************************************/

tpuRtStatus_t SgrtCreateStream(sgrtStream_t* pstream) {
  SGRT_CHECK(tpuRtStreamCreate(pstream));
  SGRT_CHECK(sgdnnInitialize ( *pstream ));
  return tpuRtSuccess;
}

tpuRtStatus_t SgrtCreateStreamWithPriority(sgrtStream_t* pstream, int stream_flag, int prority) {
  //TOOD impl priority mechainsm
  return SgrtCreateStream(pstream);
}

sgrtStreamStatus SgrtStreamQuery(sgrtStream_t stream) {
  // todo
  return SG_STREAM_STATUS_COMPLETE;
}

void SgrtSynchronizeStream(sgrtStream_t stream) {
  SGRT_CHECK(tpuRtStreamSynchronize(stream));
}

/****************************************************
 * * *                 Events                   * * *
 ****************************************************/
tpuRtStatus_t SgrtCreateEvent(sgrtEvent_t* pEvent) {
  // TODO flag
  return tpuRtEventCreate( pEvent );
}

tpuRtStatus_t SgrtCreateEventWithFlag(sgrtEvent_t* pEvent, uint32_t flag) {
  // TODO flag
  SOPHON_LOGW("flag is no use", flag);
  return tpuRtEventCreate( pEvent );
}

tpuRtStatus_t SgrtEventDestroy(sgrtEvent_t event) {
  //SGRT_CHECK(sgEventDestroy(event));
  return tpuRtSuccess;
}

tpuRtStatus_t SgrtEventRecord(sgrtEvent_t event, sgrtStream_t stream) {
  SGRT_CHECK(tpuRtEventRecord(event, stream));
  return tpuRtSuccess;
}

tpuRtStatus_t SgrtStreamWaitEvent(sgrtStream_t stream, sgrtEvent_t event) {
  SGRT_CHECK(tpuRtStreamWaitEvent(stream, event));
  return tpuRtSuccess;
}

tpuRtStatus_t SgrtEventElapsedTime(float* elapsed_time_ms, sgrtEvent_t cur_event, sgrtEvent_t pre_event) {
  SGRT_CHECK(tpuRtEventElapsedTime(elapsed_time_ms, cur_event, pre_event));
  return tpuRtSuccess;
}

tpuRtStatus_t SgrtSynchronizeEvent(sgrtEvent_t event) {
  SGRT_CHECK(tpuRtEventSynchronize(event));
  return tpuRtSuccess;
}

tpuRtStatus_t SgrtQueryEventStatus(sgrtEvent_t event, sgrtEventRecordedStatus* pStatus )
{
  *pStatus = SG_EVENT_RECORDED_STATUS_COMPLETE;
  return tpuRtSuccess;
}

/****************************************************
 * * *            Device Func                   * * *
 ****************************************************/

bool can_device_access_peer(c10::DeviceIndex device_id, c10::DeviceIndex peer_device_id) {
  return true;
}

tpuRtStatus_t SgrtDeviceSynchronize() {
  SGRT_CHECK(tpuRtDeviceSynchronize());
  return tpuRtSuccess;
}

} // namespace sgrt
} // namespace c10_tpu

#endif //BACKEND_SG2260