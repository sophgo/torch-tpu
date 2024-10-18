#pragma once

#include <c10/core/Device.h>
#include "torch_tpu/csrc/core/TPULog.h"
#include <sgdnn_api.h>
#include <tpuv7_rt.h>

namespace c10_tpu {
namespace sgrt {
/********************************************************
 *
 *               DEVICE RPOPRITYE RELATED
 *
**********************************************************/
enum MemType
{
    SGRT_GLOBAL_MEM = 0,
};

static
const char* SgrtGetSocName() { return "BM1690"; }

static
tpuRtStatus_t SgrtGetMemInfo(MemType mem_type, size_t* free_size, size_t* total_size)
{
  //TODO
  *free_size = 0;
  *total_size = 0;
  return tpuRtSuccess;
}

/********************************************************
 *
 *    DEVICE OPERATION RELATED
 *
**********************************************************/
enum sgrtStreamStatus {
    SG_STREAM_STATUS_COMPLETE  = 0,
    SG_STREAM_STATUS_NOT_READY = 1,
    SG_STREAM_STATUS_RESERVED  = 0xFFFF,
};
enum sgrtEventRecordedStatus {
    SG_EVENT_RECORDED_STATUS_NOT_READY = 0,
    SG_EVENT_RECORDED_STATUS_COMPLETE  = 1,
};

using sgrtStream = struct tpuRtStream;
using sgrtStream_t = tpuRtStream_t;
using sgrtStreamStatus = enum sgrtStreamStatus;
using sgrtEvent = struct tpuRtEvent;
using sgrtEvent_t = tpuRtEvent_t;
using sgrtEventRecordedStatus = enum sgrtEventRecordedStatus;

const char *SgGetErrMsg();

/****************************************************
 * * *             Streams Func                 * * *
 ****************************************************/

tpuRtStatus_t SgrtCreateStream(sgrtStream_t* pstream);

tpuRtStatus_t SgrtCreateStreamWithPriority(sgrtStream_t* pstream, int stream_flag, int prority);

sgrtStreamStatus SgrtStreamQuery(sgrtStream_t stream);

void SgrtSynchronizeStream(sgrtStream_t stream);

/****************************************************
 * * *            Events Func                   * * *
 ****************************************************/

tpuRtStatus_t SgrtCreateEvent(sgrtEvent_t* pEvent);

/**
 * @brief create event instance
 * @param pEvent [OUT]   created event
 * @param flag [IN]     event flag
 * @retval tpuRtSuccess The function is successfully executed.
 */
tpuRtStatus_t SgrtCreateEventWithFlag(sgrtEvent_t* pEvent, uint32_t flag);

tpuRtStatus_t SgrtEventDestroy(sgrtEvent_t event);

tpuRtStatus_t SgrtEventRecord(sgrtEvent_t event, sgrtStream_t stream);

tpuRtStatus_t SgrtStreamWaitEvent(sgrtStream_t stream, sgrtEvent_t event);

tpuRtStatus_t SgrtEventElapsedTime(float* elapsed_time_ms, sgrtEvent_t cur_event, sgrtEvent_t pre_event);

tpuRtStatus_t SgrtSynchronizeEvent(sgrtEvent_t event);

tpuRtStatus_t SgrtQueryEventStatus(sgrtEvent_t event, sgrtEventRecordedStatus* pStatus );

/****************************************************
 * * *            Device Func                   * * *
 ****************************************************/
/**
 * @brief lookup if device can access peer
 * @retval true The function is successfully executed.
 */
bool can_device_access_peer(c10::DeviceIndex device_id, c10::DeviceIndex peer_device_id);

/**
 * @brief sync all the streams of the device(which device?).
 * @retval tpuRtSuccess The function is successfully executed.
 */
tpuRtStatus_t SgrtDeviceSynchronize();

} // namespace sgrt
} // namespace c10_tpu
