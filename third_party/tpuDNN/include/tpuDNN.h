#pragma once

#include <cstdint>

extern "C"
{
    enum tpudnnStatus_t
    {
        TPUDNN_STATUS_SUCCESS,
        TPUDNN_STATUS_FAILED
    };

    typedef void *tpudnnHandle_t;

    tpudnnHandle_t tpudnnCreate(int deviceID = 0);
    void tpudnnDestroy(tpudnnHandle_t handle);

    tpudnnStatus_t tpudnnFlush(tpudnnHandle_t handle);

    void *tpudnnPhysToVirt(tpudnnHandle_t handle, uint64_t addr);
    uint64_t tpudnnVirtToPhys(tpudnnHandle_t handle, void *addr);

    tpudnnStatus_t tpudnnCheckChipMap(tpudnnHandle_t handle, int world_size, int *chipMap);
    tpudnnStatus_t tpudnnGetC2CRing(tpudnnHandle_t handle, int world_size, int *chipMap);
    tpudnnStatus_t tpudnnGetUniqueId(tpudnnHandle_t handle, char* uniqueId);
    tpudnnStatus_t tpudnnSetupC2C(tpudnnHandle_t handle, int deviceID);
    tpudnnStatus_t tpudnnSetupC2CTopology(tpudnnHandle_t handle);
    tpudnnStatus_t tpudnnGetC2CTopology(tpudnnHandle_t handle, int *chipNum, const int **outTopology);

    tpudnnHandle_t tpudnnHandleFromStream(int deviceID, void* stream, void* module);

}

#include "tpuDNNTensor.h"
