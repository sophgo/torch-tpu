#!/bin/bash

export TPUTRAIN_TOP=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)
echo "export TPUTRAIN_TOP=${TPUTRAIN_TOP}"
#source ${TPUTRAIN_TOP}/scripts/prepare_toolchains.sh

function set_cmodel_firmware {
     export TPUKERNEL_FIRMWARE_PATH=`realpath $1`
}

export BMLIB_CMODEL_PATH=$TPUTRAIN_TOP/third_party/libcmodel_firmware.so

source ${TPUTRAIN_TOP}/scripts/build_helper.sh
