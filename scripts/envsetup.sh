#!/bin/bash

export CROSS_TOOLCHAINS=$HOME/workspace/
export TPUTRAIN_TOP=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)
echo "export TPUTRAIN_TOP=${TPUTRAIN_TOP}"
#source ${TPUTRAIN_TOP}/scripts/prepare_toolchains.sh

function set_cmodel_firmware {
     export TPUKERNEL_FIRMWARE_PATH=`realpath $1`
     export SG_MAX_DEVICE_COUNT=1
}

function format_code {
     echo "formating $1"
     astyle -P -o -O -xe -xt0 -s2 -p $1
     echo "formated $1"
}

export BMLIB_CMODEL_PATH=$TPUTRAIN_TOP/third_party/libcmodel_firmware.so

source ${TPUTRAIN_TOP}/scripts/build_helper.sh
source ${TPUTRAIN_TOP}/scripts/release.sh