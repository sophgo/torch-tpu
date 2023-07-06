#!/bin/bash

export CROSS_TOOLCHAINS=$HOME/workspace/
export TPUTRAIN_TOP=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)
export CHIP_ARCH=${1:-bm1684x}
echo "export TPUTRAIN_TOP=${TPUTRAIN_TOP}"
echo "CHIP_ARCH = $CHIP_ARCH"
#source ${TPUTRAIN_TOP}/scripts/prepare_toolchains.sh

export PS1="\[\e[1;35m\]("train-"${CHIP_ARCH}):\[\e[1;33m\]\w\[\e[1;34m\]\$ \[\e[0m\]"
function set_cmodel_firmware {
     export TPUKERNEL_FIRMWARE_PATH=`realpath $1`
     export SG_MAX_DEVICE_COUNT=1
}

function format_code {
     echo "formating $1"
     astyle -P -o -O -xe -xt0 -s2 -p $1
     echo "formated $1"
}

export BMLIB_CMODEL_PATH=$TPUTRAIN_TOP/third_party/$CHIP_ARCH/libcmodel_firmware.so
echo "USING_CMODEL_PATH = $BMLIB_CMODEL_PATH"
source ${TPUTRAIN_TOP}/scripts/build_helper.sh
source ${TPUTRAIN_TOP}/scripts/release.sh