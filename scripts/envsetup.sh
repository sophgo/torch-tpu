#!/bin/bash

function set_cmodel_firmware {
     export TPUKERNEL_FIRMWARE_PATH=`realpath $1`
     export SG_MAX_DEVICE_COUNT=1
}

function get_pytorch_install_dir(){
     pytorch_path=$(python -c \
                    "import torch; \
                     import os; \
                     print(os.path.dirname(os.path.realpath(torch.__file__))) \
                    ")
     export PYTORCH_INSTALL_DIR=${pytorch_path}
}

export CROSS_TOOLCHAINS=$HOME/workspace/
export TPUTRAIN_TOP=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)
export CHIP_ARCH=${1:-bm1684x}

export LIBSOPHON_STABLE=stable
export LIBSOPHON_LATEST=latest
export LIBSOPHON_LOCAL=local
export LIBSOPHON_PATTERN=${2:-$LIBSOPHON_LATEST} 
     # stable: means using libsophon install from .deb in /opt/sophon/libsophon-current
     # latest: means using libsophon-latest  its directory usually at the same level with tpu-train
     # local:  means using third_party's bmlib_cmodel.so
echo "export TPUTRAIN_TOP=${TPUTRAIN_TOP}"
echo "[INFO]CHIP_ARCH=$CHIP_ARCH"
echo "[INFO]LIBSOPHON_PATTERN=$LIBSOPHON_PATTERN"
if [ $LIBSOPHON_PATTERN = $LIBSOPHON_STABLE ]; then
     export LIBSOPHON_TOP=/opt/sophon/libsophon-current
elif [ $LIBSOPHON_PATTERN = $LIBSOPHON_LATEST ]; then
     export LIBSOPHON_TOP=$TPUTRAIN_TOP/../libsophon
else 
     export LIBSOPHON_TOP=$TPUTRAIN_TOP/third_party
fi

#source ${TPUTRAIN_TOP}/scripts/prepare_toolchains.sh
echo "[INFO]LIBSOPHON_TOP=$LIBSOPHON_TOP"

export PS1="\[\e[1;35m\]("train-"${CHIP_ARCH}):\[\e[1;33m\]\w\[\e[1;34m\]\$ \[\e[0m\]"
export FIRMWARE_CMODEL_PATH=$TPUTRAIN_TOP/third_party/$CHIP_ARCH/libcmodel_firmware.so
export BMLIB_PATH=$TPUTRAIN_TOP/third_party/$CHIP_ARCH/libbmlib.so
echo "[INFO]USING_FIRMWARE_CMODEL_PATH=$FIRMWARE_CMODEL_PATH"
source ${TPUTRAIN_TOP}/scripts/build_helper.sh
source ${TPUTRAIN_TOP}/scripts/release.sh
source ${TPUTRAIN_TOP}/scripts/regression.sh

export LD_LIBRARY_PATH=$TPUTRAIN_TOP/third_party/oneDNN/lib:$LD_LIBRARY_PATH

get_pytorch_install_dir;
echo "[INFO]PYTORCH_INSTALL_DIR=$PYTORCH_INSTALL_DIR"