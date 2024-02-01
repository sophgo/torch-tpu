#!/bin/bash

function set_cmodel_firmware {
     export TPUKERNEL_FIRMWARE_PATH=`realpath $1`
     export SG_MAX_DEVICE_COUNT=1
}

function update_pytorch_to_2_1(){
     pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu --force-reinstall
}

function check_pytorch_version(){
     target_version="2.1.0"
     if ! python -c "import torch" &> /dev/null; then
          echo "错误：未找到 Torch，请确保已安装 Torch $target_version。"
          update_pytorch_to_2_1;
     fi
     torch_version=$(python -c "import torch; print(torch.__version__)")
     torch_major_version=$(echo "$torch_version" | cut -d. -f1-3)

     if [ "$torch_version" == "$torch_major_version" ]; then
          echo "当前Torch版本是 $target_version"
     else
          echo "错误：当前Torch版本是 $torch_version，而不是 $target_version"
          update_pytorch_to_2_1;
     fi
}

function get_pytorch_install_dir(){
     pytorch_path=$(python -c \
                    "import torch; \
                     import os; \
                     print(os.path.dirname(os.path.realpath(torch.__file__))) \
                    ")
     export PYTORCH_INSTALL_DIR=${pytorch_path}
}

######## ===== ENVS TO COMPILE TPUTRAIN ======########
export TPUTRAIN_TOP=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)
export TPUTRAIN_DEBUG=OFF
# cross toolchains
source ${TPUTRAIN_TOP}/scripts/prepare_toolchains.sh
export CHIP_ARCH=${1:-bm1684x}
export PS1="\[\e[1;35m\]("train-"${CHIP_ARCH}):\[\e[1;33m\]\w\[\e[1;34m\]\$ \[\e[0m\]"
export FIRMWARE_CMODEL_PATH=$TPUTRAIN_TOP/third_party/$CHIP_ARCH/libcmodel_firmware.so

################ BMLIB ###################
export LIBSOPHON_STABLE=stable     # stable: means using libsophon install from .deb in /opt/sophon/libsophon-current
export LIBSOPHON_LATEST=latest     # latest: means using libsophon-latest  its directory usually at the same level with tpu-train
export LIBSOPHON_LOCAL=local       # local:  means using third_party's bmlib_cmodel.so
export LIBSOPHON_PATTERN=${2:-$LIBSOPHON_LATEST} 
if [ $LIBSOPHON_PATTERN = $LIBSOPHON_STABLE ]; then
     export LIBSOPHON_TOP=/opt/sophon/libsophon-current
elif [ $LIBSOPHON_PATTERN = $LIBSOPHON_LATEST ]; then
     export LIBSOPHON_TOP=$TPUTRAIN_TOP/../libsophon
else 
     export LIBSOPHON_TOP=$TPUTRAIN_TOP/third_party
fi

############### PYTORCH ################
check_pytorch_version;
get_pytorch_install_dir;

echo "======== TPUTRAIN ENVS:"
echo "export TPUTRAIN_TOP=${TPUTRAIN_TOP}"
echo "[INFO]CHIP_ARCH=$CHIP_ARCH"
echo "[INFO]LIBSOPHON_PATTERN=$LIBSOPHON_PATTERN"
echo "[INFO]TPUTRAIN_DEBUG=$TPUTRAIN_DEBUG"
echo "[INFO]LIBSOPHON_TOP=$LIBSOPHON_TOP"
echo "[INFO]USING_FIRMWARE_CMODEL_PATH=$FIRMWARE_CMODEL_PATH"
echo "[INFO]PYTORCH_INSTALL_DIR=$PYTORCH_INSTALL_DIR"

source ${TPUTRAIN_TOP}/scripts/build_helper.sh
source ${TPUTRAIN_TOP}/scripts/regression.sh

export LD_LIBRARY_PATH=$TPUTRAIN_TOP/third_party/oneDNN/lib:$LD_LIBRARY_PATH