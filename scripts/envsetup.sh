#!/bin/bash

function set_cmodel_firmware {
     export TPUKERNEL_FIRMWARE_PATH=`realpath $1`
     export SG_MAX_DEVICE_COUNT=1
}

function update_pytorch_to_2_1(){
     echo "[INFO] updating pytorch to 2.1 ..."
     pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cpu --force-reinstall
}

function check_pytorch_version(){
     target_version="2.1.0"
     if ! python -c "import torch" &> /dev/null; then
          echo "[ERROR]错误：未找到 Torch，请确保已安装 Torch $target_version。"
          update_pytorch_to_2_1;
     fi
     torch_version=$(python -c "import torch; print(torch.__version__)")
     torch_major_version=$(echo "$torch_version" | cut -d. -f1-3)

     if [ "$torch_version" == "$torch_major_version" ]; then
          echo "[INFO]当前Torch版本是 $target_version"
     else
          echo "[ERROR]错误：当前Torch版本是 $torch_version，而不是 $target_version"
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

function set_v7runtime_env() {
     local root_path=$1
     local v7_lib_path=${root_path}/third_party/tpuv7_runtime/tpuv7-emulator_0.1.0/lib
     export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${v7_lib_path}
     export TPU_EMULATOR_PATH=${v7_lib_path}/libtpuv7_emulator.so
     export TPU_SCALAR_EMULATOR_PATH=${v7_lib_path}/libtpuv7_scalar_emulator.so

     echo 1 > /proc/sys/vm/overcommit_memory
     sysctl -w vm.overcommit_memory=1
     export TPU_KERNEL_PATH=${root_path}/build/Release/firmware_core/
     export TPUKERNEL_FIRMWARE_PATH=${root_path}/build/Release/firmware_core/libcmodel.so
}

################ MODE CHOICE ###################
export MODE_ASIC=stable        # stable: asic
export MODE_CMODEL=local       # local: cmodel
export MODE_PATTERN=${2:-$MODE_CMODEL}

######## ===== ENVS TO COMPILE TPUTRAIN ======########
export TPUTRAIN_TOP=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)
export TPUTRAIN_DEBUG=OFF
export CHIP_ARCH=${1:-bm1684x}  #sg2260
export PS1="\[\e[1;35m\]("train-"${CHIP_ARCH}-${MODE_PATTERN}):\[\e[1;33m\]\w\[\e[1;34m\]\$ \[\e[0m\]"

################cross toolchains###################
source ${TPUTRAIN_TOP}/scripts/prepare_toolchains.sh

############### PYTORCH ################
check_pytorch_version;
get_pytorch_install_dir;

echo "[INFO]export TPUTRAIN_TOP=${TPUTRAIN_TOP}"
echo "[INFO]CHIP_ARCH=$CHIP_ARCH"
echo "[INFO]TPUTRAIN_DEBUG=$TPUTRAIN_DEBUG"
echo "[INFO]LIBSOPHON_TOP=$LIBSOPHON_TOP"
echo "[INFO]PYTORCH_INSTALL_DIR=$PYTORCH_INSTALL_DIR"

source ${TPUTRAIN_TOP}/scripts/build_helper.sh
source ${TPUTRAIN_TOP}/scripts/regression.sh

export LD_LIBRARY_PATH=$TPUTRAIN_TOP/third_party/oneDNN/lib:$LD_LIBRARY_PATH

if [ "${CHIP_ARCH}" == "sg2260" ]; then
     set_v7runtime_env ${TPUTRAIN_TOP}
fi