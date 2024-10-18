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
    if [[ ! -d ${root_path}/../TPU1686 && "$LD_LIBRARY_PATH" != *tpuv7_runtime* ]]; then
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${v7_lib_path}
    fi
    build_type=Release
    if [ "$TPUTRAIN_DEBUG" = "ON" ]; then
         build_type=Debug
    fi
    export TPU_KERNEL_PATH=${root_path}/build/firmware_sg2260_cmodel/
    export TPU_EMULATOR_PATH=${root_path}/build/firmware_sg2260_cmodel/libfirmware.so
    #export TPUKERNEL_FIRMWARE_PATH=${root_path}/build/firmware_sg2260_cmodel/libfirmware.so
}

######## ===== ENVS TO COMPILE TPUTRAIN ======########
export TPUTRAIN_TOP=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)
export CHIP_ARCH=${1:-bm1684x}  #sg2260

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

if [[ "$LD_LIBRARY_PATH" != *oneDNN* ]]; then
    export LD_LIBRARY_PATH=$TPUTRAIN_TOP/third_party/oneDNN/lib:$LD_LIBRARY_PATH
fi

if [ "${CHIP_ARCH}" == "sg2260" ]; then
     set_v7runtime_env ${TPUTRAIN_TOP}
fi

export TPU1686_PATH=$(realpath $TPUTRAIN_TOP/../TPU1686)
if [ ! -d $TPU1686_PATH ]; then
    unset TPU1686_PATH
else
    echo "Found TPU1686 in ${TPU1686_PATH}"
    source ${TPU1686_PATH}/scripts/envsetup.sh
fi

export TPU_SCALAER_EMULATOR_WORKDIR=${TPUTRAIN_TOP}/build

function rebuild_TPU1686()
{
    CMODEL_FW_BINARY_DIR=build_${CHIP_ARCH} rebuild_firmware_cmodel_and_tpudnn || return -1
    FW_BINARY_DIR=build_fw_${CHIP_ARCH} rebuild_firmware || return -1
    if [[ "$EXTRA_CONFIG" != *USING_TPUDNN_TESTS* ]]; then
        export EXTRA_CONFIG="$EXTRA_CONFIG -DUSING_TPUDNN_TESTS=OFF"
    fi
}

export PS1="\[\e[1;35m\]("train-"${CHIP_ARCH}):\[\e[1;33m\]\w\[\e[1;34m\]\$ \[\e[0m\]"

function update_tpuv7()
{
  export TPURT_TOP=$TPUTRAIN_TOP/../tpuv7-runtime
  pushd $TPURT_TOP
  rm -rf build/emulator-onednn
  mkdir -p build/emulator-onednn
  cd build/emulator-onednn
  cmake -DCMAKE_INSTALL_PREFIX=$PWD/../install -DUSING_CMODEL=ON -DUSING_ONEDNN=ON ../..
  make -j$(nproc)
  make driver
  make install
  popd
  rm -rf $TPUTRAIN_TOP/third_party/tpuv7_runtime/tpuv7-emulator_0.1.0/
  cp -r $TPURT_TOP/build/install/tpuv7-runtime-emulator-onednn_0.1.0 $TPUTRAIN_TOP/third_party/tpuv7_runtime/tpuv7-emulator_0.1.0/
}
