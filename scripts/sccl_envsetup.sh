#!/bin/bash
# export TPUTRAIN_TOP=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)
# source $TPUTRAIN_TOP/scripts/envsetup.sh sg2260 latest

# if [ $CHIP_ARCH = sg2260 ] && [ $LIBSOPHON_PATTERN = latest ]; then
#     #libcmodel_fimwares of TPU1686 and tpu-train have been combined in tpu-train/build/firmware_core
#     # set_cmodel_firmware $TPUTRAIN_TOP/build/firmware_core/libcmodel.so
#     set_cmodel_firmware $TPU1686_TOP/build/firmware_core/libcmodel_firmware.so
#     echo "TPUKERNEL_FIRMWARE_PATH=$TPUKERNEL_FIRMWARE_PATH"
# else
#     echo "[ERROR] CHIP_ARCH is not sg2260 or LIBSOPHON_PATTERN is not latest"
#     exit 1
# fi

export SG1684X_TOP=$TPUTRAIN_TOP/../TPU1686

# export CMAKE_PREFIX_PATH=$TPU1686_TOP/install:$CMAKE_PREFIX_PATH
export SCCL_PATH=$TPUTRAIN_TOP/collective_extension
export PATH=$SCCL_PATH/install/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH/install/lib:$LD_LIBRARY_PATH
export TPUDNN_PATH=$SCCL_PATH/third_party/tpudnn
export TPURT_TOP=$TPUTRAIN_TOP/../tpuv7-runtime
# export TORCH_TPU_PATH=$TPUTRAIN_TOP/torch_tpu/
export GLOO_SOPHON_PATH=$TPUTRAIN_TOP/collective_extension/third_party/gloo_sophon/
export LD_LIBRARY_PATH=$TPUTRAIN_TOP/build/Release/sgdnn:/usr/local/lib/python3.10/dist-packages/torch/lib:$TPUDNN_PATH/lib:$LD_LIBRARY_PATH
export TPU_KERNEL_MODULE_PATH=$SG1684X_TOP/build/firmware_core/libcmodel_firmware.so

# These envs are here for the naughty tpurt
export TPU_KERNEL_PATH=${SG1684X_TOP}/build/firmware_core
export TPU_EMULATOR_PATH=${SG1684X_TOP}/build/firmware_core/libcmodel_firmware.so
export TPU_SCALAR_EMULATOR_PATH=${TPURT_TOP}/build/cdmlib/tp/daemon/libtpuv7_scalar_emulator.so

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=6000

#rebuild torch_tpu
# function build_torch_tpu()
# {
#     should_clean=$1
#     if [ ! -d "$TORCH_TPU_PATH/build/" ]; then
#         mkdir -p "$TORCH_TPU_PATH/build/"
#     elif [ -n "$should_clean" ]; then
#         rm -rf "$TORCH_TPU_PATH/build/"
#         mkdir -p "$TORCH_TPU_PATH/build/"
#     fi

#     pushd $TORCH_TPU_PATH/build/
#     cores=$(($(nproc)/2))
#     cmake .. -DCMAKE_BUILD_TYPE=Debug
#     make -j$cores
#     ret=$?
#     popd
#     if [ $ret -ne 0 ]; then return $ret; fi
# }

function build_openmpi() {
  if [ -e ${SCCL_PATH}/install/bin/mpirun ]; then
    return
  fi

  echo "Build openmpi"
  if [ ! -d ${SCCL_PATH}/openmpi-4.1.5 ]; then
    wget https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.5.tar.bz2 -P ${SCCL_PATH}
    tar xvf ${SCCL_PATH}/openmpi-4.1.5.tar.bz2 -C ${SCCL_PATH}
    rm -rf ${SCCL_PATH}/openmpi-4.1.5.tar.bz2
  fi

  pushd ${SCCL_PATH}/openmpi-4.1.5
  if [ ! -d build ]; then
    mkdir build
  fi
  cd build
  ../configure --disable-mpi-fortran --enable-mpi \
               --prefix=${SCCL_PATH}/install
  make -j$(nproc) install
  popd
}

function build_gloo_sophon()
{
    should_clean=$1
    if [ ! -d "$GLOO_SOPHON_PATH/build/" ]; then
        mkdir -p "$GLOO_SOPHON_PATH/build/"
    elif [ -n "$should_clean" ]; then
        rm -rf "$GLOO_SOPHON_PATH/build/"
        mkdir -p "$GLOO_SOPHON_PATH/build/"
    fi

    pushd $GLOO_SOPHON_PATH/build/
    cores=$(($(nproc)/2))
    cmake .. -DCMAKE_BUILD_TYPE=Debug
    make -j$cores
    ret=$?
    popd
    if [ $ret -ne 0 ]; then return $ret; fi
}

function rebuild_torch_tpu()
{
    build_torch_tpu 1
}

function rebuild_gloo_sophon()
{
    build_gloo_sophon 1
}

function install_sccl_collectives()
{
    pushd $GLOO_SOPHON_PATH/../../
    python3 setup2260.py develop
    ret=$?
    popd
    if [ $ret -ne 0 ]; then return $ret; fi
}

function install_dist_test2260() {
    pushd $TPUTRAIN_TOP/
    python3 setup.py bdist_wheel --plat-name linux_x86_64
    pip install dist/*.whl
    ret=$?
    popd
    if [ $ret -ne 0 ]; then return $ret; fi
}

function rebuild_sccl()
{
    # rebuild_torch_tpu
    rebuild_gloo_sophon
    install_sccl_collectives
    # install_dist_test2260
}