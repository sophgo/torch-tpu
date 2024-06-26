#!/bin/bash

export SCCL_PATH=$TPUTRAIN_TOP/collective_extension
export PATH=$SCCL_PATH/install/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH/install/lib:$LD_LIBRARY_PATH
export TPUDNN_PATH=$SCCL_PATH/third_party/tpudnn

build_type=Release
if [ "$TPUTRAIN_DEBUG" = "ON" ]; then
    build_type=Debug
fi

export GLOO_SOPHON_PATH=$TPUTRAIN_TOP/collective_extension/third_party/gloo_sophon/
export LD_LIBRARY_PATH=$TPUTRAIN_TOP/build/${build_type}/sgdnn:/usr/local/lib/python3.10/dist-packages/torch/lib:$TPUDNN_PATH/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TPUTRAIN_TOP/build/${build_type}/torch_tpu:$LD_LIBRARY_PATH
export TPU_KERNEL_MODULE_PATH=$TPUTRAIN_TOP/build/firmware_sg2260_cmodel/libfirmware.so

export MASTER_ADDR=127.0.0.1
export MASTER_PORT=6000

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

function install_sccl()
{
    pushd $GLOO_SOPHON_PATH/../../
    # python3 setup2260.py develop
    python3 setup2260.py bdist_wheel --plat-name linux_x86_64
    if [ $? -ne 0 ]; then popd; return -1; fi
    pip install dist/* --force-reinstall
    popd
}

function rebuild_sccl()
{
    # build_openmpi
    rebuild_gloo_sophon
    install_sccl
}