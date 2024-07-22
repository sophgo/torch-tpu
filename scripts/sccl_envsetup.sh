#!/bin/bash

export SCCL_PATH=$TPUTRAIN_TOP/collective_extension
export PATH=$SCCL_PATH/install/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH/install/lib:$LD_LIBRARY_PATH
export TPUDNN_PATH=$TPUTRAIN_TOP/third_party/tpuDNN

build_type=Release
if [ "$TPUTRAIN_DEBUG" = "ON" ]; then
    build_type=Debug
fi

export GLOO_SOPHON_PATH=$TPUTRAIN_TOP/collective_extension/third_party/gloo_sophon/
export LD_LIBRARY_PATH=$TPUTRAIN_TOP/build/${build_type}/sgdnn:/usr/local/lib/python3.10/dist-packages/torch/lib:$TPUDNN_PATH/${CHIP_ARCH}_lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$TPUTRAIN_TOP/torch_tpu/lib/::$LD_LIBRARY_PATH
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

function rebuild_gloo_sophon()
{
    build_gloo_sophon 1 || return -1
}

function install_sccl()
{
    should_clean=$1
    if [ ! -d "$SCCL_PATH/build/" ]; then
        mkdir -p "$SCCL_PATH/build/"
    elif [ -n "$should_clean" ]; then
        rm -rf "$SCCL_PATH/build/"
        mkdir -p "$SCCL_PATH/build/"
    fi

    pushd $SCCL_PATH
    # python3 setup2260.py develop
    python3 setup2260.py bdist_wheel --plat-name linux_x86_64
    if [ $? -ne 0 ]; then popd; return -1; fi
    pip install dist/sccl_tpu*.whl --force-reinstall
    popd
}

function rebuild_sccl()
{
    # build_openmpi
    rebuild_gloo_sophon || return -1
    install_sccl 1 || return -1
}

function test_sccl() {
    local NODES=8
    if [ -n "$2" ]; then
      NODES=$2
    fi
    if [ ! -n "$1" ]; then
        echo "Usage: test_sccl <test_name> [nodes]"
        echo "test case is empty, please choose one from:"
        echo "test_all_gather.py"
        echo "test_all_reduce.py"
        echo "test_broadcast.py"
        echo "test_gather.py"
        echo "test_reduce.py"
        echo "test_scatter.py"
        echo "test_alltoall.py"
        return
    fi
    TEST_DIR=$TPUTRAIN_TOP/python/dist_test2260/
    pushd ${TEST_DIR} > /dev/null
    echo "test dir: $TEST_DIR"

    echo "################################"
    echo "Test sccl with mpi, NODES=$NODES"
    echo "################################"
    local PATH=$SCCL_PATH/install/bin:$PATH
    local LD_LIBRARY_PATH=$SCCL_PATH/install/lib:$LD_LIBRARY_PATH
    ret=0
    mpirun --use-hwthread-cpus  -n ${NODES} --allow-run-as-root -output-filename log python3 $TEST_DIR/$1 || ret=1
    if [ $ret -ne 0 ]; then
      echo "test sccl case: $1 failed"
      popd
      return $ret
    fi
    popd
}

function regression_for_sccl() {
    local_cases=(
        test_all_gather.py
        test_all_reduce.py
        test_broadcast.py
        test_gather.py
        test_reduce.py
        test_scatter.py
        test_alltoall.py
    )
    for case in ${local_cases[@]}; do
        test_sccl $case $1
        if [ $? -ne 0 ]; then
            return 1
        fi
    done
}