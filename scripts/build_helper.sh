#!/bin/bash

function new_build()
{
  echo -e "\033[32m [warn] use develop_torch_tpu instead of new_build \033[0m"
  develop_torch_tpu
}

function develop_torch_tpu(){
  uninstall_torch_tpu_deploy
  pushd ${TPUTRAIN_TOP}
  python setup.py develop --user
  if [ $? -ne 0 ]; then popd; return -1; fi
  popd
}


function bdist_wheel()
{
  pushd ${TPUTRAIN_TOP}
  python setup.py bdist_wheel

  if [ $# -ne 1 ]; then 
    echo -e "\033[31m [warn] wheel not installed, use reinstall_torch_tpu to install wheel package \033[0m"
  fi
  popd
}

function bdist_wheel_cmodel()
{
  pushd ${TPUTRAIN_TOP}
  BUILD_CMODEL_WHEEL=ON python setup.py bdist_wheel

  if [ $# -ne 1 ]; then 
    echo -e "\033[31m [warn] wheel not installed, use reinstall_torch_tpu_cmodel to install wheel package build with cmodel \033[0m"
  fi
  popd
}

function uninstall_torch_tpu_dev(){
  python setup.py develop --uninstall  
}

function uninstall_torch_tpu_deploy(){
  pip uninstall torch_tpu -y
}

function reinstall_torch_tpu_cmodel(){
  uninstall_torch_tpu_dev
  uninstall_torch_tpu_deploy
  bdist_wheel_cmodel 1
  pip install dist/*+cmodel*

}

function reinstall_torch_tpu(){
  uninstall_torch_tpu_dev
  uninstall_torch_tpu_deploy
  bdist_wheel 1
  pip install $(find dist/ -name "*.whl" ! -name "*+cmodel*")
}


function new_clean()
{
  pushd ${TPUTRAIN_TOP}
  echo " - Delete build ..."
  rm -rf build
  echo " - Delete dist ..."
  rm -rf dist
  echo " - Delete torch_tpu.egg-info ..."
  rm -rf torch_tpu.egg-info
  echo " - Delete torch_tpu/lib ..."
  rm -rf torch_tpu/lib
  echo " - Delete torch_tpu/_C.cpython-310-x86_64-linux-gnu.so ..."
  rm -rf torch_tpu/_C.cpython-310-x86_64-linux-gnu.so
  popd
}

function soc_build()
{
  pushd ${TPUTRAIN_TOP}
  export SOC_CROSS_MODE=ON
  python setup.py build bdist_wheel --plat-name=aarch64
  unset SOC_CROSS_MODE
  popd
}

function update_sg2260_third_party()
{
  echo "updating tpuDNN.h ..."
  cp ${TPU1686_PATH}/tpuDNN/include/tpuDNN.h ${TPUTRAIN_TOP}/third_party/tpuDNN/include/
  echo "updating tpuDNNTensor.h ..."
  cp ${TPU1686_PATH}/tpuDNN/include/tpuDNNTensor.h ${TPUTRAIN_TOP}/third_party/tpuDNN/include/
  echo "updating sccl.h ..."
  cp ${TPU1686_PATH}/sccl/include/sccl.h ${TPUTRAIN_TOP}/third_party/sccl/include/
  echo "updating libtpudnn.so ..."
  cp ${TPU1686_PATH}/build_sg2260/tpuDNN/src/libtpudnn.so ${TPUTRAIN_TOP}/third_party/tpuDNN/sg2260_lib/
  echo "updating libsccl.so ..."
  cp ${TPU1686_PATH}/build_sg2260/sccl/libsccl.so ${TPUTRAIN_TOP}/third_party/sccl/sg2260_lib/
  echo "updating libtpuv7_emulator.so ..."
  cp ${TPU1686_PATH}/build_sg2260/firmware_core/libtpuv7_emulator.so ${TPUTRAIN_TOP}/third_party/tpuv7_runtime/tpuv7-emulator_0.1.0/lib/
  echo "updating libfirmware_core.a ..."
  cp ${TPU1686_PATH}/build_fw_sg2260/firmware_core/libfirmware_core.a ${TPUTRAIN_TOP}/third_party/firmware/sg2260/
}

function update_bm1684x_third_party()
{
  echo "updating tpuDNN.h ..."
  cp ${TPU1686_PATH}/tpuDNN/include/tpuDNN.h ${TPUTRAIN_TOP}/third_party/tpuDNN/include/
  echo "updating tpuDNNTensor.h ..."
  cp ${TPU1686_PATH}/tpuDNN/include/tpuDNNTensor.h ${TPUTRAIN_TOP}/third_party/tpuDNN/include/
  echo "updating libtpudnn.so ..."
  cp ${TPU1686_PATH}/build_bm1684x/tpuDNN/src/libtpudnn.so ${TPUTRAIN_TOP}/third_party/tpuDNN/bm1684x_lib/
  echo "updating libcmodel_firmware.so ..."
  cp ${TPU1686_PATH}/build_bm1684x/firmware_core/libcmodel_firmware.so ${TPUTRAIN_TOP}/third_party/firmware/bm1684x/
  echo "updating libfirmware_core.a ..."
  cp ${TPU1686_PATH}/build_fw_bm1684x/firmware_core/libfirmware_core.a ${TPUTRAIN_TOP}/third_party/firmware/bm1684x/
}

function rebuild_TPU1686()
{
    if [[ "$EXTRA_CONFIG" != *USING_TPUDNN_TESTS* ]]; then
        export EXTRA_CONFIG="-DUSING_TPUDNN_TESTS=OFF -DREMOVE_POLLS_IN_LLM=ON $EXTRA_CONFIG"
    fi
    CMODEL_FW_BINARY_DIR=build_${CHIP_ARCH} rebuild_firmware_cmodel_and_tpudnn || return -1
    FW_BINARY_DIR=build_fw_${CHIP_ARCH} rebuild_firmware || return -1
    if [ "${CHIP_ARCH}" == "sg2260" ]; then
        update_sg2260_third_party
    elif [ "${CHIP_ARCH}" == "bm1684x" ]; then
        update_bm1684x_third_party
    fi
}

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
