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
    echo -e "[message] wheel not installed, use reinstall_torch_tpu to install wheel package"
  fi
  popd
}

function uninstall_torch_tpu_dev(){
  python setup.py develop --uninstall
}

function uninstall_torch_tpu_deploy(){
  pip uninstall torch_tpu -y
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
  echo " - Delete torch_tpu/_C.cpython-*.so ..."
  rm -rf torch_tpu/_C.cpython-*.so
  popd
}

function soc_build()
{
  pushd ${TPUTRAIN_TOP}
  mkdir -p ${TPUTRAIN_TOP}/toolchains_dir
  export CROSS_TOOLCHAINS=${TPUTRAIN_TOP}/toolchains_dir/
  export SOC_CROSS_MODE=ON
  python setup.py build bdist_wheel --plat-name=aarch64
  unset SOC_CROSS_MODE
  popd
}

function update_head_files() {
  echo "updating tpuDNN.h ..."
  cp ${TPU1686_PATH}/tpuDNN/include/tpuDNN.h ${TPUTRAIN_TOP}/third_party/tpuDNN/include/
  echo "updating tpuDNNTensor.h ..."
  cp ${TPU1686_PATH}/tpuDNN/include/tpuDNNTensor.h ${TPUTRAIN_TOP}/third_party/tpuDNN/include/
  echo "updating sccl.h ..."
  cp ${TPU1686_PATH}/sccl/include/sccl.h ${TPUTRAIN_TOP}/third_party/sccl/include/
  echo "updating tpu_runtime_api.h ..."
  cp ${TPU1686_PATH}/runtime/tpu_runtime_api.h ${TPUTRAIN_TOP}/third_party/runtime_api/include/
}

function update_sg2260_third_party()
{
  update_head_files
  echo "updating libtpudnn.so ..."
  mkdir -p ${TPUTRAIN_TOP}/third_party/tpuDNN/${CHIP_ARCH}_lib/
  cp ${TPU1686_PATH}/build_${CHIP_ARCH}/tpuDNN/src/libtpudnn.so ${TPUTRAIN_TOP}/third_party/tpuDNN/${CHIP_ARCH}_lib/
  echo "updating libsccl.so ..."
  cp ${TPU1686_PATH}/build_${CHIP_ARCH}/sccl/libsccl.so ${TPUTRAIN_TOP}/third_party/sccl/lib/

  echo "updating libtpuv7_emulator.so ..."
  cp ${TPU1686_PATH}/build_${CHIP_ARCH}/firmware_core/libtpu*_emulator.so ${TPUTRAIN_TOP}/third_party/tpuv7_runtime/tpuv7-emulator_0.1.0/lib/

  echo "updating libfirmware_core.a ..."
  mkdir -p ${TPUTRAIN_TOP}/third_party/firmware/${CHIP_ARCH}/
  cp ${TPU1686_PATH}/build_fw_${CHIP_ARCH}/firmware_core/libfirmware_core.a ${TPUTRAIN_TOP}/third_party/firmware/${CHIP_ARCH}/
  echo "updating libtpurt.so ..."
  mkdir -p ${TPUTRAIN_TOP}/third_party/runtime_api/lib_${CHIP_ARCH}/
  cp ${TPU1686_PATH}/build_${CHIP_ARCH}/runtime/libtpurt.so ${TPUTRAIN_TOP}/third_party/runtime_api/lib_${CHIP_ARCH}/
}

function update_sg2260_riscv_third_party()
{
  update_head_files
  echo "updating libtpudnn.so ..."
  cp ${TPU1686_PATH}/build_${CHIP_ARCH}_riscv/tpuDNN/src/libtpudnn.so ${TPUTRAIN_TOP}/third_party/tpuDNN/${CHIP_ARCH}_lib/libtpudnn-riscv.so
  echo "updating libsccl.so ..."
  cp ${TPU1686_PATH}/build_${CHIP_ARCH}_riscv/sccl/libsccl.so ${TPUTRAIN_TOP}/third_party/sccl/lib/libsccl-riscv64.so
  echo "updating libtpuv7_emulator.so ..."
  cp ${TPU1686_PATH}/build_${CHIP_ARCH}_riscv/firmware_core/libtpuv7_emulator.so ${TPUTRAIN_TOP}/third_party/tpuv7_runtime/tpuv7-emulator_0.1.0/lib/libtpuv7_emulator-riscv.so
  echo "updating libfirmware_core.a ..."
  cp ${TPU1686_PATH}/build_fw_${CHIP_ARCH}/firmware_core/libfirmware_core.a ${TPUTRAIN_TOP}/third_party/firmware/${CHIP_ARCH}/libfirmware_core-riscv.a
  echo "updating libtpurt.so ..."
  cp ${TPU1686_PATH}/build_${CHIP_ARCH}_riscv/runtime/libtpurt.so ${TPUTRAIN_TOP}/third_party/runtime_api/lib_${CHIP_ARCH}/libtpurt-riscv64.so
}

function update_bm1684x_third_party()
{
  update_head_files
  echo "updating libtpudnn.so ..."
  cp ${TPU1686_PATH}/build_bm1684x/tpuDNN/src/libtpudnn.so ${TPUTRAIN_TOP}/third_party/tpuDNN/bm1684x_lib/
  echo "updating libcmodel_firmware.so ..."
  cp ${TPU1686_PATH}/build_bm1684x/firmware_core/libcmodel_firmware.so ${TPUTRAIN_TOP}/third_party/firmware/bm1684x/
  echo "updating libfirmware_core.a ..."
  cp ${TPU1686_PATH}/build_fw_bm1684x/firmware_core/libfirmware_core.a ${TPUTRAIN_TOP}/third_party/firmware/bm1684x/
  echo "updating libsccl.so ..."
  cp ${TPU1686_PATH}/build_bm1684x/sccl/libsccl.so ${TPUTRAIN_TOP}/third_party/sccl/lib/
  echo "updating libtpurt.so ..."
  cp ${TPU1686_PATH}/build_bm1684x/runtime/libtpurt.so ${TPUTRAIN_TOP}/third_party/runtime_api/lib_bm1684x/libtpurt.so
}

function update_bm1686_third_party()
{
  update_head_files
  echo "updating libtpudnn.so ..."
  cp ${TPU1686_PATH}/build_bm1686/tpuDNN/src/libtpudnn.so ${TPUTRAIN_TOP}/third_party/tpuDNN/bm1686_lib/
  echo "updating libcmodel_firmware.so ..."
  cp ${TPU1686_PATH}/build_bm1686/firmware_core/libcmodel_firmware.so ${TPUTRAIN_TOP}/third_party/firmware/bm1686/
  echo "updating libfirmware_core.a ..."
  cp ${TPU1686_PATH}/build_fw_bm1686/firmware_core/libfirmware_core.a ${TPUTRAIN_TOP}/third_party/firmware/bm1686/
  echo "updating libsccl.so ..."
  cp ${TPU1686_PATH}/build_bm1686/sccl/libsccl.so ${TPUTRAIN_TOP}/third_party/sccl/lib/
  echo "updating libtpurt.so ..."
  cp ${TPU1686_PATH}/build_bm1686/runtime/libtpurt.so ${TPUTRAIN_TOP}/third_party/runtime_api/lib_bm1686/libtpurt.so
}

function rebuild_TPU1686_riscv()
{
  unset skip_runtime
  if [[ "$EXTRA_CONFIG" != *USING_TPUDNN_TESTS* ]]; then
      export EXTRA_CONFIG="-DUSING_TPUDNN_TESTS=OFF $EXTRA_CONFIG"
  fi
  if [ "${CHIP_ARCH}" == "sg2260" ]; then
      SOC_CROSS_COMPILE=1 CMODEL_FW_BINARY_DIR=build_${CHIP_ARCH}_riscv DISABLE_ONEDNN=ON rebuild_firmware_cmodel_and_tpudnn || return -1
      FW_BINARY_DIR=build_fw_${CHIP_ARCH} rebuild_firmware || return -1
      update_sg2260_riscv_third_party
  elif [ "${CHIP_ARCH}" == "bm1684x" ]; then
      echo "not impl!!!"
  elif [ "${CHIP_ARCH}" == "bm1686" ]; then
      echo "not impl!!!"
  fi
}

function rebuild_TPU1686()
{
    if [[ "$EXTRA_CONFIG" != *USING_TPUDNN_TESTS* ]]; then
        export EXTRA_CONFIG="-DUSING_TPUDNN_TESTS=OFF $EXTRA_CONFIG"
    fi
    if [ "${CHIP_ARCH}" == "sg2260e" ]; then
        CMODEL_FW_BINARY_DIR=build_${CHIP_ARCH} EXTRA_CONFIG="-DUSING_RVTI=ON $EXTRA_CONFIG" rebuild_firmware_cmodel_and_tpudnn || return -1
        FW_BINARY_DIR=build_fw_${CHIP_ARCH} EXTRA_CONFIG="-DUSING_RVTI=ON $EXTRA_CONFIG" rebuild_firmware || return -1
    else
        CMODEL_FW_BINARY_DIR=build_${CHIP_ARCH} rebuild_firmware_cmodel_and_tpudnn || return -1
        FW_BINARY_DIR=build_fw_${CHIP_ARCH} rebuild_firmware || return -1
    fi
    if [ "${CHIP_ARCH}" == "sg2260" ] || [ "${CHIP_ARCH}" == "sg2260e" ]; then
        update_sg2260_third_party
    elif [ "${CHIP_ARCH}" == "bm1684x" ]; then
        update_bm1684x_third_party
    elif [ "${CHIP_ARCH}" == "bm1686" ]; then
        update_bm1686_third_party
    fi
}

function update_tpuv7()
{
  export TPURT_TOP=$TPUTRAIN_TOP/../tpuv7-runtime
  pushd $TPURT_TOP
  rm -rf build
  mkdir -p build
  cd build
  #-DUSING_DEBUG=ON
  cmake -DCMAKE_INSTALL_PREFIX=$PWD/../install -DUSING_CMODEL=ON -DUSING_ONEDNN=ON ..
  make -j$(nproc)
  make install
  popd
  cp -r $TPURT_TOP/install/tpuv7-runtime-emulator-onednn_*/* $TPUTRAIN_TOP/third_party/tpuv7_runtime/tpuv7-emulator_0.1.0/
}

function TorchTpuDebugMode()
{
  export TPUTRAIN_DEBUG=ON
}
function TorchTpuReleaseMode()
{
  unset TPUTRAIN_DEBUG
}
