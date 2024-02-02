#!/bin/bash

download_arm_toolchain(){
  local CROSS_TOOLCHAINS=$1
  mkdir -p $CROSS_TOOLCHAINS
  pushd $CROSS_TOOLCHAINS >>/dev/null
  arm_toolchain="gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu"
  if [ ! -d "${arm_toolchain}" ]; then
      if [ ! -e "${arm_toolchain}.tar.xz" ]; then
          wget https://developer.arm.com/-/media/Files/downloads/gnu-a/10.3-2021.07/binrel/${arm_toolchain}.tar.xz
      fi
      tar xvf "${arm_toolchain}.tar.xz"
  fi
  popd>>/dev/null
}

download_riscv_toolchain(){
  echo "Please download from bm_prebuilt_toolchains"
  echo "not support url to download now"
  #wget https://soc.ustc.edu.cn/CECS/appendix/riscv64.tar.gz
}


CURRENT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
PREBUILT_PATH=$CURRENT_DIR/../../bm_prebuilt_toolchains
export ARM_TOOLCHAIN=$PREBUILT_PATH/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu
export RISCV_TOOLCHAIN=$PREBUILT_PATH/riscv64-linux-x86_64

if [ ! -d "${ARM_TOOLCHAIN}" ]; then
  echo "[WARNING] not found arm_toolchain in ${ARM_TOOLCHAIN}"
  CROSS_TOOLCHAINS=$CURRENT_DIR/../toolchains_dir
  echo "[WARNING] will use $CROSS_TOOLCHAINS"
  download_arm_toolchain $CROSS_TOOLCHAINS
  export ARM_TOOLCHAIN=$CROSS_TOOLCHAINS/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu
fi

if [ ! -d "${RISCV_TOOLCHAIN}" ]; then
  echo "not found riscv_toolchain in ${ARM_TOOLCHAIN}"
  CROSS_TOOLCHAINS=$CURRENT_DIR/../toolchains_dir
  echo "try to download to  $CROSS_TOOLCHAINS ..."
  download_riscv_toolchain $CROSS_TOOLCHAINS
fi

echo "[INFO]export ARM_TOOLCHAIN=$ARM_TOOLCHAIN"
echo "[INFO]export RISCV_TOOLCHAIN=$RISCV_TOOLCHAIN"