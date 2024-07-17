#!/bin/bash

download_arm_toolchain(){
  local CROSS_TOOLCHAINS=$1
  mkdir -p $CROSS_TOOLCHAINS
  pushd $CROSS_TOOLCHAINS >>/dev/null
  arm_toolchain="gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu"
  if [ ! -d "${arm_toolchain}" ]; then
      echo "soc_cross_chain not found, try to download from web"
      if [ ! -e "${arm_toolchain}.tar.xz" ]; then
          pip3 install dfss
          pip3 install --upgrade dfss
          python3 -m dfss --url=open@sophgo.com:/toolchains/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz
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
  export CROSS_TOOLCHAINS=$CROSS_TOOLCHAINS
  export ARM_TOOLCHAIN=$CROSS_TOOLCHAINS/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu
fi

if [ ! -d "${RISCV_TOOLCHAIN}" ]; then
  echo "not found riscv_toolchain in ${ARM_TOOLCHAIN}"
  CROSS_TOOLCHAINS=$CURRENT_DIR/../toolchains_dir
  echo "try to download to  $CROSS_TOOLCHAINS ..."
  export CROSS_TOOLCHAINS=$CROSS_TOOLCHAINS
  download_riscv_toolchain $CROSS_TOOLCHAINS
fi

echo "[INFO]export ARM_TOOLCHAIN=$ARM_TOOLCHAIN"
echo "[INFO]export RISCV_TOOLCHAIN=$RISCV_TOOLCHAIN"
# popd>>/dev/null

function soc_build_env_prepare()
{
  pushd $CROSS_TOOLCHAINS >>/dev/null
  soc_cross_chain="gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu"
  if [ ! -d "${soc_cross_chain}" ]; then
      echo "soc_cross_chain not found, try to download from web"
      if [ ! -e "${soc_cross_chain}.tar.xz" ]; then
          pip3 install dfss
          pip3 install --upgrade dfss
          python3 -m dfss --url=open@sophgo.com:/toolchains/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu.tar.xz
      fi
      tar xvf "${soc_cross_chain}.tar.xz"
  fi
  # torch 
  torch_path="torchwhl"
  if [ ! -d "${torch_path}" ]; then
      echo "torch not found, try to download from web"
      wget https://pypi.tuna.tsinghua.edu.cn/packages/c3/83/67aba34223e77556ebe7d49d6d93eccfbea847f37a95f41a961b62476569/torch-2.1.0-cp38-cp38-manylinux2014_aarch64.whl
      mkdir -p ${torch_path}
      mv torch-2.1.0-cp38-cp38-manylinux2014_aarch64.whl ${torch_path}
      pushd ${torch_path}
      unzip *whl
      popd
  fi
  # python 3.82
  python382_path="Python-3.8.2"
  if [ ! -d "${python382_path}" ]; then
      if [ ! -e "${python382_path}.tar.xz" ]; then
          echo "Python-3.8.2.tar.xz not found, try to download from sophgo.com"
          pip3 install dfss
          pip3 install --upgrade dfss
          python3 -m dfss --url=open@sophgo.com:/toolchains/pythons/Python-3.8.2.tar.gz
      fi
      tar zxvf "${python382_path}.tar.gz"
  fi
  # libsophon soc
  libsophon_soc_path="libsophon_soc_0.5.0_aarch64"
  # libsophon_soc_0.5.0_aarch64.tar.gz
  if [ ! -d "${libsophon_soc_path}" ]; then
      if [ ! -e "${libsophon_soc_path}.tar.gz" ]; then
          echo "libsophon_soc_0.5.0_aarch64.tar.gz not found, try to download from sophgo.com"
          pip3 install dfss
          pip3 install --upgrade dfss
          python3 -m dfss --url=open@sophgo.com:/toolchains/libsophon_soc_0.5.0_aarch64.tar.gz
      fi
      tar zxvf "${libsophon_soc_path}.tar.gz"
  fi
  popd
}
