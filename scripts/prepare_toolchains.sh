#!/bin/bash
CURRENT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)

if [ -z "$CROSS_TOOLCHAINS" ]; then
  export CROSS_TOOLCHAINS=$CURRENT_DIR/../toolchains_dir
fi
echo "export CROSS_TOOLCHAINS=$CROSS_TOOLCHAINS"

mkdir -p $CROSS_TOOLCHAINS
pushd $CROSS_TOOLCHAINS >>/dev/null

#arm_toolchain="gcc-arm-10.3-2021.07-x86_64-aarch64-none-elf"
arm_toolchain="gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu"
if [ ! -d "${arm_toolchain}" ]; then
    if [ ! -e "${arm_toolchain}.tar.xz" ]; then
        wget https://developer.arm.com/-/media/Files/downloads/gnu-a/10.3-2021.07/binrel/${arm_toolchain}.tar.xz
    fi
    tar xvf "${arm_toolchain}.tar.xz"
fi

popd>>/dev/null
