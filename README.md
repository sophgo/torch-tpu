
Torch-TPU
=========

## SG2260 Emulator Mode Development - Recommended Setup

### Prerequisites

Firstly, get 'tpu-train', 'tpuv7-runtime' and 'TPU1686' repos:

Clone 'TPU1686' && 'tpuv7-runtime' as normally.

To clone 'tpu-train', you need first install git-lfs and pull third-party dependencies:

```bash
sudo apt install git-lfs

# MAKE SURE YOU ARE USING 2.x VERSION! Otherwise IT WILL NOT BE COMPATIBLE WITH GERRIT
git-lfs --version

# To use Gerrit HTTP protocal, you nee extra configs like:
export GIT_SSL_NO_VERIFY=1
export GIT_LFS_SKIP_SMUDGE=1

# Then clone 'tpu-train' with Gerrit HTTP protocal

# After clone, use the following commands to get large files:
cd tpu-train
git lfs install
git lfs pull --include '*' --exclude ''
```

# The three repos need orgnized as:
```
torch-tpu-dev/
├── tpuv7-runtime
├── TPU1686
└── tpu-train
```

Next, have docker installed and pull the latest torch-tpu dev docker image:

```bash
docker pull sophgo/torch_tpu:v0.1
```

Then start a docker container:

```bash
cd torch-tpu-dev/
docker run -v $(pwd):/workspace --restart always -td --name torch-tpu-dev sophgo/torch_tpu:v0.1 bash

# And into the container
docker exec -it torch-tpu-dev bash
```

### Build

```bash
cd tpu-train
source scripts/envsetup.sh sg2260 local

# Debug TPU1686, optional
export EXTRA_CONFIG='-DDEBUG=ON -DUSING_FW_PRINT=ON -DUSING_FW_DEBUG=ON'

# Build TPU base kernels
rebuild_TPU1686

# Make sure we have a clean env
pip uninstall --yes torch-tpu

# Debug torch-tpu, optional
export TPUTRAIN_DEBUG=ON

# Build torch-tpu and install editable
python setup.py develop
```

If everything went well, now we have a editable development install.

+ Any changes in .py files, you don't have to reinstall, it is usable on the fly.

+ If you change torch-tpu extension cpps, cd into `build/torch-tpu` and execute `make install` and you are ready to go.

+ If you change kernel source files in `firmware_core`, cd into `build/firmware_sg2260[_cmodel]` and execute `make`.

You can always execute `python setup.py develop` after changing source files to rebuild all binaries.
\
\
\

The following content is out of date, Please refer to `docs/developer_manual`
===========

[1]PrePare:
1) Link host device to your docker:
    sudo docker pull  sophgo/tpuc_dev:latest
    sudo docker run --restart always --privileged -v /dev:/dev -td -v $PWD:/workspace --name <YOUR_NAME> sophgo/tpuc_dev:latest bash
    sudo docker exec -it <YOUR_NAME>  bash
2) We offer two ways for libsophon
   2.A) "stable" mode: Install Libsophon from .deb, usually in /opt/sophon/libsophon-current
        2.A.1)You can download lisophon from:
                ftp 172.28.141.89
                User：AI
                PassWd：SophgoRelease2022
        2.A.2)  Then, install lisophon:
                apt install  ./sophon-libsophon_0.4.x_amd64.deb ./sophon-libsophon-dev_0.4.x_amd64.deb
        2.A.3)   Set Device Path:
                source /etc/profile.d/libsophon-bin-path.sh
        2.A.4)   Using bm-smi to check your TPU device in docker
    2.B) "latest" mode: Directly using libsphon-latest from gerrit-project
        2.B.1)  Just clone libsphon-latest
        2.B.2)  Usually, libsophon is not same with the driver installed on the server, so please do not map device info to docker.
    2.C) You need to choose stable or latest mode when source scripts/envsetup.sh

Note:
    Do not compile tpu-train or nntoolchain in ONE same docker container, because libsophon for tpu-train and LIBSOPHON_TOP for nntoolchain is not same usually.

[2]How to build:
    source scripts/envsetup.sh
    #it's default version, which means source scripts/envsetup.sh bm1684x latest
Or More detailed:
    2.a) source scripts/envsetup.sh bm1684x latest
        #CHIP_ARCH is bm1684x, and bmlib is using  tpu-train/../libsophon/build/bmlib/libbmlib.so (gerrit-latest)

    2.b) source scripts/envsetup.sh bm1684x local
        #CHIP_ARCH is bm1684x, and bmlib is using  tpu-train/third_party/bm1684x/libbmlib.so

    2.c) source scripts/envsetup.sh bm1684x stable
        #CHIP_ARCH is bm1684x, and bmlib is using  /opt/sophon/libsophon-current (.deb)

    2.d) source scripts/envsetup.sh sg2260 local
        #CHIP_ARCH is sg2260,  and tpuv7-runtime is using tpu-train/third_party/tpuv7_runtime

[3]cmodel mode:
1. new_clean && new_build
2. set_cmodel_firmware ./build/firmware_core/libcmodel.so

[4]pcie_mode or soc_mode:
firstly make USING_CMODEL OFF && PCIE_MODE or SOC_MODE ON in config_common.cmake, (or directly set -D in cmake)
1. if you have bm_prebuilt_toolchains, first set:
    export CROSS_TOOLCHAINS=path_to_bm_prebuilt_toolchains  (absolute path ex. /workspace/bm_prebuilt_toolchains )
    else run prepare_toolchains.sh first
2. mkdir build && cd build
3. cmake .. (-DCMAKE_BUILD_TYPE=Debug -DUSING_CMODEL=OFF -DPCIE_MODE=ON)
4. make kernel_module
5. make -j8
   5.1) quick build stable
      fast_build_bm1684x_stable   /workspace/libtorch_xxx   0.4.8

Note:
    apt-get install bsdmainutils
    (if error /bin/sh: 1: hexdump: not found according to https://askubuntu.com/questions/1131417/install-hexdump-in-an-ubuntu-docker-image)

How to train:

Before training, you may need prepare libtorch, get it from PyTorch official website or Github. After you have libtorch, compile libtorch_plugin as README.
cmodel mode is too slow, so suggest you use pcie mode or soc mode, cmodel mode is just use for debug.
if you use pcie mode, training model as libtorch-resnet-cifar, then compile pytorch model file to bmodel use TPU-MLIR.
if you use soc mode, first regen model file by mlir2pytorch.py, and then train model, after training finished, update coeffs in bmodel.
