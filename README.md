[1]PrePare:
1) Link host device to your docker:
    docker run --restart always --privileged -v /dev:/dev -td -v $PWD:/workspace --name <YOURS> sophgo/tpuc_dev:latest bash
    docker exec -it <YOURS>  bash
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
        #CHIP_ARCH is bm1684x, and libsophon is using  tpu-train/../libsophon (gerrit-latest)

    2.b) source scripts/envsetup.sh bm1684x stable
        #CHIP_ARCH is bm1684x, and libsophon is using  /opt/sophon/libsophon-current (.deb)

    2.c) source scripts/envsetup.sh sg2260 latest
        #CHIP_ARCH is sg2260,  and libsophon is using  tpu-train/../libsophon (gerrit-latest)

    2.d) source scripts/envsetup.sh sg2260 stable
        #CHIP_ARCH is sg2260,  and libsophon is using  /opt/sophon/libsophon-current (.deb)


[3]cmodel mode:
1. mkdir build && (re)build_all
2. set_cmodel_firmware ./build/firmware_core/libcmodel.so

[4]pcie_mode or soc_mode:
firstly make USING_CMODEL OFF && PCIE_MODE or SOC_MODE ON in config_common.cmake,
if you use SOC_MODE or need pybind, make ENABLE_PYBIND ON
1. if you have bm_prebuilt_toolchains, first set:
    export CROSS_TOOLCHAINS=path_to_bm_prebuilt_toolchains  (absolute path ex. /workspace/bm_prebuilt_toolchains )
    else run prepare_toolchains.sh first
2. mkdir build && cd build
3. cmake .. (-DCMAKE_BUILD_TYPE=Debug)
4. make kernel_module
5. make -j

Note:
    apt-get install bsdmainutils
    (if error /bin/sh: 1: hexdump: not found according to https://askubuntu.com/questions/1131417/install-hexdump-in-an-ubuntu-docker-image)

How to train:

Before training, you may need prepare libtorch, get it from PyTorch official website or Github. After you have libtorch, compile libtorch_plugin as README.
cmodel mode is too slow, so suggest you use pcie mode or soc mode, cmodel mode is just use for debug.
if you use pcie mode, training model as libtorch-resnet-cifar, then compile pytorch model file to bmodel use TPU-MLIR.
if you use soc mode, first regen model file by mlir2pytorch.py, and then train model, after training finished, update coeffs in bmodel.
