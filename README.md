[1]PrePare:
1) Link host device to your docker:
    docker run --restart always --privileged -v /dev:/dev -td -v $PWD:/workspace --name <YOUR_NAME> sophgo/tpuc_dev:latest bash
2)  docker exec -it <YOUR_NAME> bash
2.5) You can download lisophon from:
    ftp 172.28.141.89
    User：AI
    PassWd：SophgoRelease2022
3)  Install lisophon:
    apt install  ./sophon-libsophon_0.4.x_amd64.deb ./sophon-libsophon-dev_0.4.x_amd64.deb
4)  Set Path:
    source /etc/profile.d/libsophon-bin-path.sh
Note:
    Do not compile tpu-train or nntoolchain in ONE same docker container, because libsophon for tpu-train and LIBSOPHON_TOP for nntoolchain is not same usually.

[2]How to build:
    source scripts/envsetup.sh

Or More detailed:
    source scripts/envsetup.sh bm1684x (sg2260)

Notice:
   1)default backend is bm1684x. Motice the docker front change to GREEN tpu-bm1684x(sg2260).
   2)if backend is changed, you need to rewalk following compiling steps again COMPLETELY!

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
