How to build:

source scripts/envsetup.sh

cmodel mode:
1. mkdir build && (re)build_all
2. set_cmodel_firmware ./build/firmware_core/libcmodel.so

pcie_mode or soc_mode:
firstly make USING_CMODEL OFF && PCIE_MODE or SOC_MODE ON in config_common.cmake,
if you use SOC_MODE or need pybind, make ENABLE_PYBIND ON
1. if you have bm_prebuilt_toolchains, first set:
    export CROSS_TOOLCHAINS=path_to_bm_prebuilt_toolchains
    else run prepare_toolchains.sh first
2. mkdir build && cd build
3. cmake .. (-DCMAKE_BUILD_TYPE=Debug)
4. make kernel_module
5. make -j

How to train:

Before training, you may need prepare libtorch, get it from PyTorch official website or Github. After you have libtorch, compile libtorch_plugin as README.
cmodel mode is too slow, so suggest you use pcie mode or soc mode, cmodel mode is just use for debug.
if you use pcie mode, training model as libtorch-resnet-cifar, then compile pytorch model file to bmodel use TPU-MLIR.
if you use soc mode, first regen model file by mlir2pytorch.py, and then train model, after training finished, update coeffs in bmodel.
