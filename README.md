source scripts/envsetup.sh

cmodel mode:
1. mkdir build && (re)build_all
2. set_cmodel_firmware ./build/firmware_core/libcmodel.so

pcie_mode or soc_mode:
1. if you have bm_prebuilt_toolchains, first set:
    export CROSS_TOOLCHAINS=path_to_bm_prebuilt_toolchains
    else run prepare_toolchains.sh first
2. mkdir build && cd build
3. cmake .. (-DCMAKE_BUILD_TYPE=Debug)
4. make kernel_module
5. make -j
