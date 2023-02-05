source scripts/envsetup.sh
mkdir build
(re)build_all
cmodel:
  export BMLIB_CMODEL_PATH=./third_party/lib/libbm1684x_cmodel.so
  set_cmodel_firmware ./build/firmware_core/libfirmware_cmodel.so

pcie_mode or soc_mode build_firmware:
1. if you have bm_prebuilt_toolchains, first set: export CROSS_TOOLCHAINS=path_to_bm_prebuild_toolchains, else run prepare_toolchains.sh first
2. mkdir build && (re)build_all
3. load firmware to device: python3 tool/load_firmware.py --firmware path_to_bm1684x.bin
