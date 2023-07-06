# tpu_kernel
### bm1684x
    commit ac5099e5f36930d5f52fa980d098e8b3759585a5
    libbm1684x.a libbmlib_cmodel.so libcmodel_fireware.so
### sg2260
    commit ac5099e5f36930d5f52fa980d098e8b3759585a5
    libsg2260.a libbmlib_cmodel.so libcmodel_fireware.so


Note:
  [1]How to Compile .so?
  1) cd TPU1686
  2) source scripts/envsetup.sh  bm1684x (sg2260)
  3) export EXTRA_CONFIG=-DDEBUG=ON
  4) rebuild_all #gen  libcmodel_firmware.so
  5) rebuild_test sgdnn
  5.5) mkdir target && cd target
  6) cp ../build/firmware_core/libcmodel_firmware.so .
  6) cp  ../build_test/bmlib_tmp/libbmlib.so   libbmlib_cmodel.so&&ln -s libbmlib_cmodel.so libbmlib.so.0 #gen  libbmlib_cmodel.so && libbmlib.so.0
  7) cp  ../build_test/firmware_core/libfirmware_core.a libbm1684x.a (libsg2260.a) #gen libbm1684x.a or  libsg2260.a

  [2]Debug Pattern
    1)Please update commit id if you want to change .so for a specific backend.
    2)Remember to update DEBUG verions of .so using export EXTRA_CONFIG=-DDEBUG=ON