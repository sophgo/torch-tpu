# tpu_kernel
### include
    commit 2f18e869d9662fcaaadb20d6b7937a2ae94b3bd5
    tpu_defs.h tpu_kernel.h
### bm1684x
    commit 604e732b5d7336769b14477cc38e12b9bf544e3d
    libbm1684x.a libbmlib_cmodel.so libcmodel_fireware.so
### sg2260
    commit 01b05535b034085854f3e418a3a9bc68ff399a93
    libbmlib.so.0 libcmodel_fireware.so


Note:
  [1]How to Compile BM1684X .so?
    --- for BM1684X
      1)Please check  tpu-train/third_party/bm1684x/README.md
    --- for sg2260
      1) cd TPU1686
      2) source scripts/envsetup.sh sg2260
      3) export EXTRA_CONFIG=-DDEBUG=ON
      4) rebuild_test sgdnn
      5) cp libsophon/build/bmlib/libbmlib.so.0 libbmlib.so.0
      6) ln -s libbmlib.so.0 libbmlib.so
      7) cp build/firmware_core/libcmodel_firmware.so libcmodel_fireware.so

  [2]Debug Pattern
    1)Please update commit id if you want to change .so for a specific backend.
    2)Remember to update DEBUG verions of .so using export EXTRA_CONFIG=-DDEBUG=ON
  
  [3] include
   cp TPU1686/kernel/include/tpu_kernel.h .
   cp TPU1686/kernel/include/tpu_defs.h .
