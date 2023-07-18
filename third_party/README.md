# tpu_kernel
### include
    commit 2f18e869d9662fcaaadb20d6b7937a2ae94b3bd5
    tpu_defs.h tpu_kernel.h
### bm1684x
    commit 604e732b5d7336769b14477cc38e12b9bf544e3d
    libbm1684x.a libbmlib_cmodel.so libcmodel_fireware.so
### sg2260
    commit 6fc547f6cc8c34283cc73b0c0d5603cf2068da58
    libbmlib.so


Note:
  [1]How to Compile BM1684X .so?
    --- for BM1684X
      1)Please check  tpu-train/third_party/bm1684x/README.md
    --- for sg2260
      1) cd TPU1686
      2) source scripts/envsetup.sh  bm1684x
      3) export EXTRA_CONFIG=-DDEBUG=ON
      4) rebuild_test sgdnn
      5) cp  ../build_test/bmlib_tmp/libbmlib.so libbmlib.so

  [2]Debug Pattern
    1)Please update commit id if you want to change .so for a specific backend.
    2)Remember to update DEBUG verions of .so using export EXTRA_CONFIG=-DDEBUG=ON
  
  [3] include
   cp TPU1686/kernel/include/tpu_kernel.h .
   cp TPU1686/kernel/include/tpu_defs.h .
