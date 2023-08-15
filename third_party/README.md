# tpu_kernel
### include
    commit 2f18e869d9662fcaaadb20d6b7937a2ae94b3bd5
    tpu_defs.h tpu_kernel.h
### bm1684x
    commit 8c1da04efed24f66872a93e0f47ed981b3d8bd4f
    libbm1684x.a libbmlib_cmodel.so libcmodel_fireware.so
### sg2260
    commit 710b06ef54fd7eab0a9c4a5a0ea346358e8b5c9c
    libbmlib.so.0 libcmodel_fireware.so


Note:
  [1]How to Compile BM1684X .so?
    --- for BM1684X
      1)Please check  tpu-train/third_party/bm1684x/README.md

    --- for sg2260
      #Inside nntc docker
      cd /workspace/nntoolchain/TPU1686
      source scripts/envsetup.sh sg2260
      export EXTRA_CONFIG=-DDEBUG=ON
      rebuild_test sgdnn

      #Inside tpu-train docker
      cp /workspace/libsophon/build/bmlib/libbmlib.so.0 /workspace/tpu-train/third_party/sg2260/libbmlib.so.0
      cd /workspace/tpu-train/third_party/sg2260/
      ln -s libbmlib.so.0 libbmlib.so
      cp /workspace/nntoolchain/TPU1686/build/firmware_core/libcmodel_firmware.so /workspace/tpu-train/third_party/sg2260/libcmodel_firmware.so
      ln -s libcmodel_firmware.so libcmodel.so

  [2]Debug Pattern
    1)Please update commit id if you want to change .so for a specific backend.
    2)Remember to update DEBUG verions of .so using export EXTRA_CONFIG=-DDEBUG=ON

  [3] include
   cp TPU1686/kernel/include/tpu_kernel.h .
   cp TPU1686/kernel/include/tpu_defs.h .
