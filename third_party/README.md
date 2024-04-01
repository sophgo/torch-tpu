#libsophon -latest currently support verion-id:
   commit f5e30d4e32e5accef7de39a7e21f66292f3417d9

# tpu_kernel
### include
    commit 1c4644118773ed8294afb37310c654805d12a3d2
    tpu_defs.h tpu_kernel.h
### bm1684x
    commit 1c4644118773ed8294afb37310c654805d12a3d2
    libbm1684x.a libbmlib_cmodel.so libcmodel_firmware.so
### sg2260
    [TPU1686] commit 7951085d312f05501788429b8158f1abed30da89
    [libsophon] commit b9622adb89a712c1d2cadec9dd83eabc8850f000
    libbmlib.so.0 libcmodel_firmware.so


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
