### tpuv7 runtime

## commit id
- tpuv7_runtime
7219ce77b39fc11eb56a2b7d51a04537520573d5
- TPU1686
7975ce0d46403d69e60eeaf0e20eb1b410e4eddc

## intro
`tpuv7_runtime` 是TPUv7.0的异步运行时库。其中，

- `tpuv7-emulator` 是cmodel版本的device runtime, bmodel runtime和后端算子库.
包含以下的工具和内容:
- - tools
- - - bin/host_test : test tool for device control.
- - - bin/tpu_model_tool : cli to analyse bmodel.
- - - bin/tpu-model-rt : cli to run bmodel.
- - include
- - - tpuv7_modelrt.h : users api for model inference.
- - - tpuv7_rt.h : users api for device control.
- - lib
- - - libbmodel.so : bmodel format. generated from tpuv7-model-runtime/tpuv7-bmodel
- - - libcdm_daemon_emulator.so : chip daemon emulator. generated from cdmlib/ap/daemon
- - - libtpuv7_emulator.so : operator backend emulator.
- - - libtpuv7_scalar_emulator.so : tpu scalar emulator(dynamic loading).
- - - libtpuv7_modelrt.so : bmodel's runtime.
- - - libtpuv7_rt.so : runtime api lib for users.


## how to update

更新 `tpuv7-emulator`
----
  1) `git clone ssh://yu.hu@gerrit-ai.sophgo.vip:29418/tpuv7-runtime`(tpuv7-runtime and TPU1686 should in same folder)
  2) compile `libtpuv7_emulator.so`, which is the cmodel of operator backend.
   ```shell
   $cd TPU1686
   $git checkout origin/sg2260
   $source scripts/envsetup.sh sg2260
   $export EXTRA_CONFIG=-DDEBUG=ON
   $rebuild_firmware_cmodel
   ```
  3) compile others
    YOU CAN FOUND IN `tpuv7-runtime/README` generate `tpuv7-emulator_0.1.0`.
```shell
mkdir build && cd build
mkdir emulator && cd emulator
cmake -DCMAKE_INSTALL_PREFIX=$PWD/../install  -DUSING_CMODEL=ON ../..
#Debug版本 cmake -DCMAKE_INSTALL_PREFIX=$PWD/../install  -DUSING_CMODEL=ON -DUSING_DEBUG=ON ../..
make -j4
make install
#### then
cp -r PATH/of/build/install/tpuv7-runtime-emulator_0.1.0/* PATH/of/tpu-train/third_party/tpuv7_runtime/tpuv7-emulator_0.1.0/
```
注意：因为`tpuv7-emulator`在编译时会将TPU1686的cmodel版本的固件库打包，所以需要提前编译TPU1686的cmodel固件库。
