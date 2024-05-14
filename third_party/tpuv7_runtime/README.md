### tpuv7 runtime

## commit id
- tpuv7_runtime
feea0f1577793b9f974c9dd8b536ea9d67332b6f
- TPU1686
1607fc8ad38c6750aaceaceae1b0db0b4bb2ed74

## intro
`tpuv7_runtime` 是TPUv7.0的异步运行时库。其中，

- `tpuv7-emulator` 是cmodel版本的device runtime, bmodel runtime and firmware.
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
- - - libtpuv7_emulator.so : tpu's runtime and firmware emulator.
- - - libtpuv7_modelrt.so : bmodel's runtime.
- - - libtpuv7_rt.so : runtime api lib for users.


- `tpuv7_0.1.0` 芯片ASIC版本的device runtime， bmodel runtime。
包含的工具和内容与上述类似。

- `sg2260_firmware` 是芯片ASIC版本的firmware。


## how to update

为了cmodel与实际芯片保持一致。更新该库文件时，请注意需要同时更新cmodel版本（即`tpuv7-emulator`）和 芯片版本（即`tpuv7_0.1.0`和`sg2260_firmware`）

更新 `tpuv7-emulator`
----
  1) `git clone ssh://yu.hu@gerrit-ai.sophgo.vip:29418/tpuv7-runtime`(tpuv7-runtime and TPU1686 should in same folder)
  2) compile `libtpuv7_emulator.so`, which is the cmodel of chip firmware.
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
cmake -DCMAKE_INSTALL_PREFIX=$PWD/../install  -DUSING_CMODEL=ON -DCMAKE_BUILD_TYPE=Debug ../..
make -j4
make install
#### then
cp -r PATH/of/build/install/tpuv7-emulator_0.1.0 .
```
注意：因为`tpuv7-emulator`在编译时会将TPU1686的cmodel版本的固件库打包，所以需要提前编译TPU1686的cmodel固件库。

更新`tpuv7_0.1.0`
----
refer to tpuv7-runtime/README generate asic version `tpuv7_0.1.0`.
```shell
	mkdir build && cd build
	mkdir asic && cd asic
	cmake -DCMAKE_INSTALL_PREFIX=$PWD/../install  -DUSING_CMODEL=OFF  ../..
	make -j4
	make install
### then
    cp -r PATH/of/build/install/tpuv7_0.1.0 .
```


更新`sg2260_firmware`
----
现在暂时需要 modify some cmakelist.txt of TPU1686.
Can refer to commit id: Ia2863ed631fbc266b5c844e533d6ad801f491d32.
```shell
$ source scripts/envsetup.sh sg2260
$ rebuild_firmware
$ cp build/firmware_core/libfirmware_core.so path/of/sg2260_firmware
```

注意：更新库完成后，请记得更新对应commit id。由于芯片固件库来自于TPU1686工程，所以需要同时更新两个库的id。