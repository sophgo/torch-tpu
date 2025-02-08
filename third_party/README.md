tpu-train的第三方库

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
### bmlib
Cmodel版本的BMlib第三方库。参考bmlib下readme。

### firmware
芯片的固件库。
- include 芯片算子开发头文件。
- bm1684x 芯片固件库。参考bm1684x路径下的readme。
- sg2260  芯片固件库。参考sg2260路径下的readme。

### tpuv7_runtime
tpuv7的运行时库。参考对应的readme

### oneDNN
sg2260 cmodel依赖库。

### tpuDNN
对于arm版本的tpudnn更新可以参考tpu1686的soc编译readme