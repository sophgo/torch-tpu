### bm1684x

## commit id
1c4644118773ed8294afb37310c654805d12a3d2

## intro
该目录下是BM1684x的Firmware软件库文件。其中，
- libbm1684x.a 是实际芯片ASIC的firmware。
- libcmodel_firmware.so 是Cmodel版本的firmware.    

## how to update

该库的更新方式如下:
```shell
# step1. 进入TPU1686工程目录下
cd </Path/of/TPU1686>

# step2. 初始化环境
source scripts/envsetup.sh

# step3. 编译cmodel版本firmware
export EXTRA_CONFIG=-DDEBUG=ON
rebuild_firmware_cmodel
cp build/firmware_core/libcmodel_firmware.so PATH/OF/tpu-train/third_party/bm1684x

#step4. 编译板卡实际跑的firmware
rebuild_firmware
cp build/firmware_core/libfirmware_core.a PATH/OF/tpu-train/third_party/bm1684x/libbm1684x.a
```

注意：更新库完成后，请记得更新对应commit id