bm1684x thirdparty 更新

```shell
# step0 使用nntc的docker（Ubuntu18.04）sophgo:tpuc_dev:latest

# step1. 进入TPU1686工程目录下
cd </Path/of/TPU1686>
# step2. 初始化环境
source scripts/envsetup.sh

# step3. 编译cmodel版本的芯片运行时库，bmlibcmodel.so
## 注意，在此之前，确保已经拉下libsophon的工程，
## 并保证libsophon工程位置位于</Path/of/TPU1686>/../../libsophon
rebuild_bmlib_cmodel
cp ../../libsophon/build/bmlib/libbmlib.so PATH/OF/tpu-train/third_party/bm1684x

# step4. 编译cmodel版本firmware
export EXTRA_CONFIG=-DDEBUG=ON
rebuild_firmware_cmodel
cp build/firmware_core/libcmodel_firmware.so PATH/OF/tpu-train/third_party/bm1684x

#step5. 编译板卡实际跑的fireware
rebuild_firmware
cp build/firmware_core/libfirmware_core.a PATH/OF/tpu-train/third_party/bm1684x/libbm1684x.a

#step6 建立libbmlib的软连接
cd PATH/OF/tpu-train/third_party/bm1684x
ln -s libbmlib.so libbmlib.so.0
```