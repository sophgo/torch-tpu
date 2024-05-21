source scripts/envsetup.sh sg2260 local
new_build

set_cmodel_firmware ./build/Release/firmware_core/libcmodel.so

#从TPU1686/common/include/拷贝头文件到tpu-train
cp /workspace/TPU1686/common/include/common_def.h /workspace/tpu-train/common/include/common_def.h

source scripts/sccl_envsetup.sh
rebuild_sccl