 #!/bin/bash
echo $PWD

cd /workspace/tpu-train/
source scripts/envsetup.sh

if ls "/workspace/tpu-train/libtorch_plugin/build" &>/dev/null
then
    cd /workspace/tpu-train/libtorch_plugin/build
    cmake ..
    make -j
else
    mkdir /workspace/tpu-train/libtorch_plugin/build
    cd /workspace/tpu-train/libtorch_plugin/build
    cmake ..
    make -j
fi

cd ../../
rebuild_all
sleep 1s

set_cmodel_firmware /workspace/tpu-train/build/firmware_core/libcmodel.so
