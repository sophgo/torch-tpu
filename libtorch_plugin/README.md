1. Download libtorch cpu version 2.0.1 compatiable with ubuntu22.04 or sophgo/tpuc_dev:latest

wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.0.1%2Bcpu.zip

2. Unzip

unzip libtorch-shared-with-deps-xxxx.zip

3. Compile libtorch_plugin

mkdir build && cd build

cmake ..

make -j$(($(nproc)-2)) -DCMAKE_BUILD_TYPE=Debug
