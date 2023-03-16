1. Download libtorch cpu version 1.13

wget https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-1.13.1%2Bcpu.zip

2. Unzip

unzip libtorch-shared-with-deps-xxxx.zip

3. Compile libtorch_plugin

mkdir build && cd build

cmake ..

make -j
