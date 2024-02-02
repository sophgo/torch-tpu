### BMLIB

## commit id
f5e30d4e32e5accef7de39a7e21f66292f3417d9

## intro
该目录下是芯片设备控制相关接口的软件库-BMLIB。其中，
- bmlib_runtime.h 
- libbmlib.so cmodel版本的库文件。

注意: 
该目录下仅是CMODEL版本的相关软件库，真实芯片版本的需要通过deb包安装的方式，安装在/opt/libsophon下.

## how to update
该库的更新方式如下:
```shell
# step1. 进入libsophon工程目录下
cd </Path/of/libsophon>

# step2. 编译cmodel版本bmlib
mkdir -p build
cd build
cmake .. -DPLATFORM="cmodel"
make bmlib

cp bmlib/libbmlib.so.0 PATH/OF/tpu-train/third_party/bmlib/lib

#step3 建立libbmlib的软连接
cd PATH/OF/tpu-train/third_party/third_party/bmlib/lib
ln -s libbmlib.so.0 libbmlib.so
```
bmlib的编译，另外可以通过TPU1686工程下的相关脚本命令`rebuild_bmlib_cmodel`。

注意：更新库完成后，请记得更新对应commit id