"torch-tpu_src.tar.gz”中包含了torch-tpu的python-binding代码和库文件，用户可以自己编译torch-tpu.whl包。

环境依赖：
 - cmake
 - g++/gcc
 - python3-dev
 - patchelf

安装方式：
```shell
apt install cmake \
            g++ \
            python3-dev \
            patchelf
```

使用方法：
解压后，执行下面命令
```shell
cd torch_tpu
python setup.py bdist_wheel
pip install dist/*.whl
```