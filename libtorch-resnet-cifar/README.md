#https://github.com/leimao/LibTorch-ResNet-CIFAR
#https://blog.csdn.net/weixin_44966641/article/details/122070241

# LibTorch C++ ResNet CIFAR Example

## Introduction

ResNet implementation, training, and inference using LibTorch C++ API. The saved model from LibTorch C++ API cannot be used for PyTorch Python API and vice versa. LibTorch C++ API is not as rich as PyTorch Python API and its implementation really takes way too much time. The performance benefits that LibTorch C++ API brought is almost negligible over PyTorch Python API.

## Usages

### Download Dataset

```
$ cd dataset/
$ bash download-cifar10-binary.sh
```

### Build Application

```
$ cmake -B build -DCMAKE_PREFIX_PATH=`python -c 'import torch;print(torch.utils.cmake_prefix_path)'`
$ or cmake -B build -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch
$ cmake --build build --config Release
```

### Run Application

```
$ cd build/src/
$ ./resnet-cifar-demo
```
