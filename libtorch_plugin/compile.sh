#!/bin/bash
g++ test.cpp \
tpu/TPUGuardImpl.cpp \
tpu/TPUAllocator.cpp \
tpu/TPUDeviceManager.cpp \
tpu/TPUModule.cpp \
ops/*.cpp \
-I/opt/sophon/libsophon-current/include/ \
-I../../libtorch/include/torch/csrc/api/include/ \
-I../../libtorch/include/ \
-I. \
-I./tpu \
-I../../ \
-D_GLIBCXX_USE_CXX11_ABI=0 \
-L../../ \
-L../../libtorch/lib \
-L/opt/sophon/libsophon-current/lib \
-lc10 \
-ltorch_cpu \
-lbmlib \
-lsgdnn
