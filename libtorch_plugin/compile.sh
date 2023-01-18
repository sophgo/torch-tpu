#!/bin/bash
g++ test.cpp \
TPUGuardImpl.cpp \
TPUAllocator.cpp \
TPUDeviceManager.cpp \
ops/*.cpp \
-I/opt/sophon/libsophon-current/include/ \
-I../libtorch/include/torch/csrc/api/include/ \
-I../libtorch/include/ \
-I. \
-D_GLIBCXX_USE_CXX11_ABI=0 \
-L../libtorch/lib \
-L/opt/sophon/libsophon-current/lib \
-lc10 \
-ltorch_cpu \
-lbmlib
