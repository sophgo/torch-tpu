#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd )"
python docs/generate_operation.py ${DIR}/torch_tpu/csrc/ops/native_ops ${DIR}/torch_tpu/csrc/ops/my_ops/myOps.cpp ${DIR}/build/supported_ops.rst

# generate torch-tpu_quick_start
# ------------------------------------------------------------------------------
pushd ${DIR}/docs/quick_start
make clean
make pdf || exit -1
popd

# generate torch-tpu_technical_manual
# ------------------------------------------------------------------------------
# # generate doxygen document for custom ops
# apt-get update && apt-get install -y doxygen || exit -1
pip install breathe || exit -1
doxygen config.doxy || exit -1

pushd ${DIR}/docs/developer_manual
make clean
make pdf || exit -1
popd
# ------------------------------------------------------------------------------

mkdir -p ${DIR}/dist/docs
cp -rf ${DIR}/docs/quick_start/build_zh/torch-tpu_quick_start_zh.pdf \
   ${DIR}/dist/docs/"TORCH-TPU快速入门指南.pdf"
cp -rf ${DIR}/docs/developer_manual/build_zh/torch-tpu_technical_manual_zh.pdf \
   ${DIR}/dist/docs/"TORCH-TPU开发参考手册.pdf"
