#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}")"/.. >/dev/null 2>&1 && pwd )"

# generate torch-tpu_quick_start
# ------------------------------------------------------------------------------
pushd ${DIR}/docs/quick_start
make clean
make pdf
popd

# generate torch-tpu_technical_manual
# ------------------------------------------------------------------------------
pushd ${DIR}/docs/developer_manual
make clean
make pdf
popd
# ------------------------------------------------------------------------------

mkdir -p ${DIR}/dist/docs
cp -rf ${DIR}/docs/quick_start/build_zh/torch-tpu_quick_start_zh.pdf \
   ${DIR}/dist/docs
cp -rf ${DIR}/docs/developer_manual/build_zh/torch-tpu_technical_manual_zh.pdf \
   ${DIR}/dist/docs
