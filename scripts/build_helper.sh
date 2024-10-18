#!/bin/bash

function new_build()
{
  pip uninstall torch_tpu -y
  pushd ${TPUTRAIN_TOP}
  python setup.py develop
  if [ $? -ne 0 ]; then popd; return -1; fi
  popd
}

function bdist_wheel()
{
  python setup.py develop --uninstall
  pushd ${TPUTRAIN_TOP}
  python setup.py bdist_wheel
  if [ $? -ne 0 ]; then popd; return -1; fi
  pip install dist/* --force-reinstall
  popd
}

function new_clean()
{
  pushd ${TPUTRAIN_TOP}
  echo " - Delete build ..."
  rm -rf build
  echo " - Delete dist ..."
  rm -rf dist
  echo " - Delete torch_tpu.egg-info ..."
  rm -rf torch_tpu.egg-info
  echo " - Delete torch_tpu/lib ..."
  rm -rf torch_tpu/lib
  echo " - Delete torch_tpu/_C.cpython-310-x86_64-linux-gnu.so ..."
  rm -rf torch_tpu/_C.cpython-310-x86_64-linux-gnu.so
  popd
}

function soc_build()
{
  pushd ${TPUTRAIN_TOP}
  export SOC_CROSS_MODE=ON
  python setup.py build bdist_wheel --plat-name=aarch64
  unset SOC_CROSS_MODE
  popd
}

function update_sg2260_third_party()
{
  pushd ${TPUTRAIN_TOP}
  echo "update libtpudnn.so ..."
  cp ../TPU1686/build_sg2260/tpuDNN/src/libtpudnn.so third_party/tpuDNN/sg2260_lib/
  echo "update libtpuv7_emulator.so ..."
  cp ../TPU1686/build_sg2260/firmware_core/libtpuv7_emulator.so ./third_party/tpuv7_runtime/tpuv7-emulator_0.1.0/lib/
  popd
}