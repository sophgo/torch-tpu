function build_all()
{
  should_clean=$1

  if [ -z "${TRAIN_BUILD_FOLDER}" ]; then
    TRAIN_BUILD_FOLDER=build
  fi
  if [ -n "$should_clean" ]; then
    rm -rf "$TPUTRAIN_TOP/${TRAIN_BUILD_FOLDER}/"
    mkdir -p "$TPUTRAIN_TOP/${TRAIN_BUILD_FOLDER}/"
  fi
  if [ ! -d "$TPUTRAIN_TOP/${TRAIN_BUILD_FOLDER}/" ]; then
    echo "Couldn't find path $TPUTRAIN_TOP/${TRAIN_BUILD_FOLDER}/"
    return -1
  fi
  pushd $TPUTRAIN_TOP/${TRAIN_BUILD_FOLDER}/
  cores=$(($(nproc)-2))
  cmake .. -DCMAKE_BUILD_TYPE=Debug -DUSING_CMODEL=ON -DPCIE_MODE=OFF -DSOC_MODE=OFF
  make -j$cores
  ret=$?
  popd
  if [ $ret -ne 0 ]; then return $ret; fi
}

function rebuild_all()
{
  build_all 1
}

export TRAIN_BUILD_FOLDER=build

function new_build()
{
  pushd ${TPUTRAIN_TOP}
  python setup.py build bdist_wheel
  pip install dist/torch_tpu-2.0.1.post1-cp39-cp39-linux_x86_64.whl --force-reinstall
  popd
}
function new_clean()
{
  pushd ${TPUTRAIN_TOP}
  rm -rf build
  rm -rf dist
  popd
}