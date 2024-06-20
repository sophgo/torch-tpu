# function build_all()
# {
#   should_clean=$1

#   if [ -z "${TRAIN_BUILD_FOLDER}" ]; then
#     TRAIN_BUILD_FOLDER=build
#   fi
#   if [ -n "$should_clean" ]; then
#     rm -rf "$TPUTRAIN_TOP/${TRAIN_BUILD_FOLDER}/"
#     mkdir -p "$TPUTRAIN_TOP/${TRAIN_BUILD_FOLDER}/"
#   fi
#   if [ ! -d "$TPUTRAIN_TOP/${TRAIN_BUILD_FOLDER}/" ]; then
#     echo "Couldn't find path $TPUTRAIN_TOP/${TRAIN_BUILD_FOLDER}/"
#     return -1
#   fi
#   pushd $TPUTRAIN_TOP/${TRAIN_BUILD_FOLDER}/
#   cores=$(($(nproc)-2))
#   cmake .. -DCMAKE_BUILD_TYPE=Debug -DUSING_CMODEL=ON -DPCIE_MODE=OFF -DSOC_MODE=OFF
#   make -j$cores
#   ret=$?
#   popd
#   if [ $ret -ne 0 ]; then return $ret; fi
# }

# function rebuild_all()
# {
#   build_all 1
# }

export TRAIN_BUILD_FOLDER=build

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
  rm -rf build
  rm -rf dist
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
