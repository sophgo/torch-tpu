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
  cmake ..
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
