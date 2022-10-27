function build_all()
{
  should_clean=$1

  if [ -z "${TRAIN_BUILD_FOLDER}" ]; then
    TRAIN_BUILD_FOLDER=build
  fi
  if [ -n "$should_clean" ]; then
    rm -rf "$TRAIN_TOP/${TRAIN_BUILD_FOLDER}/"
    mkdir -p "$TRAIN_TOP/${TRAIN_BUILD_FOLDER}/"
  fi
  if [ ! -d "$TRAIN_TOP/${TRAIN_BUILD_FOLDER}/" ]; then
    echo "Couldn't find path $TRAIN_TOP/${TRAIN_BUILD_FOLDER}/"
    return -1
  fi
  pushd $TRAIN_TOP/${TRAIN_BUILD_FOLDER}/
  cores=$(($(nproc)-2))
  cmake .. ${TRAIN_BUILD_CONFIG} ${PLATFORM_BUILD_CONFIG}
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
