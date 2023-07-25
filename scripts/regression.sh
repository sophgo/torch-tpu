#!/bin/bash
#[NOTE] very important and serious INFO
#[INFO] echo information
#[STEP] when some procedures are started
#[RESULT] the results of some procedures

function ops_utest() {
    CURRENT_DIR=$(dirname ${BASH_SOURCE})
    PYTHON_UTEST_PATH=$CURRENT_DIR/../python/utest_ops
    echo "[INFO]PYTHON_UTEST_PATH:$PYTHON_UTEST_PATH"
    pushd $PYTHON_UTEST_PATH
    cmd_utest="python3 utest_cmd.py"
    $cmd_utest
    popd
    echo "*********** UTEST ENDED ***************"
}


export stable_libsophon_path="sophon-libsophon_0.4.6_amd64.deb"
# function for online regression
function link_libsophon() {
    echo "********************************************"
    echo "[STEP]install libsophon"
    CURRENT_DIR=$(dirname ${BASH_SOURCE})
    LIBSOPHON_LINK_PATTERN=${1:-latest}
    if [ $LIBSOPHON_LINK_PATTERN = 'stable' ]; then
      echo "[NOTE]STABLE LIBSOHON IS ADAPATED FROM .deb"
      LIBSOPHON_PATH_TPU_TRAIN=$CURRENT_DIR/../../libsophon_install
      echo "[INFO]LIBSOPHON_PATH_TPU_TRAIN:$LIBSOPHON_PATH_TPU_TRAIN"
      pushd "$LIBSOPHON_PATH_TPU_TRAIN"
      if [  ! -r "$stable_libsophon_path" ]; then
        echo "libsophon_dependency $stable_libsophon_path is not found!"
        exit 1
      else
        libsohpon_install_cmd="apt install  ./sophon-libsophon_0.4.6_amd64.deb ./sophon-libsophon-dev_0.4.6_amd64.deb"
        $libsohpon_install_cmd
      fi
      popd
    else
      echo "LATEST LIBSOHON IS ADAPATED from libsophon-file"
    fi
    echo "********************************************"
}

function build_libtorch_plugin() {
  CURRENT_DIR=$(dirname ${BASH_SOURCE})
  LIBTORCH_PLUGIN_PATH=$CURRENT_DIR/../libtorch_plugin
  echo "[INFO]LIBTORCH_PLUGIN_PATH:$LIBTORCH_PLUGIN_PATH"
  pushd "$LIBTORCH_PLUGIN_PATH"
  if [ ! -f "./build" ]; then
    rm -rf ./build
  fi
  mkdir build&&cd build
  cmake .. -DCMAKE_BUILD_TYPE=Debug
  make -j && cd ..
  popd
}

function run_online_regression_test() {
  CURRENT_DIR=$(dirname ${BASH_SOURCE})
  echo "********************************************"
  echo "[NOTE]Print_necessary_info"
  echo "[INFO]CURRENT_DIR:$CURRENT_DIR"
#   TEST_DIR=$CURRENT_DIR/../tpu_train_regression_test/
#   echo "[INFO]TEST_DIR:$TEST_DIR"
  # if [ ! -f "$TEST_DIR" ]; then
  #   mkdir $TEST_DIR
  # fi
  test_CHIP_ARCH=${1:-bm1684x}
  LIBSOPHON_LINK_PATTERN=${2:-latest} #latest or stable
  TEST_PATTERN=${3:-normal} #normal or fast
  echo "[INFO]test_CHIP_ARCH:$test_CHIP_ARCH"
  echo "[INFO]LIBSOPHON_LINK_PATTERN=$LIBSOPHON_LINK_PATTERN"

  link_libsophon $LIBSOPHON_LINK_PATTERN; ret_libsophon=$?
  if [ $ret_libsophon -eq 1 ]; then
    exit 1
  else
    echo "************** $LIBSOPHON_LINK_PATTERN-LIBSOPHON IS REAEDY *********"
    source  $CURRENT_DIR/envsetup.sh $test_CHIP_ARCH
    rebuild_all
    TPU_TRAIN_CMODEL_PATH=$CURRENT_DIR/../build/firmware_core/libcmodel.so
    echo "[INFO]tpu_train_cmodel_path:$TPU_TRAIN_CMODEL_PATH"
    set_cmodel_firmware $TPU_TRAIN_CMODEL_PATH
    echo "*************** CMODEL IS SET *************"
  fi
  if [ $TEST_PATTERN = "normal" ];then
    build_libtorch_plugin
    echo "*************** LIBTORCH_PLUGIN IS BUILT *************"
    ops_utest; ret_ops_utest=$?
    if [ $ret_ops_utest -ne 1 ];then
      echo "[RESULT-$test_CHIP_ARCH] all ops_utest are computed, Please check Results above"
    else
      echo "[RESULT-$test_CHIP_ARCH] some ops_utest are failed!"
    fi
  fi
}
