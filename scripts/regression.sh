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
    $cmd_utest; cmd_utest_result=$?
    echo "[INFO]cmd_utest_result:$cmd_utest_result"
    popd
    echo "*********** UTEST ENDED ***************"
    return $cmd_utest_result
}


export stable_libsophon_path="/workspace/libsophon_Release_20230605_025400"
# function for online regression
function link_libsophon() {
    echo "********************************************"
    echo "[STEP]install libsophon"
    CURRENT_DIR=$(dirname ${BASH_SOURCE})
    LIBSOPHON_LINK_PATTERN=${1:-latest}
    DEB_PATH_STABLE=${2:-none}
    VERSION_PATH_STABLE=${3:-0.4.8}
    if [ $LIBSOPHON_LINK_PATTERN = 'stable' ]; then
      echo "[NOTE]STABLE LIBSOHON IS ADAPATED FROM .deb"
      if [ $DEB_PATH_STABLE = 'none' ]; then
        echo "You are giving wrong libsophon .deb upper-path!"
        return 255
      else
        echo "[INFO]LIBSOPHON_PATH_TPU_TRAIN:$DEB_PATH_STABLE"
        pushd "$DEB_PATH_STABLE"
        if [  ! -r "$DEB_PATH_STABLE" ]; then
          echo "libsophon_dependency: $DEB_PATH_STABLE is not found!"
        else
          libsohpon_install_cmd="apt install  ./sophon-libsophon_${VERSION_PATH_STABLE}_amd64.deb ./sophon-libsophon-dev_${VERSION_PATH_STABLE}_amd64.deb"
          $libsohpon_install_cmd
          device_project="source /etc/profile.d/libsophon-bin-path.sh"
          $device_project
        fi
        popd
      fi
    else
      echo "LATEST LIBSOHON IS ADAPATED from libsophon-file"
    fi
    echo "********************************************"
}

function build_libtorch_plugin() {
  CURRENT_DIR=$(dirname ${BASH_SOURCE})
  pushd  $CURRENT_DIR/..
  cmd_rm="rm -rf libtorch"
  $cmd_rm
  get_libtorch="wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.0.1%2Bcpu.zip"
  $get_libtorch
  unzip_code="unzip libtorch-cxx11-abi-shared-with-deps-2.0.1+cpu.zip"
  $unzip_code
  popd

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
  echo "[INFO]Ubuntu version"
  version_cmd="cat /etc/os-release"
  $version_cmd
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
  DEB_PATH_STABLE=${4:-none} #none or given path
  VERSION_PATH_STABLE=${5:-0.4.8}

  echo "[INFO]test_CHIP_ARCH:$test_CHIP_ARCH"
  echo "[INFO]LIBSOPHON_LINK_PATTERN=$LIBSOPHON_LINK_PATTERN"

  link_libsophon $LIBSOPHON_LINK_PATTERN $DEB_PATH_STABLE $VERSION_PATH_STABLE; ret_libsophon=$?
  if [ $ret_libsophon -eq 255 ]; then
    echo "[INFO]test_CHIP_ARCH:$test_CHIP_ARCH libsophon setting failed!"
  else
    if [ $LIBSOPHON_LINK_PATTERN = 'stable' ];then
      echo "[INFO]test_CHIP_ARCH:$test_CHIP_ARCH"
      export CROSS_TOOLCHAINS="$CURRENT_DIR/../../bm_prebuilt_toolchains/"
      if [ ! -d $CROSS_TOOLCHAINS ]; then
        echo "[bm_prebuilt_toolchains]:$CROSS_TOOLCHAINS is not found !"
      else
        dumpinstall="apt-get install bsdmainutils"
        $dumpinstall
        pushd $CURRENT_DIR/..
        rm -rf build
        mkdir build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Debug -DUSING_CMODEL=OFF -DPCIE_MODE=ON
        make kernel_module
        make -j
        popd
      fi

    elif [ $LIBSOPHON_LINK_PATTERN = 'latest' ];then
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
      echo "[INFO]ret_ops_utest:$ret_ops_utest"
      if [ $ret_ops_utest -eq 0 ];then #must return [0,255] otherwise it will cause scripts fault early
        echo "[RESULT-$test_CHIP_ARCH] all ops_utest are computed, Please check Results above"
      else
        echo "[RESULT-$test_CHIP_ARCH] some ops_utest are failed!"
      fi
    fi
  fi
}


function fast_build_bm1684x_stable() {
  DEB_PATH_STABLE=${1:-none}
  run_online_regression_test bm1684x stable fast $DEB_PATH_STABLE 0.4.8
}

function fast_build_bm1684x_latest() {
  run_online_regression_test bm1684x latest fast
}

function fast_build_sg2260_latest() {
  run_online_regression_test sg2260 latest fast
}