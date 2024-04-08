#!/bin/bash
#[NOTE] very important and serious INFO
#[INFO] echo information
#[STEP] when some procedures are started
#[RESULT] the results of some procedures
#[ERROR]  some params are wrong or files not exist

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

function gpt3block_test() {
    start_time=$(date +%s) 
    CURRENT_DIR=$(dirname ${BASH_SOURCE})
    PYTHON_TEST_PATH=$CURRENT_DIR/../python/gen_ins/
    echo "[INFO]PYTHON_TEST_PATH:$PYTHON_TEST_PATH"
    pushd $PYTHON_TEST_PATH
    export FORBID_CMD_EXECUTE=0
    export CMODEL_FAST_EXEC=1
    cmd_gpt3block_test="python3 gpt3block_TP16_fb.py"
    $cmd_gpt3block_test; cmd_gpt3block_test_result=$?
    echo "[INFO]cmd_gpt3block_test_result:$cmd_gpt3block_test_result"
    popd
    echo "*********** GPT3BLOCK TEST ENDED ***************"
    end_time=$(date +%s)
    echo "Time elapsed: $(($end_time - $start_time)) seconds"
    return $cmd_gpt3block_test_result
}

export stable_libsophon_path="/workspace/libsophon_Release_20230605_025400"
# function for online regression
function link_libsophon() {
    echo "********************************************"
    echo "[STEP]install libsophon"
    CURRENT_DIR=$(dirname ${BASH_SOURCE})
    LIBSOPHON_LINK_PATTERN=${1:-local}
    DEB_PATH_STABLE=${2:-none}
    VERSION_PATH_STABLE=${3:-0.4.8}
    if [ $LIBSOPHON_LINK_PATTERN = 'stable' ]; then
      echo "[NOTE]STABLE LIBSOHON IS ADAPATED FROM .deb"
      if [ $DEB_PATH_STABLE = 'none' ]; then
        echo "[ERROR]Wrong libsophon .deb upper-path!"
        return 255
      else
        echo "[INFO]LIBSOPHON_PATH_TPU_TRAIN:$DEB_PATH_STABLE"
        pushd "$DEB_PATH_STABLE"
        if [  ! -r "$DEB_PATH_STABLE" ]; then
          echo "[ERROR]libsophon_dependency: $DEB_PATH_STABLE is not found!"
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

function make_kernel_module() {
  test_CHIP_ARCH=${1:-bm1684x}
  CURRENT_DIR=$(dirname ${BASH_SOURCE})
  if [ "${test_CHIP_ARCH}" = "sg2260" ]; then
      echo "chip 2260 kernel_module building is 'invalid'"
  else
    dumpinstall="apt-get install bsdmainutils"
    $dumpinstall
    pushd $CURRENT_DIR/..
    rm -rf build
    mkdir build && cd build
    cmake_cmd_kerenl="cmake .. -DCMAKE_BUILD_TYPE=Debug -DUSING_CMODEL=OFF -DPCIE_MODE=ON -DSOC_MODE=OFF"
    echo "[CMD-INFO] $cmake_cmd_kerenl"
    $cmake_cmd_kerenl
    make kernel_module
    make -j$(($(nproc)-2))
    popd
  fi
}

#this function is fake  as make -j exists error
function build_kernel_module() {
  CURRENT_DIR=$(dirname ${BASH_SOURCE})
  export CROSS_TOOLCHAINS="$CURRENT_DIR/../../bm_prebuilt_toolchains/"
  test_CHIP_ARCH=${1:-bm1684x}
  if [ ! -d $CROSS_TOOLCHAINS ]; then
    echo "[bm_prebuilt_toolchains]:$CROSS_TOOLCHAINS is not found !"
  else
 if [ "${test_CHIP_ARCH}" = "sg2260" ]; then
      echo "chip 2260 kernel_module building is 'invalid'"
  else
    dumpinstall="apt-get install bsdmainutils"
    $dumpinstall
    pushd $CURRENT_DIR/..
    rm -rf build
    mkdir build && cd build
    cmake_cmd_kerenl="cmake .. -DCMAKE_BUILD_TYPE=Debug -DUSING_CMODEL=OFF -DPCIE_MODE=ON -DSOC_MODE=OFF"
    echo "[CMD-INFO] $cmake_cmd_kerenl"
    $cmake_cmd_kerenl
    make kernel_module
    #make -j$(($(nproc)-2)) #the only difference!!!!!!!!!!!!!
    popd
  fi
  fi
}

function build_kernel_module_real() {
  CURRENT_DIR=$(dirname ${BASH_SOURCE})
  export CROSS_TOOLCHAINS="$CURRENT_DIR/../../bm_prebuilt_toolchains/"
  test_CHIP_ARCH=${1:-bm1684x}
  if [ ! -d $CROSS_TOOLCHAINS ]; then
    echo "[bm_prebuilt_toolchains]:$CROSS_TOOLCHAINS is not found !"
  else
    make_kernel_module $test_CHIP_ARCH
  fi
}

function run_online_regression_test() {
  echo "[INFO]Ubuntu version"
  version_cmd="cat /etc/os-release"
  $version_cmd
  CURRENT_DIR=$(dirname ${BASH_SOURCE})
  echo "********************************************"
  echo "[NOTE]Print_necessary_info"
  echo "[INFO]CURRENT_DIR:$CURRENT_DIR"
  
  test_CHIP_ARCH=${1:-bm1684x}
  LIBSOPHON_LINK_PATTERN=${2:-local} #local or stable
  TEST_PATTERN=${3:-online} #online,local, or fast
  DEB_PATH_STABLE=${4:-none} #none or given path
  VERSION_PATH_STABLE=${5:-0.4.8}

  echo "[INFO]test_CHIP_ARCH:$test_CHIP_ARCH"
  echo "[INFO]LIBSOPHON_LINK_PATTERN=$LIBSOPHON_LINK_PATTERN"
  echo "[INFO]TEST_PATTERN=$TEST_PATTERN"
  link_libsophon $LIBSOPHON_LINK_PATTERN $DEB_PATH_STABLE $VERSION_PATH_STABLE; ret_libsophon=$?
  if [ $ret_libsophon -eq 255 ]; then
    echo "[INFO]test_CHIP_ARCH:$test_CHIP_ARCH libsophon setting failed!"
  else
    if [ $LIBSOPHON_LINK_PATTERN = 'stable' ];then
      echo "[INFO]test_CHIP_ARCH:$test_CHIP_ARCH"
      source  $CURRENT_DIR/envsetup.sh $test_CHIP_ARCH $LIBSOPHON_LINK_PATTERN
      new_clean;new_build
    elif [ $LIBSOPHON_LINK_PATTERN = 'local' ];then
      echo "************** $LIBSOPHON_LINK_PATTERN-LIBSOPHON IS REAEDY *********"
      source  $CURRENT_DIR/envsetup.sh $test_CHIP_ARCH $LIBSOPHON_LINK_PATTERN
      new_clean;new_build
      TPU_TRAIN_CMODEL_PATH=$CURRENT_DIR/../build/Release/firmware_core/libcmodel.so
      echo "[INFO]tpu_train_cmodel_path:$TPU_TRAIN_CMODEL_PATH"
      set_cmodel_firmware $TPU_TRAIN_CMODEL_PATH
      echo "*************** CMODEL IS SET *************"
    fi
    if [ $TEST_PATTERN = "online" ] || [ $TEST_PATTERN = "local" ];then
      # build_libtorch_plugin $TEST_PATTERN
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

function run_daily_regression_test() {
  echo "[INFO]Ubuntu version"
  version_cmd="cat /etc/os-release"
  $version_cmd
  CURRENT_DIR=$(dirname ${BASH_SOURCE})
  echo "********************************************"
  echo "[NOTE]Print_necessary_info"
  echo "[INFO]CURRENT_DIR:$CURRENT_DIR"
  
  
  test_CHIP_ARCH=${1:-bm1684x}
  LIBSOPHON_LINK_PATTERN=${2:-local} #local or stable
  TEST_PATTERN=${3:-online} #online,local, or fast
  DEB_PATH_STABLE=${4:-none} #none or given path
  VERSION_PATH_STABLE=${5:-0.4.8}

  echo "[INFO]test_CHIP_ARCH:$test_CHIP_ARCH"
  echo "[INFO]LIBSOPHON_LINK_PATTERN=$LIBSOPHON_LINK_PATTERN"
  echo "[INFO]TEST_PATTERN=$TEST_PATTERN"
  link_libsophon $LIBSOPHON_LINK_PATTERN $DEB_PATH_STABLE $VERSION_PATH_STABLE; ret_libsophon=$?
  if [ $ret_libsophon -eq 255 ]; then
    echo "[INFO]test_CHIP_ARCH:$test_CHIP_ARCH libsophon setting failed!"
  else
    if [ $LIBSOPHON_LINK_PATTERN = 'stable' ];then
      echo "[INFO]test_CHIP_ARCH:$test_CHIP_ARCH"
      build_kernel_module_real $test_CHIP_ARCH
    elif [ $LIBSOPHON_LINK_PATTERN = 'local' ];then
      echo "************** $LIBSOPHON_LINK_PATTERN-LIBSOPHON IS REAEDY *********"
      source  $CURRENT_DIR/envsetup.sh $test_CHIP_ARCH $LIBSOPHON_LINK_PATTERN
      new_clean; new_build
      TPU_TRAIN_CMODEL_PATH=$CURRENT_DIR/../build/Release/firmware_core/libcmodel.so
      echo "[INFO]tpu_train_cmodel_path:$TPU_TRAIN_CMODEL_PATH"
      set_cmodel_firmware $TPU_TRAIN_CMODEL_PATH
      echo "*************** CMODEL IS SET *************"
    fi
    if [ $TEST_PATTERN = "online" ] || [ $TEST_PATTERN = "local" ];then
      # build_libtorch_plugin $TEST_PATTERN
      echo "*************** LIBTORCH_PLUGIN IS BUILT *************"
      ops_utest; ret_ops_utest=$?
      echo "[INFO]ret_ops_utest:$ret_ops_utest"
      if [ $ret_ops_utest -eq 0 ];then #must return [0,255] otherwise it will cause scripts fault early
        echo "[RESULT-$test_CHIP_ARCH] all ops_utest are computed, Please check Results above"
      else
        echo "[RESULT-$test_CHIP_ARCH] some ops_utest are failed!"
      fi
      if [ "${test_CHIP_ARCH}" = "sg2260" ]; then
        gpt3block_test; ret_gpt3block_test=$?
        echo "[INFO]ret_gpt3block_test:$ret_gpt3block_test"
        if [ $ret_gpt3block_test -eq 0 ];then #must return [0,255] otherwise it will cause scripts fault early
          echo "[RESULT-$test_CHIP_ARCH] all gpt3block results are computed, Please check Results above"
        else
          echo "[RESULT-$test_CHIP_ARCH] some gpt3block results are failed!"
        fi
      fi
    fi
  fi
}

function fast_build_bm1684x_stable() {
  DEB_PATH_STABLE=${1:-none}
  run_online_regression_test bm1684x stable fast $DEB_PATH_STABLE 0.4.8
}

function fast_build_bm1684x_latest() {
  run_online_regression_test bm1684x local fast
}

function fast_build_bm1684x_latest_and_libtorch_plugin() {
  run_online_regression_test bm1684x local fast
}

function fast_build_sg2260_latest() {
  run_online_regression_test sg2260 local fast
}

function fast_build_sg2260_latest_and_libtorch_plugin() {
  run_online_regression_test sg2260 local fast
}
