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

function test_sccl() {
    local NODES=8
    export CHIP_MAP=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
    if [ -n "$2" ]; then
      NODES=$2
    fi
    if [ ! -n "$1" ]; then
        echo "Usage: test_sccl <test_name> [nodes]"
        echo "test case is empty, please choose one from:"
        echo "test_all_gather.py"
        echo "test_all_reduce.py"
        echo "test_broadcast.py"
        echo "test_gather.py"
        echo "test_reduce.py"
        echo "test_scatter.py"
        echo "test_all_to_all.py"
        echo "test_send.py"
        return
    fi
    TEST_DIR=$TPUTRAIN_TOP/python/dist_test2260/
    pushd ${TEST_DIR} > /dev/null
    echo "test dir: $TEST_DIR"

    echo "################################"
    echo "Test sccl with torchrun, NODES=$NODES"
    echo "################################"

    ret=0
    torchrun --nproc_per_node ${NODES} --nnodes 1 $TEST_DIR/$1 || ret=1
    if [ $ret -ne 0 ]; then
      echo "test sccl case: $1 failed"
      popd
      return $ret
    fi
    unset CHIP_MAP
    popd
}

function regression_for_sccl() {
    ulimit -n 65535
    local_cases=(
        test_all_gather.py
        test_all_reduce.py
        test_broadcast.py
        test_gather.py
        test_reduce.py
        test_scatter.py
        test_all_to_all.py
        test_send.py
    )
    for case in ${local_cases[@]}; do
        test_sccl $case $1
        if [ $? -ne 0 ]; then
            return 1
        fi
        echo "sleep 15s to release socket resources(TIME_WAIT limit)... maybe have good methods"
        sleep 15
    done
}

function regression_for_tgi() {
  TEST_DIR=$TPUTRAIN_TOP/python/tgi_test/
  pushd ${TEST_DIR} > /dev/null
  echo "test dir: $TEST_DIR"
  echo "################################"
  echo "Test tgi op"
  echo "################################"
  ret=0
  pip install -i https://pypi.tuna.tsinghua.edu.cn/simple loguru
  python3 tgi_op.py
  if [ $ret -ne 0 ]; then
    echo "test tgi cases: $1 failed"
    popd
    return $ret
  fi
  popd
}

function test_scclHost() {
    local NODES=8
    if [ -n "$2" ]; then
      NODES=$2
    fi
    if [ ! -n "$1" ]; then
        echo "Usage: test_scclHost <test_name> [nodes]"
        echo "test case is empty, please choose one from:"
        echo "all_gather.py"
        echo "all_reduce.py"
        echo "broadcast.py"
        echo "gather.py"
        echo "reduce.py"
        echo "scatter.py"
        echo "all_to_all.py"
        return
    fi
    TEST_DIR=$TPUTRAIN_TOP/python/dist_test/
    pushd ${TEST_DIR} > /dev/null
    echo "test dir: $TEST_DIR"

    echo "################################"
    echo "Test scclHost with torchrun, NODES=$NODES"
    echo "################################"

    ret=0
    torchrun --nproc_per_node ${NODES} --nnodes 1 $TEST_DIR/$1 || ret=1
    if [ $ret -ne 0 ]; then
      echo "test scclHost case: $1 failed"
      popd
      return $ret
    fi
    popd
}

function regression_for_scclHost() {
    local_cases=(
        all_gather.py
        all_reduce.py
        broadcast.py
        gather.py
        reduce.py
        scatter.py
        all_to_all.py
    )
    for case in ${local_cases[@]}; do
        test_scclHost $case $1
        if [ $? -ne 0 ]; then
            return 1
        fi
    done
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

function build_kernel_module() {
  CURRENT_DIR=$(dirname ${BASH_SOURCE})
  export CROSS_TOOLCHAINS="$CURRENT_DIR/../../bm_prebuilt_toolchains/"
  test_CHIP_ARCH=${1:-bm1684x}
  if [ ! -d $CROSS_TOOLCHAINS ]; then
    echo "[bm_prebuilt_toolchains]:$CROSS_TOOLCHAINS is not found !"
  else
    make_kernel_module $test_CHIP_ARCH
  fi
}

# currently, for sg2260 libfirmware_core.a must compiled with DE
function check_third_party() {
  strings third_party/firmware/sg2260/libfirmware_core.a | grep 'remove_polls_flag';
  ret_firmware_check=$?
  if [ $ret_firmware_check -ne 0 ]; then
    echo "[Error] Not Found 'remove_polls_flag' in third_party/firmware/sg2260/libfirmware_core.a !!!"
    echo "[Error] You should rebuild_TPU1686, with export EXTRA_CONFIG='-DREMOVE_POLLS_IN_LLM=ON'"
    return -1;
  fi
  check_riscv_third_party_version;
  return $?;
}

function build_riscv_whl() {
  origin_cross_mode=${SOC_CROSS_MODE}
  export SOC_MODE=ON
  export SOC_CROSS_MODE=ON
  export CROSS_TOOLCHAINS="$CURRENT_DIR/../../bm_prebuilt_toolchains/"
  export SOC_CROSS_COMPILE=1
  new_clean && bdist_wheel
  ret=$?
  unset SOC_MODE
  export SOC_CROSS_MODE=${origin_cross_mode}
  unset SOC_CROSS_COMPILE
  return $ret
}

function check_riscv_third_party_version() {
  FILE1="$CURRENT_DIR/../third_party/tpuDNN/sg2260_lib/libtpudnn.so"
  FILE2="$CURRENT_DIR/../third_party/tpuDNN/sg2260_lib/libtpudnn-riscv.so"
  OUTPUT1=$(nm "$FILE1" | grep 1686 | awk 'NR==2 {print $3}')
  OUTPUT2=$(nm "$FILE2" | grep 1686 | awk 'NR==2 {print $3}')
  if [ "$OUTPUT1" != "$OUTPUT2" ]; then
    echo "Version numbers are different, Please using rebuild_TPU1686_riscv to update riscv shared lib."
    return -1
  else
    echo "Version numbers are the same"
    return 0
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
  bash scripts/release.sh || return -1
  export SKIP_DOC=true
  export DISABLE_CACHE=1

  test_CHIP_ARCH=${1:-bm1684x}
  LIBSOPHON_LINK_PATTERN=${2:-local} #local or stable
  TEST_PATTERN=${3:-online} #online,local, or fast
  DEB_PATH_STABLE=${4:-none} #none or given path
  VERSION_PATH_STABLE=${5:-0.4.8}

  echo "[INFO]test_CHIP_ARCH:$test_CHIP_ARCH"
  echo "[INFO]LIBSOPHON_LINK_PATTERN=$LIBSOPHON_LINK_PATTERN"
  echo "[INFO]TEST_PATTERN=$TEST_PATTERN"

  echo "********************************* thirdparty checking... ...*********************************"
  check_third_party; ret_check_third_party=$?
  if [ $ret_check_third_party -eq 0 ];then
    echo "[PRECHECK-$test_CHIP_ARCH] third_party's so lib check true!"
  else
    echo "[PRECHECK-$test_CHIP_ARCH] third_party's so lib check failed!"
    return -1
  fi
  echo "******************************* thirdparty check Successful *********************************"

  echo "********************************* Building... ...*************************************"
  source  $CURRENT_DIR/envsetup.sh $test_CHIP_ARCH $LIBSOPHON_LINK_PATTERN
  new_clean;new_build || return -1
  TPU_TRAIN_CMODEL_PATH=$CURRENT_DIR/../build/firmware_${test_CHIP_ARCH}_cmodel/libfirmware.so
  echo "[INFO]tpu_train_cmodel_path:$TPU_TRAIN_CMODEL_PATH"
  set_cmodel_firmware $TPU_TRAIN_CMODEL_PATH
  echo "********************************* Build Successful *********************************"

  echo "****************************** tgi regression...*************************************"
  if [ $test_CHIP_ARCH = 'sg2260' ]; then
    regression_for_tgi; ret_regression_for_tgi=$?
    if [ $ret_regression_for_tgi -eq 0 ];then
      echo "[RESULT-$test_CHIP_ARCH] regression_for_tgi is computed successfully!"
    else
      echo "[RESULT-$test_CHIP_ARCH] regression_for_tgi is computed failed!"
      return -1
    fi
  fi
  echo "****************************** tgi  Successful *************************************"


  echo "****************************** ops utesting... ...***********************************"
  ops_utest; ret_ops_utest=$?
  echo "[INFO]ret_ops_utest:$ret_ops_utest"
  if [ $ret_ops_utest -eq 0 ];then #must return [0,255] otherwise it will cause scripts fault early
    echo "[RESULT-$test_CHIP_ARCH] all ops_utest are computed, Please check Results above"
  else
    echo "[RESULT-$test_CHIP_ARCH] some ops_utest are failed!"
    return -1
  fi
  echo "****************************** ops utest Successful***********************************"


  echo "****************************** sccl utesting ... ...***********************************"
  if [ $test_CHIP_ARCH = 'sg2260' ]; then
    regression_for_sccl; ret_regression_for_sccl=$?
    if [ $ret_regression_for_sccl -eq 0 ];then
      echo "[RESULT-$test_CHIP_ARCH] regression_for_sccl is computed successfully!"
    else
      echo "[RESULT-$test_CHIP_ARCH] regression_for_sccl is computed failed!"
      return -1
    fi
  fi
  echo "****************************** sccl utest Successful ***********************************"

  echo "****************************** scclHost utesting...  ***********************************"
  if [ $test_CHIP_ARCH = 'sg2260' ]; then
    regression_for_scclHost; ret_regression_for_scclHost=$?
    if [ $ret_regression_for_scclHost -eq 0 ];then
      echo "[RESULT-$test_CHIP_ARCH] ret_regression_for_scclHost is computed successfully!"
    else
      echo "[RESULT-$test_CHIP_ARCH] ret_regression_for_scclHost is computed failed!"
      return -1
    fi
  fi
  echo "****************************** scclHost utest Successful ********************************"

  echo "****************************** build_riscv_whl ... ...   ********************************"
  if [ $test_CHIP_ARCH = 'sg2260' ]; then
    build_riscv_whl; ret_regression_for_riscv=$?
    if [ $ret_regression_for_riscv -eq 0 ];then
      echo "[RESULT-$test_CHIP_ARCH] riscv torch-tpu.whl is build successfully!"
    else
      echo "[RESULT-$test_CHIP_ARCH] riscv torch-tpu.whl is build failed!"
      return -1
    fi
  fi
  echo "****************************** build_riscv_whl Successful ********************************"
  unset DISABLE_CACHE
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

  echo "************** $LIBSOPHON_LINK_PATTERN-LIBSOPHON IS REAEDY *********"
  source  $CURRENT_DIR/envsetup.sh $test_CHIP_ARCH $LIBSOPHON_LINK_PATTERN
  new_clean; new_build
  TPU_TRAIN_CMODEL_PATH=$CURRENT_DIR/../build/firmware_${test_CHIP_ARCH}_cmodel/libfirmware.so
  echo "[INFO]tpu_train_cmodel_path:$TPU_TRAIN_CMODEL_PATH"
  set_cmodel_firmware $TPU_TRAIN_CMODEL_PATH
  echo "*************** CMODEL IS SET *************"
  
  if [ $TEST_PATTERN = "online" ] || [ $TEST_PATTERN = "local" ];then
    # build_libtorch_plugin $TEST_PATTERN
    echo "*************** LIBTORCH_PLUGIN IS BUILT *************"
    ops_utest; ret_ops_utest=$?
    echo "[INFO]ret_ops_utest:$ret_ops_utest"
    if [ $ret_ops_utest -eq 0 ];then #must return [0,255] otherwise it will cause scripts fault early
      echo "[RESULT-$test_CHIP_ARCH] all ops_utest are computed, Please check Results above"
    else
      echo "[RESULT-$test_CHIP_ARCH] some ops_utest are failed!"
      return -1
    fi
    if [ "${test_CHIP_ARCH}" = "sg2260" ]; then
      gpt3block_test; ret_gpt3block_test=$?
      echo "[INFO]ret_gpt3block_test:$ret_gpt3block_test"
      if [ $ret_gpt3block_test -eq 0 ];then #must return [0,255] otherwise it will cause scripts fault early
        echo "[RESULT-$test_CHIP_ARCH] all gpt3block results are computed, Please check Results above"
      else
        echo "[RESULT-$test_CHIP_ARCH] some gpt3block results are failed!"
        return -1
      fi
    fi
    if [ $test_CHIP_ARCH = 'sg2260' ]; then
      regression_for_sccl; ret_regression_for_sccl=$?
      if [ $ret_regression_for_sccl -eq 0 ];then
        echo "[RESULT-$test_CHIP_ARCH] regression_for_sccl is computed successfully!"
      else
        echo "[RESULT-$test_CHIP_ARCH] regression_for_sccl is computed failed!"
        return -1
      fi
    fi
    if [ $test_CHIP_ARCH = 'sg2260' ]; then
      build_riscv_whl; ret_regression_for_riscv=$?
      if [ $ret_regression_for_riscv -eq 0 ];then
        echo "[RESULT-$test_CHIP_ARCH] riscv torch-tpu.whl is build successfully!"
      else
        echo "[RESULT-$test_CHIP_ARCH] riscv torch-tpu.whl is build failed!"
        return -1
      fi
    fi
  fi
}