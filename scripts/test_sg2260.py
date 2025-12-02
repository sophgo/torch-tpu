#!/usr/bin/env python3
import os
import time
import subprocess
import pytest
import re
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from python.utest_ops.utest_cmd import Global_Regression_Tester
from python.tgi_test.tgi_op import get_ops_files

current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['CURRENT_DIR'] = current_dir

output_dir = os.path.join(current_dir, "../report")
os.makedirs(output_dir, exist_ok=True)

os.environ['CHIP_ARCH'] = 'sg2260'
os.environ['LIBSOPHON_LINK_PATTERN'] = 'local'
os.environ['TEST_PATTERN'] = 'online'

TEST_ALL_SCCL = [
    "001_test_all_gather.py",
    "002_test_all_reduce.py",
    "003_test_broadcast.py",
    "004_test_gather.py",
    "005_test_reduce.py",
    "006_test_scatter.py",
    "007_test_all_to_all.py",
    "008_test_send.py"
]


def test_release():

    shell_script = f"""
    set -ex
    set -o pipefail
    export CLICOLOR_FORCE=1
    source $CURRENT_DIR/envsetup.sh $CHIP_ARCH $LIBSOPHON_LINK_PATTERN

    echo "################################"
    echo "Ubuntu Information"
    echo "################################"
    echo "[INFO]Ubuntu version"
    version_cmd="cat /etc/os-release"
    $version_cmd


    # echo "################################"
    # echo "THIRDPARTY CHECKING"
    # echo "################################"
    # check_third_party; ret=$?;
    # if [ $ret -ne 0 ]; then
    #     echo "[PRECHECK-$CHIP_ARCH] third_party's so lib check failed!";
    #     exit $ret;
    # fi

    source $CURRENT_DIR/envsetup.sh sg2260 || exit 1
    rebuild_TPU1686 || exit 1
    echo "################################"
    echo "Test release"
    echo "################################"
    bash scripts/release.sh || exit 1

    """

    process = subprocess.Popen(
        ["bash", "-c", shell_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    while True:
        line = process.stdout.readline()
        if line == '' and process.poll() is not None:
            break
        else:
            print(line.strip())

    assert process.returncode == 0, f"Test failed! Error message: {process.stderr}"


def test_sg2260e():

    shell_script = f"""
    set -x
    echo "################################"
    echo "SG2260E regression"
    echo "################################"

    export CLICOLOR_FORCE=1
    source $CURRENT_DIR/envsetup.sh sg2260e || exit 1
    rebuild_TPU1686 || exit 1
    bash $CURRENT_DIR/sg2260e_regression.sh || exit 1

    """

    process = subprocess.Popen(
        ["bash", "-c", shell_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    while True:
        line = process.stdout.readline()
        if line == '' and process.poll() is not None:
            break
        else:
            print(line.strip())

    assert process.returncode == 0, f"Test failed! Error message: {process.stderr}"


def test_building():

    shell_script = f"""
    set -x
    export CLICOLOR_FORCE=1
    source $CURRENT_DIR/envsetup.sh $CHIP_ARCH $LIBSOPHON_LINK_PATTERN
    export SKIP_DOC=true
    export DISABLE_CACHE=1

    echo "################################"
    echo "DEVELOP_TORCH_TPU"
    echo "################################"
    new_clean || exit 1
    rebuild_TPU1686 || exit 1
    new_build || exit 1

    TPU_TRAIN_CMODEL_PATH=$CURRENT_DIR/../build/firmware_${{CHIP_ARCH}}_cmodel/libfirmware.so
    echo "[INFO]tpu_train_cmodel_path:$TPU_TRAIN_CMODEL_PATH"
    set_cmodel_firmware $TPU_TRAIN_CMODEL_PATH

    """

    process = subprocess.Popen(
        ["bash", "-c", shell_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    while True:
        line = process.stdout.readline()
        if line == '' and process.poll() is not None:
            break
        else:
            print(line.strip())

    assert process.returncode == 0, f"Test failed! Error message: {process.stderr}"

def test_ppl_ops():

    current_dir = os.environ.get('CURRENT_DIR')
    depend_dir = os.path.join(current_dir, '..', 'torch_tpu', 'lib', 'libtpudnn.sg2260.so')
    if not os.path.exists(depend_dir):
        print(f"Dependency file does not exist, starting automatic compilation...")
        test_building()

    shell_script = f"""
    set -x
    export CLICOLOR_FORCE=1
    source $CURRENT_DIR/envsetup.sh $CHIP_ARCH $LIBSOPHON_LINK_PATTERN
    export SKIP_DOC=true
    export DISABLE_CACHE=1
    TPU_TRAIN_CMODEL_PATH=$CURRENT_DIR/../build/firmware_${{CHIP_ARCH}}_cmodel/libfirmware.so
    echo "[INFO]tpu_train_cmodel_path:$TPU_TRAIN_CMODEL_PATH"
    set_cmodel_firmware $TPU_TRAIN_CMODEL_PATH

    echo "################################"
    echo "PPL OPS SMOKE TEST"
    echo "################################"
    pushd $CURRENT_DIR/../python/test
    USE_PPL=1 python rmsnorm.py || exit 1
    popd

    """

    process = subprocess.Popen(
        ["bash", "-c", shell_script],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    while True:
        line = process.stdout.readline()
        if line == '' and process.poll() is not None:
            break
        else:
            print(line.strip())

    assert process.returncode == 0, f"Test failed! Error message: {process.stderr}"

test_files, test_ids = get_ops_files()
@pytest.mark.parametrize("test_file", test_files, ids=test_ids)
def test_single_tgi_ops(test_file):

    current_dir = os.environ.get('CURRENT_DIR')
    depend_dir = os.path.join(current_dir, '..', 'torch_tpu', 'lib', 'libtpudnn.sg2260.so')
    test_dir = os.path.join(current_dir, '..', 'python', 'tgi_test')
    if not os.path.exists(depend_dir):
        print(f"Dependency file does not exist, starting automatic compilation...")
        test_building()

    shell_script = f"""
    set -x
    export CLICOLOR_FORCE=1
    source $CURRENT_DIR/envsetup.sh $CHIP_ARCH $LIBSOPHON_LINK_PATTERN
    export SKIP_DOC=true
    export DISABLE_CACHE=1
    TPU_TRAIN_CMODEL_PATH=$CURRENT_DIR/../build/firmware_${{CHIP_ARCH}}_cmodel/libfirmware.so
    echo "[INFO]tpu_train_cmodel_path:$TPU_TRAIN_CMODEL_PATH"
    set_cmodel_firmware $TPU_TRAIN_CMODEL_PATH
    echo "=== TEST_START ==="

    """
    test_name = test_ids
    if len(test_file) == 6:
        model, batch, tp, quantize, case, seq = test_file
        full_command = shell_script + f"cd {test_dir} && python -u tgi_test.py --model {model} --batch {batch} --tp {tp} --case {case} --seq {seq} --test"
        if quantize == 'gptq' and case in ['mmqkv', 'attn_fc', 'mlp']:
            full_command += ' --w4a16'

        test_type = "decode"
        test_name =  f"{model}_b{batch}_tp{tp}_{quantize}_{case}_seq{seq}_{test_type}"

    elif len(test_file) == 5:
        model, batch, tp, quantize, case = test_file
        if case in ['mlp']:  # mlp 情况直接返回
            return
        full_command = shell_script + f"cd {test_dir} && python -u tgi_test.py --model {model} --batch {batch} --tp {tp} --case {case} --seq 6 --prefill --test"
        test_type = "prefill"
        test_name =  f"{model}_b{batch}_tp{tp}_{quantize}_{case}_seq6_{test_type}"
    else:
        test_name =  f"invalid_test_{len(test_file)}"

    print(f"Executing: {full_command}")
    try:
        process = subprocess.Popen(
            ["bash", "-c", full_command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        output_lines = []
        shell_lines = shell_script.strip().split('\n')
        for shell_line in shell_lines:
            if shell_line.strip():
                output_lines.append(shell_line.strip())

        test_start = 0
        while True:
            line = process.stdout.readline()
            if line == '' and process.poll() is not None:
                break
            if line:
                stripped_line = line.strip()
                print(stripped_line)

                if test_start:
                    if not (stripped_line.startswith('[ap]') or stripped_line.startswith('[cdm host')):
                        output_lines.append(stripped_line)

                if "=== TEST_START ===" in stripped_line:
                    test_start=1

        returncode = process.poll()
        full_output = "\n".join(output_lines)
        if returncode == 0:
            print(f"\n{'='*60}")
            print(f"✅ {test_name} - PASSED")
            print(f"{'='*60}")
        else:
            pytest.fail(f"❌ {test_name} - FAILED (exit code: {returncode})\n{full_output}", pytrace=False)

    except subprocess.TimeoutExpired:
        pytest.fail(f"⏰ {test_name} - TIMEOUT", pytrace=False)
    except Exception as e:
        pytest.fail(f"💥 {test_name} - EXCEPTION: {e}", pytrace=False)


def get_utest_files():
    tester = Global_Regression_Tester()
    tester.prepare_utests()
    utest_files = tester.utest_files_list
    utest_files.sort()
    if(len(utest_files)==0): return ["ERROR"]
    numbered_files = []
    for i, file in enumerate(utest_files, 1):
        numbered_files.append(f"{i:03d}_{file}")

    return numbered_files

@pytest.mark.parametrize("test_file", get_utest_files())
def test_single_utest(test_file):

    if test_file == "ERROR":
        pytest.fail("❌ Test case list is empty !!!", pytrace=False)

    original_filename = '_'.join(test_file.split('_')[1:])

    current_dir = os.environ.get('CURRENT_DIR')
    depend_dir = os.path.join(current_dir, '..', 'torch_tpu', 'lib', 'libtpudnn.sg2260.so')
    test_dir = os.path.join(current_dir, '..', 'python', 'utest_ops')
    if not os.path.exists(depend_dir):
        print(f"Dependency file does not exist, starting automatic compilation...")
        test_building()

    shell_script = f"""
    set -x
    export CLICOLOR_FORCE=1
    source $CURRENT_DIR/envsetup.sh $CHIP_ARCH $LIBSOPHON_LINK_PATTERN
    export SKIP_DOC=true
    export DISABLE_CACHE=1
    TPU_TRAIN_CMODEL_PATH=$CURRENT_DIR/../build/firmware_${{CHIP_ARCH}}_cmodel/libfirmware.so
    echo "[INFO]tpu_train_cmodel_path:$TPU_TRAIN_CMODEL_PATH"
    set_cmodel_firmware $TPU_TRAIN_CMODEL_PATH
    echo "=== TEST_START ==="

    """

    test_name = os.path.basename(original_filename)

    try:
        full_command = shell_script + f"cd {test_dir} && {test_name}"
        process = subprocess.Popen(
            ["bash", "-c", full_command],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )

        output_lines = []
        shell_lines = shell_script.strip().split('\n')
        for shell_line in shell_lines:
            if shell_line.strip():
                output_lines.append(shell_line.strip())

        test_start = 0
        while True:
            line = process.stdout.readline()
            if line == '' and process.poll() is not None:
                break
            if line:
                stripped_line = line.strip()
                print(stripped_line)

                if test_start:
                    if not (stripped_line.startswith('[ap]') or stripped_line.startswith('[cdm host')):
                        output_lines.append(stripped_line)

                if "=== TEST_START ===" in stripped_line:
                    test_start=1

        returncode = process.poll()
        full_output = "\n".join(output_lines)
        if returncode == 0:
            print(f"\n{'='*60}")
            print(f"✅ {test_name} - PASSED")
            print(f"{'='*60}")
        else:
            pytest.fail(f"❌ {test_name} - FAILED (exit code: {returncode})\n{full_output}", pytrace=False)

    except subprocess.TimeoutExpired:
        pytest.fail(f"⏰ {test_name} - TIMEOUT", pytrace=False)
    except Exception as e:
        pytest.fail(f"💥 {test_name} - EXCEPTION: {e}", pytrace=False)

@pytest.mark.parametrize("test_command", TEST_ALL_SCCL)
def test_single_all_sccl(test_command):

    original_filename = '_'.join(test_command.split('_')[1:])
    current_dir = os.environ.get('CURRENT_DIR')
    depend_dir = os.path.join(current_dir, '..', 'torch_tpu', 'lib', 'libtpudnn.sg2260.so')
    if not os.path.exists(depend_dir):
        print(f"Dependency file does not exist, starting automatic compilation...")
        test_building()

    shell_script = f"""
    set -x
    export CLICOLOR_FORCE=1
    source $CURRENT_DIR/envsetup.sh $CHIP_ARCH $LIBSOPHON_LINK_PATTERN
    export SKIP_DOC=true
    export DISABLE_CACHE=1
    TPU_TRAIN_CMODEL_PATH=$CURRENT_DIR/../build/firmware_${{CHIP_ARCH}}_cmodel/libfirmware.so
    echo "[INFO]tpu_train_cmodel_path:$TPU_TRAIN_CMODEL_PATH"
    set_cmodel_firmware $TPU_TRAIN_CMODEL_PATH
    echo "=== TEST_START ==="

    echo "################################"
    echo "Testing sccl: {test_command}"
    echo "################################"
    ulimit -n 65535
    test_sccl {original_filename} || exit 1
    echo "sleep 15s to release socket resources(TIME_WAIT limit)... maybe have good methods"
    sleep 15
    """
    try:
        process = subprocess.Popen(
            ["bash", "-c", shell_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines = []
        shell_lines = shell_script.strip().split('\n')
        for shell_line in shell_lines:
            if shell_line.strip():
                output_lines.append(shell_line.strip())

        test_start = 0
        while True:
            line = process.stdout.readline()
            if line == '' and process.poll() is not None:
                break
            if line:
                stripped_line = line.strip()
                print(stripped_line)

                if test_start:
                    if not (stripped_line.startswith('[ap]') or stripped_line.startswith('[cdm host')):
                        output_lines.append(stripped_line)

                if "=== TEST_START ===" in stripped_line:
                    test_start=1

        returncode = process.poll()
        full_output = "\n".join(output_lines)
        if returncode == 0:
            print(f"\n{'='*60}")
            print(f"✅ {test_command} - PASSED")
            print(f"{'='*60}")
        else:
            pytest.fail(f"❌ {test_command} - FAILED (exit code: {returncode})  - Simplified error information\n{full_output}", pytrace=False)

    except subprocess.TimeoutExpired:
        pytest.fail(f"⏰ {test_command} - TIMEOUT", pytrace=False)
    except Exception as e:
        pytest.fail(f"💥 {test_command} - EXCEPTION: {e}", pytrace=False)

if __name__ == "__main__":

    report_file = os.path.join(output_dir, "tpu-train_sg2260_regression.html")

    pytest_args = [
        __file__,
        f"--html={report_file}",
        "--self-contained-html",
        "--capture=tee-sys"
    ]

    print(f"Starting full test  execution...")
    print(f"Test report will be saved to: {report_file}")

    exit_code = pytest.main(pytest_args)

    print(f"Test completed, exit code: {exit_code}")
    sys.exit(exit_code)
