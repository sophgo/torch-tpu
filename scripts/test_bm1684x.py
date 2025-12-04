#!/usr/bin/env python3
import os
import time
import subprocess
import pytest
import re
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from python.utest_ops.utest_cmd import Global_Regression_Tester

current_dir = os.path.dirname(os.path.abspath(__file__))
os.environ['CURRENT_DIR'] = current_dir


output_dir = os.path.join(current_dir, "../report")
os.makedirs(output_dir, exist_ok=True)

os.environ['CHIP_ARCH'] = 'bm1684x'
os.environ['LIBSOPHON_LINK_PATTERN'] = 'local'
os.environ['TEST_PATTERN'] = 'online'

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

    echo "################################"
    echo "THIRDPARTY CHECKING"
    echo "################################"
    check_third_party; ret=$?;
    if [ $ret -ne 0 ]; then
        echo "[PRECHECK-$CHIP_ARCH] third_party's so lib check failed!";
        exit $ret;
    fi

    if [ -d "$CURRENT_DIR/../../TPU1686" ]; then
        echo "################################"
        echo "TPU1686 folder found, rebuilding"
        echo "################################"
        source $CURRENT_DIR/envsetup.sh sg2260 || exit 1
        rebuild_TPU1686 || exit 1
    fi

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
    if [ -d "$CURRENT_DIR/../../TPU1686" ]; then
        echo "################################"
        echo "TPU1686 folder found, rebuilding"
        echo "################################"
        rebuild_TPU1686 || exit 1
    fi
    new_build || exit 1

    TPU_TRAIN_CMODEL_PATH=$CURRENT_DIR/../build/firmware_${{CHIP_ARCH}}_cmodel/libfirmware.so
    echo "[INFO]tpu_train_cmodel_path:$TPU_TRAIN_CMODEL_PATH"
    set_cmodel_firmware $TPU_TRAIN_CMODEL_PATH || exit 1

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
    depend_dir = os.path.join(current_dir, '..', 'torch_tpu', 'lib', 'libtpudnn.bm1684x.so')
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
            pytest.fail(f"❌ {test_name} - FAILED (exit code: {returncode})  - Simplified error information\n{full_output}", pytrace=False)

    except subprocess.TimeoutExpired:
        pytest.fail(f"⏰ {test_name} - TIMEOUT", pytrace=False)
    except Exception as e:
        pytest.fail(f"💥 {test_name} - EXCEPTION: {e}", pytrace=False)

if __name__ == "__main__":

    report_file = os.path.join(output_dir, "tpu-train_bm1684x_regression.html")

    pytest_args = [
        __file__,
        f"--html={report_file}",
        "--self-contained-html",
        "--capture=tee-sys",
        "--maxfail=1"
    ]

    print(f"Starting full test  execution...")
    print(f"Test report will be saved to: {report_file}")

    exit_code = pytest.main(pytest_args)

    print(f"Test completed, exit code: {exit_code}")
    sys.exit(exit_code)

