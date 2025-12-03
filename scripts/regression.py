#!/usr/bin/env python3
import pytest
import sys
import os
import importlib
import shutil

def run_tests_sequentially():
    """Execute test modules sequentially"""
    current_dir = os.path.dirname(os.path.abspath(__file__))

    temp_output_dir = os.path.join(current_dir, "../../report")
    print(temp_output_dir)
    os.makedirs(temp_output_dir, exist_ok=True)

    test_modules = [
        ("test_bm1684x.py", "bm1684x"),
        ("test_sg2260.py", "sg2260")
    ]

    print("=" * 60)
    print("Starting regression testing")
    print("=" * 60)

    exit_codes = []

    source_sg2260 = os.path.join(current_dir,"test_sg2260.py")
    target_sg2260 = os.path.join(current_dir,"../../report/test_sg2260.py")

    shutil.copy2(source_sg2260, target_sg2260)

    for module_file, chip_name in test_modules:
        print(f"\n{'='*60}")
        print(f"Starting {chip_name} test execution")
        print(f"{'='*60}")

        shutil.copy2(target_sg2260, source_sg2260)

        html_report_path = os.path.join(temp_output_dir, f"tpu-train_{chip_name}_regression.html")

        test_file_path = f"{current_dir}/{module_file}"
        pytest_args = [
            test_file_path,
            f"--html={html_report_path}",
            "--self-contained-html",
            "--capture=tee-sys",
            "--maxfail=1"
        ]

        exit_code = pytest.main(pytest_args)
        exit_codes.append(exit_code)

        print(f"{'='*60}")
        print(f"{chip_name} test completed, exit code: {exit_code}")
        print(f"HTML report saved to: {html_report_path}")
        print(f"{'='*60}")


    final_output_dir = os.path.join(current_dir, "../report")
    print(final_output_dir)

    if os.path.exists(final_output_dir):
        print(f"Removing existing directory: {final_output_dir}")
        shutil.rmtree(final_output_dir)

    print(f"\nMoving report directory: {temp_output_dir} -> {final_output_dir}")
    shutil.move(temp_output_dir, final_output_dir)
    print(f"Report moved to: {final_output_dir}")

    return exit_codes

def main():
    exit_codes = run_tests_sequentially()

    overall_exit_code = 0 if all(code == 0 for code in exit_codes) else 1

    print("=" * 60)
    print("All test executions completed")
    print(f"bm1684x exit code: {exit_codes[0]}")
    print(f"sg2260 exit code: {exit_codes[1]}")
    print(f"Overall exit code: {overall_exit_code}")
    print("=" * 60)

    sys.exit(overall_exit_code)

if __name__ == "__main__":
    main()