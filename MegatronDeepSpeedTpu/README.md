# MegatronDeepSpeedTpu

## Description

MegatronDeepSpeedTpu is a library that adapts the Megatron-DeepSpeed and DeepSpeed framework to Sophgo TPU.

This repository contains the following contents:
- megatron_deepspeed_tpu: the main source code
    - adaptor: the adaptors for Megatron-DeepSpeed and DeepSpeed
    - debugger: debuggers for module and framework for development use
    - tpu_workarounds: workarounds for TPU, which are to be fixed in the future
- examples: examples of using MegatronDeepSpeedTpu
    - gpt: training gpt using Megatron-DeepSpeed (forked from Megatron-Deepspeed/examples_deepspeed)
    - bert: training bert using DeepSpeed (forked from DeepSpeedExamples/training/HelloDeepSpeed)

## Features

- Support DeepSpeed Zero Stage 1 and 2 with CPU offloading
- Support Megatron Tensor Parallelism

## Installation

When you make docker container, make sure to pass the `--network host` option to the `docker run` command to enable network connection.

To install MegatronDeepSpeedTpu, follow these steps after you have installed torch_tpu and collective extension for sg2260 in the docker:

> **In Progress warning:** you should cherry pick this commit on the basis of commit-id 518d749159a943b0fe97505227630c2721342042. Newer commit is not tested and will have problems.

1. Install DeepSpeed v0.13.5 (now we only tested v0.13.5):

    ```shell
    git clone https://github.com/microsoft/DeepSpeed
    cd DeepSpeed
    git checkout v0.13.5
    pip install .
    ``` 

1. Install Megatron-DeepSpeed (now we only tested commit id bcedecd1ff788d4d363f3365fd396053a08d65be):

    ```shell
    git clone https://github.com/microsoft/Megatron-DeepSpeed
    cd Megatron-DeepSpeed
    git checkout bcedecd1ff788d4d363f3365fd396053a08d65be
    pip install .
    ```

1. Install nvcc:

    ```shell
    sudo apt update
    sudo apt install nvidia-cuda-toolkit -y
    ```

    We do not use it but it is required by apex.

1. Install apex: 

    ```shell
    git clone https://github.com/NVIDIA/apex
    cd apex
    pip install -v --disable-pip-version-check --no-build-isolation --no-cache-dir ./
    ```

    We do not use it but it is required by Megatron-DeepSpeed.

    for development use, if you are using cuda for data comparison, you can install cuda version of apex using

    ```shell
    pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
    ```

    We strongly recommend that you should use a seperate conda environment for cuda.

1. Install the required dependencies:

    Change directory to the root of this repository and

    ```shell
    pip install -r requirements.txt
    ```

1. Install MegatronDeepSpeedTpu:
    ```shell
    pip install .
    ```
    (or `pip install -e .` for development use)


## Usage

To use MegatronDeepSpeedTpu in your project, follow these steps:

1. Import the library in your original code before importing megatron:
    ```Python
    import torch_tpu
    import deepspeed
    import megatron_deepspeed_tpu # <=====
    import megatron
    ```
1. Add debuggers that you need (you can see some usage of debuggers in the examples)
1. Run your code

## Run Examples

See the `README.md` in the `examples` directory for more information.