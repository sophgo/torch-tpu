# Torch-TPU

Torch-TPU is a PyTorch extension that enables running PyTorch models on Sophgo TPU devices.

## Table of Contents

- [Torch-TPU](#torch-tpu)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Quick Start](#quick-start)
  - [Install sophgo tpu driver](#install-sophgo-tpu-driver)
  - [Development](#development)
    - [Directory Structure](#directory-structure)
    - [Download the required repositories](#download-the-required-repositories)
    - [Prepare Environment](#prepare-environment)
    - [Build and Install](#build-and-install)
    - [Debugging](#debugging)
    - [Usage](#usage)
    - [Advanced Features](#advanced-features)
      - [JIT Mode (SG2260 only)](#jit-mode-sg2260-only)
      - [Storage Format](#storage-format)
  - [Examples with Other Training Frameworks](#examples-with-other-training-frameworks)
  - [License](#license)

## Features

- PyTorch model execution on Sophgo TPU devices
- Support for JIT and Eager execution modes
- Flexible storage format options
- Integration with popular deep learning frameworks:
  - DeepSpeed Zero Stage 1 and 2 with CPU offloading
  - Megatron Tensor Parallelism
  - Support for large models like Qwen2


## Quick Start

1. Enter the Docker container:
```bash
docker run -it  \
        --user `id -u`:`id -g` --privileged --cap-add SYS_ADMIN \
        --name torch_tpu \
        --env HOME=$HOME \
        -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v /etc/shadow:/etc/shadow:ro \
        -v /etc/sudoers:/etc/sudoers:ro \
        --shm-size=32G \
        -v $HOME:$HOME \
        -v /workspace:/workspace \
        -v /opt:/opt \
        sophgo/torch_tpu:v0.1-py312  /bin/bash

docker exec -it torch_tpu /bin/bash
```

2. Install torch-tpu

```
pip install torch-tpu
```

3. Prepare a example like:

```python
import torch
import torch_tpu

device = "tpu:0"

a = torch.randn(3, 3).to(device)
b = torch.randn(3, 3).to(device)
c = a + b
print(c)
```

you will see the result like:

```
2024-12-10 19:57:38,880 - torch_tpu.utils.apply_revert_patch - INFO - Patch directory: /workspace/tpu-train/torch_tpu/utils/../demo/patch
...
Stream created for device 0
tensor([[ 0.5913, -2.9734, -3.1663],
        [-1.0463, -0.4574, -1.6539],
        [-0.5939,  0.9554, -1.2390]])
```

## Install sophgo tpu driver

TODO

## Development

### Directory Structure

```
/workspace/
├── tpuv7-runtime
├── TPU1686
└── tpu-train
```


### Download the required repositories

```bash
# Setup git-lfs
sudo apt install git-lfs

# MAKE SURE YOU ARE USING 2.x VERSION! Otherwise IT WILL NOT BE COMPATIBLE WITH GERRIT
git-lfs --version

# install 2.x lfs
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs=2.13.1

# To use Gerrit HTTP protocal, you nee extra configs like:
export GIT_SSL_NO_VERIFY=1
export GIT_LFS_SKIP_SMUDGE=1

# Then clone 'tpu-train' with Gerrit HTTP protocal

# After clone, use the following commands to get large files:
cd tpu-train
git lfs install
git lfs pull --include '*' --exclude ''

```

### Prepare Environment

Download the required repositories:

```bash
git clone <tpu-train-repo-url> /workspace/tpu-train
git clone <tpuv7-runtime-repo-url> /workspace/tpuv7-runtime
git clone <TPU1686-repo-url> /workspace/TPU1686
```

### Build and Install

For developers who need to modify and build from source:

```bash
cd tpu-train
source scripts/envsetup.sh sg2260

# Build TPU base kernels
rebuild_TPU1686

# Install torch-tpu in editable mode
develop_torch_tpu
```

If everything went well, now we have a editable development install.

+ Any changes in .py files, you don't have to reinstall, it is usable on the fly.

+ If you change torch-tpu extension cpps, cd into `build/torch-tpu` and execute `make install` and you are ready to go.

+ If you change kernel source files in `firmware_core`, cd into `build/firmware_sg2260[_cmodel]` and execute `make`.

### Debugging


```bash
cd tpu-train
source scripts/envsetup.sh sg2260

# Debug TPU1686, optional
export EXTRA_CONFIG='-DDEBUG=ON -DUSING_FW_PRINT=ON -DUSING_FW_DEBUG=ON'

# Build TPU base kernels
rebuild_TPU1686

# Make sure we have a clean env
pip uninstall --yes torch-tpu

# Debug torch-tpu, optional
export TPUTRAIN_DEBUG=ON

# Build torch-tpu and install editable
develop_torch_tpu
```

RV version:

```bash
cd tpu-train
source scripts/envsetup.sh sg2260

# Debug TPU1686, optional
export EXTRA_CONFIG='-DDEBUG=ON -DUSING_FW_PRINT=ON -DUSING_FW_DEBUG=ON'

# Build TPU base kernels
rebuild_TPU1686_riscv

# if failed, try to update bm_prebuilt_toolchains
```

RV version torch-tpu(built in SOC environment):
```bash
# Debug torch-tpu, optional
export TPUTRAIN_DEBUG=ON

# build torch-tpu in SOC environment
build_riscv_whl_soc
```


### Usage

```shell
python examples/hello_world.py
```

### Advanced Features

#### JIT Mode (SG2260 only)

Torch-TPU defaults to JIT Mode. To use Eager Mode:
- Locate `TPU_CACHE_BACKEND` in `__init__.py`
- Comment it out

#### Storage Format

To use 32IC format for convolution weights:

```bash
export TORCHTPU_STORAGE_CAST=ON
```


## License

TPU-TRAIN is licensed under the 2-Clause BSD License except for the third-party components.
