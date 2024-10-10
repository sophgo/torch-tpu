
Torch-TPU
=========

## SG2260 Emulator Mode Development - Recommended Setup

### Prerequisites

Firstly, get 'tpu-train', 'tpuv7-runtime' and 'TPU1686' repos:

Clone 'TPU1686' && 'tpuv7-runtime' as normally.

To clone 'tpu-train', you need first install git-lfs and pull third-party dependencies:

```bash
sudo apt install git-lfs

# MAKE SURE YOU ARE USING 2.x VERSION! Otherwise IT WILL NOT BE COMPATIBLE WITH GERRIT
git-lfs --version

# To use Gerrit HTTP protocal, you nee extra configs like:
export GIT_SSL_NO_VERIFY=1
export GIT_LFS_SKIP_SMUDGE=1

# Then clone 'tpu-train' with Gerrit HTTP protocal

# After clone, use the following commands to get large files:
cd tpu-train
git lfs install
git lfs pull --include '*' --exclude ''
```

# The three repos need orgnized as:
```
torch-tpu-dev/
├── tpuv7-runtime
├── TPU1686
└── tpu-train
```

Next, have docker installed and pull the latest torch-tpu dev docker image:

```bash
docker pull sophgo/torch_tpu:v0.1
```

Then start a docker container:

```bash
cd torch-tpu-dev/
docker run -v $(pwd):/workspace --restart always -td --name torch-tpu-dev sophgo/torch_tpu:v0.1 bash

# And into the container
docker exec -it torch-tpu-dev bash
```

### Build

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
python setup.py develop
```

If everything went well, now we have a editable development install.

+ Any changes in .py files, you don't have to reinstall, it is usable on the fly.

+ If you change torch-tpu extension cpps, cd into `build/torch-tpu` and execute `make install` and you are ready to go.

+ If you change kernel source files in `firmware_core`, cd into `build/firmware_sg2260[_cmodel]` and execute `make`.

You can always execute `python setup.py develop` after changing source files to rebuild all binaries.
\
\
\
