
Torch-TPU
=========

"Torch-TPU" is a plug-in library for connecting Sophgo computing devices to the PyTorch framework, which uses the PRIVATEUSEONE backend of PyTorch to support computing devices. Now supports Sophgo BM1684x and SG2260 AI acceleators.

## SG2260 Emulator Mode Development - Recommended Setup

### Prerequisites

We recommend installing docker and you can pull the latest torch-tpu dev docker image with:

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
cd torch-tpu
source scripts/envsetup.sh sg2260

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

### Run

#### JIT MODE( only support sg2260)
You can run with JIT MODE( Instruction Cache MODE), with `export TPU_CACHE_BACKEND=/path/of/cmodel_fw`.
example: `export TPU_CACHE_BACKEND=/workspace/tpu-train/build/firmware_sg2260_cmodel/libfirmware.so`.
You can recover to Eager MODE( default mode), with `unset TPU_CACHE_BACKEND`.

#### STORAGE FORMAT
IF you want to make conv's weight with 32IC format on TPU, with `export TORCHTPU_STORAGE_CAST=ON`, then run.
You can close it with `unset TPU_CACHE_BACKEND`.
