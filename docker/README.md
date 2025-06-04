## TORCH-TPU Dockerfile Repository

This folder hosts the `Dockerfile` to build docker images with various platforms.

### Available Docker Images

There are two versions of the docker image available:
- `sophgo/torch_tpu:v0.1` (Python 3.10)
- `sophgo/torch_tpu:v0.1-py311` (Python 3.11, recommended)

We recommend using the Python 3.11 version (`v0.1-py311`) as it provides better compatibility and performance.

### Build torch_tpu from Docker container

**Build docker image**

```Shell
cd tpu-train/docker
source build.sh py310  # For Python 3.10 version
# or
source build.sh py311  # For Python 3.11 version (recommended)
```

You can also directly pull the pre-built images:
```Shell
docker pull sophgo/torch_tpu:v0.1        # Python 3.10 version
docker pull sophgo/torch_tpu:v0.1-py311  # Python 3.11 version (recommended)
```

## Enter docker Container

Generally, you can enter the docker container by running the following command:

```Shell
# For Python 3.11 version (recommended)
docker run -it  --user `id -u`:`id -g` --privileged --cap-add SYS_ADMIN \
                --name torch_tpu \
                --env HOME=$HOME \
                -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v /etc/shadow:/etc/shadow:ro \
                -v $HOME:$HOME \
                -v /workspace:/workspace \
                -v /opt:/opt \
                --shm-size=32G \
                sophgo/torch_tpu:v0.1-py311  /bin/bash

# For Python 3.10 version
docker run -it  --user `id -u`:`id -g` --privileged --cap-add SYS_ADMIN \
                --name torch_tpu \
                --env HOME=$HOME \
                -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v /etc/shadow:/etc/shadow:ro \
                -v $HOME:$HOME \
                -v /workspace:/workspace \
                -v /opt:/opt \
                --shm-size=32G \
                sophgo/torch_tpu:v0.1  /bin/bash
```

If you have install driver in host machine (refer  ), you can also use the following command to enter the docker container:

```
# For Python 3.11 version (recommended)
docker run -it  --user `id -u`:`id -g` \
                --privileged --cap-add SYS_ADMIN \
                --name torch_tpu_device \
                --env HOME=$HOME \
                -v /dev:/dev \
                -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v /etc/shadow:/etc/shadow:ro \
                -v $HOME:$HOME \
                -v ./code:/workspace \
                -v /home/share/sc11_driver/tpuv7-current/:/opt/tpuv7/tpuv7-current/ \
                -v /opt:/opt \
                --shm-size=32G \
                --env LD_LIBRARY_PATH=/opt/tpuv7/tpuv7-current/lib/ \
                --env PATH=/opt/tpuv7/tpuv7-current/bin/:$PATH \
                sophgo/torch_tpu:v0.1-py311  /bin/bash
```
