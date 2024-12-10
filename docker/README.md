## TORCH-TPU Dockerfile Repository

This folder hosts the `Dockerfile` to build docker images with various platforms.

### Build torch_tpu from Docker container

**Build docker image**

```Shell
cd tpu-train/docker
docker build -t sophgo/torch_tpu:v0.1 .
```

this docker is already built, you can directly use it by `docker pull sophgo/torch_tpu:v0.1`.

## Enter docker Container

Generally, you can enter the docker container by running the following command:

```Shell
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
                sophgo/torch_tpu:v0.1  /bin/bash
```
