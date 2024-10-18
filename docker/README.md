## TORCH-TPU Dockerfile Repository

This folder hosts the `Dockerfile` to build docker images with various platforms.

### Build torch_tpu from Docker container

**Build docker image**

```Shell
cd tpu-train/docker
docker build -t sophgo/torch_tpu:v0.1 .
```
**Enter docker Container**

```Shell
docker run -it  --user `id -u`:`id -g` --privileged --cap-add SYS_ADMIN \
                --name torch_tpu \
                --env HOME=$HOME \
                -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro -v /etc/shadow:/etc/shadow:ro \
                -v $HOME:$HOME \
                -v /opt:/opt \
                sophgo/torch_tpu:v0.1  /bin/bash
# {code_path} is the torch_npu source code path
```