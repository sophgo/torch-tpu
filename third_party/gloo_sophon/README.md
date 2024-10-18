# SOPHON_SOPHON

本仓库为添加了用于在SOPHON设备上进行通信的后端代码的SOPHON库

## Date 2023.8.28
所有gloo命名改为sophon，删除`sophon`、`cuda`相关代码，准备增加2260相关算子

## Date 2023.8.18
删除`example_sophgo`例程及相关代码，参考cuda增加`sophon.h`, `sophon_allreduce_ring.h`, `sophon_collectives_host.h`等文件，增加`test_sophgo`例程，申请设备内存并进行通信，支持多机通信(多机通信依赖redis，需要参考文末配置redis服务)。

编译命令：

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_EXAMPLES=1 -DUSE_REDIS=1 -DUSE_SOPHGO=1
make -j4
```

运行命令：

```bash
./test_sophgo <rank> <size>
```

三机器通信结果：

![](./pics/369.jpg)

## Date 2023.8.14
学习了cuda例程之后，感觉之前增加sophon backend方式不太好，可以参考cuda的方法，着重修改Op。

## Date 2023.8.11
增加cuda单测，编译命令（需要cuda环境，没有的话cmake不能通过）：
```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_EXAMPLES=1 -DUSE_REDIS=1 -DUSE_SOPHGO=0 -DUSE_CUDA=1
make -j4
```

运行方式:
```bash
./test_cuda <rank> <size>
```

## Date 2023.8.10
增加example_sophgo.cc

编译命令：

```bash
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_EXAMPLES=1 -DUSE_REDIS=1 -DUSE_SOPHGO=1
make -j4
```

目前关于sophgo的cmake选项好像有点问题，如果不编译sophgo相关，需要将USE_SOPHGO显式设为0

可以编译通过，可以识别到两个chip是否在同一个机器的同一张卡上

运行结果如下：

![](./pics/example_sophgo.jpg)

## Date 2023.8.9
本仓库中，example1从原本的单机多进程通信改为了单机/多机多进程通信

对于用于通信的主机，需要使用命令：```sudo apt install redis-server``` 安装redis服务
运行该例程，需要参考下面的命令

```bash
# 安装redis
sudo apt-get install libhiredis-dev
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug -DBUILD_EXAMPLES=1 -DUSE_REDIS=1
make -j4
```

安装redis并编译好之后，直接运行可能会报无法访问redis主机的错误。

这时，需要修改redis配置文件(redis.conf)中的bind选项为0.0.0.0







