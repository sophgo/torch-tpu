开发环境配置
============

本章介绍开发环境配置。

.. _code_load:

代码下载
----------------

代码路径: https://gerrit-ai.sophgo.vip:8443/#/q/project:tpu-train（TO Release）

克隆该代码后, 可以在Linux主机上进行编译开发，建议在Docker中编译（保证环境的一致性）。参考下文配置Docker。

.. _libsophon_install:

lisophon安装（可选）
----------------

`libsophon`是算能板卡设备的驱动和运行时库。当开发人员的代码需要板卡上进行开发与调试时（以下简称`PCIE模式`)，首先需要安装libsophon库。

TORCH-TPU同时提供虚拟仿真的开发与调试环境（以下简称`CMODEL模式`)，此时将会在主机上起一段程序模拟板卡的运行。Cmodel的相关库已经集成在了TORCH-TPU当中，
因此用户无需安装`libsophon`。

建议单个功能时，采用`CMODEL模式`进行调试开发。进行系统级任务时，使用`PCIE模式`进行调试开发。

libsophon的安装文档见 https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/libsophon/guide/html/1_install.html ，
安装包SDK在 https://developer.sophgo.com/site/index/material/41/all.html 。


.. _env_setup:

Docker配置
----------------

TORCH-TPU可以在Docker中开发, 配置好Docker就可以编译和运行了。

从 DockerHub https://hub.docker.com/r/sophgo/torch_tpu 下载所需的镜像:


.. code-block:: shell

   $ docker pull sophgo/torch_tpu:v0.1


如果是首次使用Docker, 可执行下述命令进行安装和配置(仅首次执行):


.. _docker configuration:

.. code-block:: shell
   :linenos:

   $ sudo apt install docker.io
   $ sudo systemctl start docker
   $ sudo systemctl enable docker
   $ sudo groupadd docker
   $ sudo usermod -aG docker $USER
   $ newgrp docker


确保安装包在当前目录, 然后在当前目录创建容器如下:


.. code-block:: shell

  $ docker run --cap-add SYS_ADMIN -itd --restart always --privileged=true \
      -e LD_LIBRARY_PATH=/opt/sophon/libsophon-current/lib:$LD_LIBRARY_PATH \
      -e PATH=/opt/sophon/libsophon-current/bin:$PATH \
      --device=/dev/bmdev-ctl:/dev/bmdev-ctl \
      --device=/dev/bm-sophon0:/dev/bm-sophon0 \
      -v /opt:/opt   \
      -v $HOME:/workspace \
      --name my sophgo/torch_tpu:v0.1 bash
  # myname只是举个名字的例子, 请指定成自己想要的容器的名字
  # -e LD_LIBRARY_PATH=... -e PATH=... 是引入libsophon的环境变量
  # --device=...  是将设备映射到容器内，可以按需添加

注意TORCH-TPU工程在docker中的路径应该是/workspace/tpu-train

.. _compiler :

代码编译
----------------

TORCH-TPU的代码依赖于Pytorch，因此需要提前安装Pytorch。在docker容器中已经安装好了pytorch==2.1.0，
如果未使用docker请按照如下的方式安装Pytorch:

.. code-block:: shell

   $ pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu

TORCH-TPU会从安装的Pytorch包导入头文件和库文件，请确保Pytorch的版本正确。

代码编译方式如下:

如果使用`PCIE模式`进行开发，

.. code-block:: shell

   $ cd tpu-train
   $ source scripts/envsetup.sh bm1684x stable
   $ new_build

如果使用`CMODEL模式`进行开发，

.. code-block:: shell

   $ cd tpu-train
   $ source scripts/envsetup.sh bm1684x local
   $ new_build
   $ set_cmodel_firmware ./build/Release/firmware_core/libcmodel.so #path of libcmodel

回归验证, 如下:

.. code-block:: shell

   # 本工程包含许多测试用例, 可以直接用来验证
   $ pushd python/test
   $ python bmm.py
   $ popd

.. _crosscompile :

SE7交叉编译
--------------------

SE7交叉编译是指在x86主机上编译出SE7平台上可安装执行的wheel包。SE7平台是一款基于BM1684X芯片的AI加速边缘设备，其架构为arm不同，因此需要在x86上交叉编译。  

SE7环境为python3.8，因此需要在x86 docker上安装python3.8。

首先安装python3.8，并创建python3.8的虚拟环境，这里命名为 `crossp`, 可以通过如下指令完成:

.. code-block:: shell

   $ apt update
   $ apt install software-properties-common -y
   $ add-apt-repository ppa:deadsnakes/ppa
   $ apt update
   $ apt install python3.8 -y
   $ apt install python3.8-dev -y
   $ apt install python3.8-venv -y
   $ python3.8 -m venv crossp

使用 `PCIE模式` , 

.. code-block:: shell

   $ cd tpu-train
   $ source scripts/envsetup.sh bm1684x stable

下载并检查必要的文件, 

.. code-block:: shell

   $ soc_build_env_prepare

随后开启python3.8的虚拟环境，

.. code-block:: shell

   $ source crossp/bin/activate

编译SE7的wheel包，

.. code-block:: shell

   $ soc_build

编译完后可以在 `dist` 目录下找到编译好的wheel包。 