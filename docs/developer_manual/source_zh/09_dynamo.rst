使用TorchDynamo训练流程
=========================

本节使用TorchDynamo接入sophgo tpu-mlir编译器完成在compile mode下的模型训练。

环境准备
----------

Docker准备
~~~~~~~~~~

从 DockerHub https://hub.docker.com/r/sophgo/tpuc_dev 下载所需的镜像

.. code-block:: shell

   $ docker pull sophgo/tpuc_dev:v3.1

创建容器如下：

.. code-block:: shell

   $ docker run --privileged 
   -it 
   --name tpuc_dev 
   -e NVIDIA_DRIVER_CAPABILITIES=compute,utility 
   -v /dev:/dev 
   -v /workpath/:/workspace 
   --ipc=host 
   --ulimit memlock=-1 
   --ulimit stack=67108864 
   sophgo/tpuc_dev:v3.1

其中--name 表示容器名称，/workpath表示host中工作路径，会将此路径映射为/workspace，-e NVIDIA_DRIVER_CAPABILITIES=compute,utility 
为可选项，加载部分模型时需要，若加入此参数需要安装NVIDIA显卡驱动。

Python环境准备
~~~~~~~~~~~~~~

torch 2.8.0，若使用上述NVIDIA驱动选项参数，torch需安装为cuda版本

transformers 4.31.0

下载tpu-mlir工程的whl包 https://www.yunpan.com/surl_ybcPvTGA28X

目前是临时的whl包，后续会更新在 https://github.com/sophgo/tpu-mlir/releases/

使用pip install安装

.. code-block:: bash

    $ pip install tpu_mlir-1.7b122-py3-none-any.whl

安装完毕后使用python脚本运行以下代码检验

.. code-block:: bash

    $ python 
   >> import tpu_mlir

如果没有报错证明安装成功。

下载torch_tpu插件的whl包 https://github.com/sophgo/torch-tpu/releases

.. code-block:: bash

   $ pip install torch-tpu.whl

这里torch-tpu.whl文件名以实际下载文件名为准，安装完毕后使用python脚本运行以下代码检验

.. code-block:: bash

    $ python 
   >> import torch_tpu

如果没有报错证明安装成功。

tpu-train环境准备
~~~~~~~~~~~~~~~~~

切换到tpu-train的目录下执行

.. code-block:: bash

    $ source scripts/envsetup.sh sg2260 

使用说明
----------

模型训练脚本torch_tpu/dynamo/main.py，目前支持resnet50与bert-large两个模型

.. code-block:: bash

    $ python3 main.py --chip bm1690 --debug const_name --model resnet50

chip参数默认选择bm1690，debug参数选择const_name可以统一训练过程生成的中间文件文件名，model为需要训练模型名称

.. code-block:: bash

   $ python
   $ // 初始化tpu device，通过torch_tpu插件实现tpu设备支持
   $ device = torch.device("tpu:0")
   $ // 初始化网络与输入
   $ input = torch.randn((1, 3, 224, 224))
   $ import torchvision.models as models
   $ mod = models.resnet50()
   $ net_d = copy.deepcopy(mod)
   $ net_d.train()
   $ // 将输入与网络放到tpu device，用法与cuda一致
   $ net_d.to(device)
   $ input_d = input.to(device)
   $ optimizer = torch.optim.SGD(net_d.parameters(), lr=0.01)
   $ optimizer.zero_grad()
   $ // aot_backend定义在tpu_mlir_jit.py，会将模型的计算图接入tpu_mlir_compiler编译器
   $ model_opt = torch.compile(net_d, backend=aot_backend)
   $ loss_d = model_opt(input_d)
   $ loss_d[0,0].backward()
   $ optimizer.step()

运行中会生成若干文件，其中npz类型文件为存放的模型权重，mlir类型文件为模型转换的IR文件，bmodel类型文件为模型转换成的芯片指令文件。

框架说明
----------

torch_tpu/dynamo/tpu_mlir_jit.py

定义了tpu_mlir_compiler，配合aot_autograd将捕获的fx graph通过tpu_mlir_compiler编译器优化训练模型前后向计算图。

其中使用了decompositions功能，定义了decompositions_dict这样一个算子分解集合，会对fx graph中的node进行分解，将算子集合中大算子分解为prim torch中定义的基本算子。

torch_tpu/dynamo/FxGraphConvertor.py与torch_tpu/dynamo/FxMlirImportor.py

负责前端转换工作，将fx graph中的node转换为tpu_mlir中定义的Top Dialect的IR格式，进而通过tpu_mlir中的mlir_opt_for_top, mlir_lowering, mlir_to_model三个接口将mlir文件转换为bmodel格式，实现将算子转换为芯片可读取的计算指令。

torch_tpu/dynamo/TpuMlirModule.py

负责调用tpu_mlir工程中bmodel推理接口，将计算指令下发到芯片设备模拟器中，完成计算。

所用接口具体实现参考 https://github.com/sophgo/tpu-mlir/

