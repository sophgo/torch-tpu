模型训练实例
============

本节使用 torch.nn 类来定义MNIST网络模型并介绍说明具体的训练流程。

本节分别给出不同精度（FP16、FP32、混合精度）下的训练代码示例以及训练结果说明，并设定迭代5个epoch。

本节默认使用者已经按照第2章说明完成torch-tpu环境配置安装。


模型训练_FP32
------------------

单精度浮点（FP32）提供了较高的数值精度，可以进行更精确的计算，减少了数值误差的累积，适用于对精度要求较高的应用。

高精度实现的同时也带来了更多的内存带宽和计算资源的占用，因此模型和数据集占用的内存空间也更大，而这也导致FP32虽然在计算速度上较慢，但是其训练精度高。

本示例是在默认的FP32精度下实现的模型训练。

项目示例代码可以从 torch-tpu*.whl包中的 torch_tpu/demo/mnist/mnist_fp32.py 获取。(解压whl包即可看到)

执行上述 mnist_fp32.py 文件，得到如下结果（此处只显示训练5个epoch的训练和验证数据）：

.. code-block:: shell

  """
  ===== log:
  Epoch 1
  -------------------------------
  loss: 2.314975  [   64/60000]
  loss: 2.292758  [ 6464/60000]
  loss: 2.284506  [12864/60000]
  loss: 2.266513  [19264/60000]
  loss: 2.262220  [25664/60000]
  loss: 2.239860  [32064/60000]
  loss: 2.231590  [38464/60000]
  loss: 2.205216  [44864/60000]
  loss: 2.206877  [51264/60000]
  loss: 2.162311  [57664/60000]
  Test Error:
  Accuracy: 41.1%, Avg loss: 2.160750

  Epoch 2
  ·····························
  Accuracy: 58.0%, Avg loss: 1.901019
  Epoch 3
  ·····························
  Accuracy: 61.7%, Avg loss: 1.536028
  Epoch 4
  ·····························
  Accuracy: 63.5%, Avg loss: 1.266675
  Epoch 5
  ·····························
  Accuracy: 64.7%, Avg loss: 1.097683
  Total execution time = 48.419 sec
  """


模型训练_FP16
------------------

半精度浮点（FP16）模式下，可以减少模型和数据集的内存占用，降低内存带宽要求，从而在支持 FP16 运算的硬件上，较于FP32模式取得加快计算速度，同时相较于FP32在训练精度方面也会有明显的下降。

本示例是在FP16精度下实现的模型训练。

在示例中，首先对网络模型进行初始化定义：

.. code-block:: shell

    class NeuralNetwork(nn.Module):
            def __init__(self):
                    super().__init__()
                    self.flatten = nn.Flatten()
                    self.linear_relu_stack = nn.Sequential(
                        nn.Linear(28*28, 512),
                        nn.ReLU(),
                        nn.Linear(512, 512),
                        nn.ReLU(),
                        nn.Linear(512, 10)
                    )

            def forward(self, x):
                    x = self.flatten(x)
                    logits = self.linear_relu_stack(x)
                    return logits
    model = NeuralNetwork().to(device)

随后，通过 model.half() 来将所有浮点参数和缓冲区转换为half数据类型，即FP16模式。

.. code-block:: shell

            model.half()    

项目示例代码可以从 torch-tpu*.whl包中的 torch_tpu/demo//mnist/mnist_fp16.py 获取。

执行 mnist_fp16.py 文件，得到如下结果（此处只显示训练5个epoch的训练和验证数据）：

.. code-block:: shell

  """
  ===== log:
  Epoch 1
  -------------------------------
  loss: 2.296875  [   64/60000]
  loss: 2.300781  [ 6464/60000]
  loss: 2.300781  [12864/60000]
  loss: 2.304688  [19264/60000]
  loss: 2.291016  [25664/60000]
  loss: 2.289062  [32064/60000]
  loss: 2.287109  [38464/60000]
  loss: 2.279297  [44864/60000]
  loss: 2.283203  [51264/60000]
  loss: 2.273438  [57664/60000]
  Test Error:
  Accuracy: 13.9%, Avg loss: 2.274557
  
  Epoch 2
  ·····························
  Accuracy: 19.7%, Avg loss: 2.243021
  Epoch 3
  ·····························
  Accuracy: 22.9%, Avg loss: 2.209171
  Epoch 4
  ·····························
  Accuracy: 25.9%, Avg loss: 2.169723
  Epoch 5
  ·····························
  Accuracy: 34.6%, Avg loss: 2.121927
  Total execution time = 42.809 sec
  """

模型训练_混合精度
------------------

自动混合精度(AMP)是一种优化训练过程的技术，它可以在保持模型精度的同时减少计算资源的使用。这是通过在训练过程中使用不同的数据类型（如float16和float32）来完成的。

混合精度模式结合使用 FP16 和 FP32。关键的权重、梯度和中间计算可以在 FP32 中进行以保持数值稳定性，而其他操作则可以使用 FP16 来加速计算和减少内存使用。

其可以在不牺牲太多精度的情况下加快训练速度和提高内存效率。它还可以允许更大的模型和批量大小在相同的硬件配置上运行。

本示例是在混合精度下实现的模型训练。

首先，我们初始化 GradScaler ， GradScaler 会帮助调整梯度的比例，防止在float16计算中出现梯度下溢。

.. code-block:: shell

    scaler = torch.tpu.amp.GradScaler()

然后，使用autocast上下文管理器。在训练循环中，将模型的前向传播过程包装在autocast上下文管理器中。这将临时将选定的操作转换为float16，以加速计算。

.. code-block:: shell

    with autocast(device_type = device, dtype = torch.float16):
            X, y = X.to(device, dtype=torch.float16), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

在计算 loss 时，调用scaler.scale()方法来缩放损失，然后进行反向传播。

.. code-block:: shell

    scaler.scale(loss.float()).backward()

权重更新阶段，使用 scaler.step() 替代 optimizer.step() 来更新权重，并通过 scaler.update() 来更新scaler的状态。

.. code-block:: shell

    scaler.step(optimizer)
    scaler.update()

最后，在每次迭代结束后清除模型的梯度。

.. code-block:: shell

    optimizer.zero_grad()

完整的项目示例代码可以从 torch-tpu*.whl包中的 torch_tpu/demo/mnist/mnist_mix_precision.py 获取。

执行上述 mnist_mix_precision.py 文件，得到如下结果（此处只显示训练5个epoch的训练和验证数据）：

.. code-block:: shell

  """
  Epoch 1
  -------------------------------
  loss: 2.304688  [   64/60000]
  loss: 2.291016  [ 6464/60000]
  loss: 2.271484  [12864/60000]
  loss: 2.253906  [19264/60000]
  loss: 2.246094  [25664/60000]
  loss: 2.210938  [32064/60000]
  loss: 2.212891  [38464/60000]
  loss: 2.181641  [44864/60000]
  loss: 2.179688  [51264/60000]
  loss: 2.126953  [57664/60000]
  Test Error:
  Accuracy: 42.5%, Avg loss: 2.133435

  Epoch 2
  ·····························
  Accuracy: 57.5%, Avg loss: 1.841567
  Epoch 3
  ·····························
  Accuracy: 60.6%, Avg loss: 1.484369
  Epoch 4
  ·····························
  Accuracy: 63.3%, Avg loss: 1.234953
  Epoch 5
  ·····························
  Accuracy: 64.8%, Avg loss: 1.078707
  Total execution time = 60.383 sec
  """