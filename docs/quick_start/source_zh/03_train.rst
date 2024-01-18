模型训练实例
============

本节使用 torch.nn 类来定义MNIST网络模型并介绍说明具体的训练流程。

本节分别给出不同精度（FP16、FP32、混合精度）下的训练代码示例以及训练结果说明，并设定迭代5个epoch。

本节默认使用者已经按照第2章说明完成torch-tpu环境配置安装。


模型训练_FP16
------------------

半精度浮点（FP16）模式下，可以减少模型和数据集的内存占用，降低内存带宽要求，从而在支持 FP16 运算的硬件上加快计算速度。
从接下来三种模式的训练结果所含的时间统计信息，可以明显的观察到计算速度的提升。

本示例是在FP16精度下实现的模型训练。

.. code-block:: shell

    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import datasets
    from torchvision.transforms import ToTensor
    from torch import autocast
    import torch_tpu

    import time
    start_time = None
    def start_timer():
        global start_time
        start_time = time.time()
    def end_timer_and_print(local_msg=""):
        end_time = time.time()
        print("\n" + local_msg)
        print("Total execution time = {:.3f} sec".format(end_time - start_time))

    #### 1.Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    #### 2. create a dataset
    batch_size = 64
    # Create data loaders.
    train_dataloader = DataLoader(training_data, batch_size=batch_size)
    test_dataloader = DataLoader(test_data, batch_size=batch_size)

    #### 3. Get cpu, tpu device for training.
    device = (
        "tpu"
        if torch.tpu.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    ##### 4. Define model loss optimizer
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
    print(model)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    model.half()
    def train(dataloader, model, loss_fn, optimizer):
        size = len(dataloader.dataset)
        model.train()
        for batch, (X, y) in enumerate(dataloader):
            # Compute prediction error
            X, y = X.to(device, torch.float16), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), (batch + 1) * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test(dataloader, model, loss_fn):
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        model.eval()
        test_loss, correct = 0, 0
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(device, torch.float16), y.to(device)
                pred = model(X)
                test_loss += loss_fn(pred, y).item()
                correct += (pred.float().argmax(1) == y).type(torch.float).sum().item()
        test_loss /= num_batches
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    ##### 5. train
    epochs = 5
    start_timer()
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")
    end_timer_and_print()

    checkpoint = {"model": model.state_dict(),
                  "optimizer": optimizer.state_dict(),
                  }
    torch.save(checkpoint, "model_fp16_state.pth")
    print("Saved PyTorch Model State to model.pth")

    ##### 6. infer
    print("start inference")
    checkpoint = torch.load("model_fp16_state.pth")
    model = NeuralNetwork().to(device)

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')

执行上述python文件，得到如下结果（此处只显示训练5个epoch的训练和验证数据）：

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
  -------------------------------
  loss: 2.267578  [   64/60000]
  loss: 2.273438  [ 6464/60000]
  loss: 2.265625  [12864/60000]
  loss: 2.277344  [19264/60000]
  loss: 2.263672  [25664/60000]
  loss: 2.248047  [32064/60000]
  loss: 2.263672  [38464/60000]
  loss: 2.246094  [44864/60000]
  loss: 2.255859  [51264/60000]
  loss: 2.240234  [57664/60000]
  Test Error:
  Accuracy: 19.7%, Avg loss: 2.243021
  Epoch 3
  -------------------------------
  loss: 2.242188  [   64/60000]
  loss: 2.248047  [ 6464/60000]
  loss: 2.232422  [12864/60000]
  loss: 2.250000  [19264/60000]
  loss: 2.236328  [25664/60000]
  loss: 2.208984  [32064/60000]
  loss: 2.238281  [38464/60000]
  loss: 2.212891  [44864/60000]
  loss: 2.226562  [51264/60000]
  loss: 2.207031  [57664/60000]
  Test Error:
  Accuracy: 22.9%, Avg loss: 2.209171
  Epoch 4
  -------------------------------
  loss: 2.210938  [   64/60000]
  loss: 2.220703  [ 6464/60000]
  loss: 2.195312  [12864/60000]
  loss: 2.218750  [19264/60000]
  loss: 2.203125  [25664/60000]
  loss: 2.167969  [32064/60000]
  loss: 2.207031  [38464/60000]
  loss: 2.171875  [44864/60000]
  loss: 2.191406  [51264/60000]
  loss: 2.166016  [57664/60000]
  Test Error:
  Accuracy: 25.9%, Avg loss: 2.169723
  Epoch 5
  -------------------------------
  loss: 2.175781  [   64/60000]
  loss: 2.183594  [ 6464/60000]
  loss: 2.150391  [12864/60000]
  loss: 2.181641  [19264/60000]
  loss: 2.160156  [25664/60000]
  loss: 2.117188  [32064/60000]
  loss: 2.167969  [38464/60000]
  loss: 2.125000  [44864/60000]
  loss: 2.146484  [51264/60000]
  loss: 2.117188  [57664/60000]
  Test Error:
  Accuracy: 34.6%, Avg loss: 2.121927
  Total execution time = 42.809 sec

  Predicted: "Ankle boot", Actual: "Ankle boot"
  """

模型训练_FP32
------------------

单精度浮点（FP32）提供了较高的数值精度，可以进行更精确的计算，减少了数值误差的累积，适用于对精度要求较高的应用。
高精度实现的同时也带来了更多的内存带宽和计算资源的占用，因此模型和数据集占用的内存空间也更大，而这也导致FP32相较于FP16，在计算速度上较慢，而训练精度在5个epoch迭代中也大幅度提升。
从接下来的训练结果所含的时间统计信息，可以明显的观察到。

本示例是在FP32精度下实现的模型训练。

.. code-block:: shell

  import torch
  from torch import nn
  from torch.utils.data import DataLoader
  from torchvision import datasets
  from torchvision.transforms import ToTensor

  import torch_tpu

  import time
  start_time = None
  def start_timer():
      global start_time
      start_time = time.time()
  def end_timer_and_print(local_msg=""):
      end_time = time.time()
      print("\n" + local_msg)
      print("Total execution time = {:.3f} sec".format(end_time - start_time))

  #### 1.Download training data from open datasets.
  training_data = datasets.FashionMNIST(
      root="data",
      train=True,
      download=True,
      transform=ToTensor(),
  )

  # Download test data from open datasets.
  test_data = datasets.FashionMNIST(
      root="data",
      train=False,
      download=True,
      transform=ToTensor(),
  )


  #### 2. create a dataset
  batch_size = 64
  # Create data loaders.
  train_dataloader = DataLoader(training_data, batch_size=batch_size)
  test_dataloader = DataLoader(test_data, batch_size=batch_size)

  #### 3. Get cpu, tpu device for training.
  device = (
      "tpu"
      if torch.tpu.is_available()
      else "cpu"
  )
  print(f"Using {device} device")

  ##### 4. Define model loss optimizer
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
  print(model)

  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

  def train(dataloader, model, loss_fn, optimizer):
      size = len(dataloader.dataset)
      model.train()
      for batch, (X, y) in enumerate(dataloader):
          X, y = X.to(device), y.to(device)

          # Compute prediction error
          pred = model(X)
          loss = loss_fn(pred, y)

          # Backpropagation
          loss.backward()
          optimizer.step()
          optimizer.zero_grad()

          if batch % 100 == 0:
              loss, current = loss.item(), (batch + 1) * len(X)
              print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

  def test(dataloader, model, loss_fn):
      size = len(dataloader.dataset)
      num_batches = len(dataloader)
      model.eval()
      test_loss, correct = 0, 0
      with torch.no_grad():
          for X, y in dataloader:
              X, y = X.to(device), y.to(device)
              pred = model(X)
              test_loss += loss_fn(pred, y).item()
              correct += (pred.argmax(1) == y).type(torch.float).sum().item()
      test_loss /= num_batches
      correct /= size
      print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

  ##### 5. train
  epochs = 5
  start_timer()
  for t in range(epochs):
      print(f"Epoch {t+1}\n-------------------------------")
      train(train_dataloader, model, loss_fn, optimizer)
      test(test_dataloader, model, loss_fn)
  print("Done!")
  end_timer_and_print()

  torch.save(model.state_dict(), "model.pth")
  print("Saved PyTorch Model State to model.pth")

  ##### 6. infer
  print("start inference")
  model = NeuralNetwork().to(device)
  model.load_state_dict(torch.load("model.pth"))

  classes = [
      "T-shirt/top",
      "Trouser",
      "Pullover",
      "Dress",
      "Coat",
      "Sandal",
      "Shirt",
      "Sneaker",
      "Bag",
      "Ankle boot",
  ]

  model.eval()
  x, y = test_data[0][0], test_data[0][1]
  with torch.no_grad():
      x = x.to(device)
      pred = model(x)
      predicted, actual = classes[pred[0].argmax(0)], classes[y]
      print(f'Predicted: "{predicted}", Actual: "{actual}"')

执行上述python文件，得到如下结果（此处只显示训练5个epoch的训练和验证数据）：

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
  -------------------------------
  loss: 2.177251  [   64/60000]
  loss: 2.159091  [ 6464/60000]
  loss: 2.114692  [12864/60000]
  loss: 2.116190  [19264/60000]
  loss: 2.088197  [25664/60000]
  loss: 2.024787  [32064/60000]
  loss: 2.043993  [38464/60000]
  loss: 1.970529  [44864/60000]
  loss: 1.977841  [51264/60000]
  loss: 1.897436  [57664/60000]
  Test Error:
  Accuracy: 58.0%, Avg loss: 1.901019
  Epoch 3
  -------------------------------
  loss: 1.935800  [   64/60000]
  loss: 1.898408  [ 6464/60000]
  loss: 1.800847  [12864/60000]
  loss: 1.822994  [19264/60000]
  loss: 1.747357  [25664/60000]
  loss: 1.683806  [32064/60000]
  loss: 1.692517  [38464/60000]
  loss: 1.597422  [44864/60000]
  loss: 1.626284  [51264/60000]
  loss: 1.509956  [57664/60000]
  Test Error:
  Accuracy: 61.7%, Avg loss: 1.536028

  Epoch 4
  -------------------------------
  loss: 1.600918  [   64/60000]
  loss: 1.562001  [ 6464/60000]
  loss: 1.426221  [12864/60000]
  loss: 1.484659  [19264/60000]
  loss: 1.393732  [25664/60000]
  loss: 1.371750  [32064/60000]
  loss: 1.376287  [38464/60000]
  loss: 1.302687  [44864/60000]
  loss: 1.344361  [51264/60000]
  loss: 1.235606  [57664/60000]
  Test Error:
  Accuracy: 63.5%, Avg loss: 1.266675
  Epoch 5
  -------------------------------
  loss: 1.341337  [   64/60000]
  loss: 1.322475  [ 6464/60000]
  loss: 1.166358  [12864/60000]
  loss: 1.262510  [19264/60000]
  loss: 1.152689  [25664/60000]
  loss: 1.169933  [32064/60000]
  loss: 1.181840  [38464/60000]
  loss: 1.119510  [44864/60000]
  loss: 1.165092  [51264/60000]
  loss: 1.077655  [57664/60000]
  Test Error:
  Accuracy: 64.7%, Avg loss: 1.097683

  Total execution time = 48.419 sec

  Predicted: "Ankle boot", Actual: "Ankle boot"
  """


模型训练_混合精度
------------------

混合精度模式结合使用 FP16 和 FP32。关键的权重、梯度和中间计算可以在 FP32 中进行以保持数值稳定性，而其他操作则可以使用 FP16 来加速计算和减少内存使用。
其可以在不牺牲太多精度的情况下加快训练速度和提高内存效率。它还可以允许更大的模型和批量大小在相同的硬件配置上运行。

本示例是在混合精度下实现的模型训练。

.. code-block:: shell

  import torch
  from torch import nn
  from torch.utils.data import DataLoader
  from torchvision import datasets
  from torchvision.transforms import ToTensor
  from torch import autocast
  import torch_tpu
  import torch, time, gc

  '''
  Usage:
      python mnist_mix_precision.py

  == forward
  model.weight:f32       -> cast -> model.weight(copy):f16      -> op          -> result(f16)
  == backward
  model.weight_grad:f32  <- cast <- model.weight_grad(copy):f16 <- op_backward <- result(f16)
  '''

  # Timing utilities
  start_time = None

  def start_timer():
      global start_time
      start_time = time.time()

  def end_timer_and_print(local_msg=""):
      end_time = time.time()
      print("\n" + local_msg)
      print("Total execution time = {:.3f} sec".format(end_time - start_time))

  #### 1.Download training data from open datasets.
  training_data = datasets.FashionMNIST(
      root="data",
      train=True,
      download=True,
      transform=ToTensor(),
  )

  # Download test data from open datasets.
  test_data = datasets.FashionMNIST(
      root="data",
      train=False,
      download=True,
      transform=ToTensor(),
  )


  #### 2. create a dataset
  batch_size = 64
  # Create data loaders.
  train_dataloader = DataLoader(training_data, batch_size=batch_size)
  test_dataloader = DataLoader(test_data, batch_size=batch_size)

  #### 3. Get cpu, tpu device for training.
  device = (
      "tpu"
      if torch.tpu.is_available()
      else "cpu"
  )
  print(f"Using {device} device")

  ##### 4. Define model loss optimizer
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
  model = NeuralNetwork().tpu()
  print(model)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
  scaler = torch.tpu.amp.GradScaler()
  def train(dataloader, model, loss_fn, optimizer):
      size = len(dataloader.dataset)
      model.train()
      for batch, (X, y) in enumerate(dataloader):
          with autocast(device_type = device, dtype = torch.float16):
              X, y = X.to(device, dtype=torch.float16), y.to(device)
              pred = model(X)
              loss = loss_fn(pred, y)

          scaler.scale(loss.float()).backward()
          scaler.step(optimizer)
          scaler.update()
          optimizer.zero_grad() # set_to_none=True here can modestly improve performance

          if batch % 100 == 0:
              loss, current = loss.item(), (batch + 1) * len(X)
              print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

  def test(dataloader, model, loss_fn):
      size = len(dataloader.dataset)
      num_batches = len(dataloader)
      model.eval()
      test_loss, correct = 0, 0
      with torch.no_grad():
          for X, y in dataloader:
              with autocast(device_type = device, dtype = torch.float16):
                  X, y = X.to(device, dtype=torch.float16), y.to(device)
                  pred = model(X)
                  test_loss += loss_fn(pred, y).item()
                  correct += (pred.float().argmax(1) == y).type(torch.float).sum().item()
      test_loss /= num_batches
      correct /= size
      print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

  ##### 5. train
  epochs = 5
  start_timer()
  for t in range(epochs):
      print(f"Epoch {t+1}\n-------------------------------")
      train(train_dataloader, model, loss_fn, optimizer)
      test(test_dataloader, model, loss_fn)
  end_timer_and_print("Default precision:")
  print("Done!")

  checkpoint = {"model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict()}
  torch.save(checkpoint, "model_fp16_state.pth")
  print("Saved PyTorch Model State to model.pth")

  ##### 6. infer
  print("start inference")
  dev = torch.tpu.current_device()
  checkpoint = torch.load("model_fp16_state.pth")
  model = NeuralNetwork().to(device)

  model.load_state_dict(checkpoint["model"])
  optimizer.load_state_dict(checkpoint["optimizer"])
  scaler.load_state_dict(checkpoint["scaler"])

  classes = [
      "T-shirt/top",
      "Trouser",
      "Pullover",
      "Dress",
      "Coat",
      "Sandal",
      "Shirt",
      "Sneaker",
      "Bag",
      "Ankle boot",
  ]

  model.eval()
  x, y = test_data[0][0], test_data[0][1]
  with torch.no_grad():
      x = x.to(device, )
      pred = model(x)
      predicted, actual = classes[pred[0].argmax(0)], classes[y]
      print(f'Predicted: "{predicted}", Actual: "{actual}"')

执行上述python文件，得到如下结果（此处只显示训练5个epoch的训练和验证数据）：

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
  -------------------------------
  loss: 2.152344  [   64/60000]
  loss: 2.138672  [ 6464/60000]
  loss: 2.074219  [12864/60000]
  loss: 2.078125  [19264/60000]
  loss: 2.039062  [25664/60000]
  loss: 1.972656  [32064/60000]
  loss: 1.994141  [38464/60000]
  loss: 1.916992  [44864/60000]
  loss: 1.926758  [51264/60000]
  loss: 1.826172  [57664/60000]
  Test Error:
  Accuracy: 57.5%, Avg loss: 1.841567

  Epoch 3
  -------------------------------
  loss: 1.886719  [   64/60000]
  loss: 1.849609  [ 6464/60000]
  loss: 1.728516  [12864/60000]
  loss: 1.756836  [19264/60000]
  loss: 1.666016  [25664/60000]
  loss: 1.618164  [32064/60000]
  loss: 1.633789  [38464/60000]
  loss: 1.542969  [44864/60000]
  loss: 1.574219  [51264/60000]
  loss: 1.452148  [57664/60000]
  Test Error:
  Accuracy: 60.6%, Avg loss: 1.484369

  Epoch 4
  -------------------------------
  loss: 1.554688  [   64/60000]
  loss: 1.521484  [ 6464/60000]
  loss: 1.370117  [12864/60000]
  loss: 1.435547  [19264/60000]
  loss: 1.333008  [25664/60000]
  loss: 1.329102  [32064/60000]
  loss: 1.340820  [38464/60000]
  loss: 1.269531  [44864/60000]
  loss: 1.308594  [51264/60000]
  loss: 1.203125  [57664/60000]
  Test Error:
  Accuracy: 63.3%, Avg loss: 1.234953

  Epoch 5
  -------------------------------
  loss: 1.305664  [   64/60000]
  loss: 1.294922  [ 6464/60000]
  loss: 1.125977  [12864/60000]
  loss: 1.229492  [19264/60000]
  loss: 1.113281  [25664/60000]
  loss: 1.138672  [32064/60000]
  loss: 1.162109  [38464/60000]
  loss: 1.098633  [44864/60000]
  loss: 1.141602  [51264/60000]
  loss: 1.053711  [57664/60000]
  Test Error:
  Accuracy: 64.8%, Avg loss: 1.078707


  Default precision:
  Total execution time = 60.383 sec
  Done!
  Saved PyTorch Model State to model.pth
  start inference
  [W CopyFrom.cpp:62] dtypeconvert use cpu impl
  Predicted: "Ankle boot", Actual: "Ankle boot"
  """