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