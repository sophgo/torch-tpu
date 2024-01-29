"""
this case use prue fp16, without loss scale.
will cause lower Accuracy.
"""
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