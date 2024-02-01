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