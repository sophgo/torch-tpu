#https://blog.csdn.net/scarecrow_sun/article/details/119699797
import torch
from torch.utils.tensorboard.summary import image
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

from resnet50 import Bottleneck
from resnet50 import ResNet
from resnet50 import resnet50

#from torch.utils.tensorboard import SummaryWriter
#myWriter = SummaryWriter('./tensorboard/log/')

myTransforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

#  load
train_dataset = torchvision.datasets.CIFAR10(root='./cifar-10-python/', train=True, download=True, transform=myTransforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

test_dataset = torchvision.datasets.CIFAR10(root='./cifar-10-python/', train=False, download=True, transform=myTransforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)

# 定义模型
#myModel = torchvision.models.resnet50(pretrained=True)
# 将原来的ResNet18的最后两层全连接层拿掉,替换成一个输出单元为10的全连接层
#inchannel = myModel.fc.in_features
#myModel.fc = nn.Linear(inchannel, 10)

net=torchvision.models.resnet50(pretrained=True)
model_dict = net.state_dict()

myModel=ResNet(block=Bottleneck, block_num=[3, 4, 6, 3], num_classes=1000)
myModel.load_state_dict(model_dict)

inchannel = myModel.fc.in_features
myModel.fc = nn.Linear(inchannel, 10)

# 损失函数及优化器
# GPU加速
myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(myDevice)

learning_rate=0.001
myOptimzier = optim.SGD(myModel.parameters(), lr = learning_rate, momentum=0.9)
myLoss = torch.nn.CrossEntropyLoss()

# 测试初始精度
correct = 0
total = 0
myModel.eval()
for images,labels in test_loader:
    # GPU加速
    images = images.to(myDevice)
    labels = labels.to(myDevice)
    outputs = myModel(images)
    numbers,predicted = torch.max(outputs.data,1)
    total += labels.size(0)
    correct += (predicted==labels).sum().item()

print('Testing Accuracy : %.3f %%' % ( 100 * correct / total))
#myWriter.add_scalar('test_Accuracy',100 * correct / total)

for _epoch in range(10):
    training_loss = 0.0
    myModel.train()
    for _step, input_data in enumerate(train_loader):
        image, label = input_data[0].to(myDevice), input_data[1].to(myDevice)   # GPU加速
        predict_label = myModel.forward(image)

        loss = myLoss(predict_label, label)

        #myWriter.add_scalar('training loss', loss, global_step = _epoch*len(train_loader) + _step)

        myOptimzier.zero_grad()
        loss.backward()
        myOptimzier.step()

        training_loss = training_loss + loss.item()
        if _step % 10 == 0 :
            print('[iteration - %3d] training loss: %.3f' % (_epoch*len(train_loader) + _step, training_loss/10))
            training_loss = 0.0
            print()
    correct = 0
    total = 0
    #torch.save(myModel, 'Resnet50_Own.pkl') # 保存整个模型
    myModel.eval()
    for images,labels in test_loader:
        # GPU加速
        images = images.to(myDevice)
        labels = labels.to(myDevice)
        outputs = myModel(images)   # 在非训练的时候是需要加的，没有这句代码，一些网络层的值会发生变动，不会固定
        numbers,predicted = torch.max(outputs.data,1)
        total += labels.size(0)
        correct += (predicted==labels).sum().item()

    print('Testing Accuracy : %.3f %%' % ( 100 * correct / total))
    #myWriter.add_scalar('test_Accuracy',100 * correct / total)

