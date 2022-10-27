#https://blog.csdn.net/weixin_42748371/article/details/124036725
#https://blog.csdn.net/weixin_52527544/article/details/127129303

# database
#https://github.com/pytorch/examples/blob/main/imagenet/extract_ILSVRC.sh

#train with extend ops
#https://zhuanlan.zhihu.com/p/545221832

import torch
from torchvision import transforms
from torchvision import datasets
from torchvision import models
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

#from model import Bottleneck
#from model import ResNet
from resnet50 import Bottleneck
from resnet50 import ResNet
from resnet50 import resnet50

save_dir='./resnet50/'

train_dir='./dataset/train/'
test_dir='./dataset/test/'
net_name='resnet50'

batch_size = 64
epochs=30
lr=0.001

import os
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#normalize = transforms.Normalize(mean=[0.5,0.5,0.5],
#                                 std=[0.5,0.5,0.5])
normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
train_dataset = datasets.ImageFolder(train_dir,
                                     transforms.Compose([
                                         transforms.RandomResizedCrop((224, 224)),
                                         transforms.RandomHorizontalFlip(),
                                         transforms.ToTensor(),
                                         normalize]))
#test_dataset = datasets.ImageFolder(test_dir,
#                                    transforms.Compose([
#                                        transforms.Resize((224, 224)),
#                                        transforms.ToTensor(),
#                                        normalize]))
train_loader = DataLoader(train_dataset,
               batch_size=batch_size,
               shuffle=True)
#test_loader = DataLoader(test_dataset,
#               batch_size=batch_size,
#               shuffle=True)

net=models.resnet50(pretrained=True)
model_dict = net.state_dict()

#model=ResNet()
model=ResNet(block=Bottleneck, block_num=[3, 4, 6, 3], num_classes=1000)
#model=resnet50
model.load_state_dict(model_dict)

model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.5)

def train(epoch):
    running_loss = 0.0
    correct = 0
    total = 0
    model.train()
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()
        # forward + backward + update
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        loss = criterion(outputs, target)
        loss.backward()
        # grad update use tpu after
        optimizer.step()
        running_loss += loss.item()
        if (batch_idx+1) % 30 == 0:
            print('[%d, %3d]  loss: %.3f  acc: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 30, correct / total))
            #print(predicted)
            #print(total)
            #print(correct)
            running_loss = 0.0
    print('[%d] Accuracy on train set: %d %% [%d/%d]' % (epoch+1, 100 * correct / total, correct, total))
    return correct / total
'''
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            #print(predicted)
            #print(target.size(0))
            #print((predicted == target).sum().item())
    print('[%d] Accuracy on test set: %d %% [%d/%d]' % (epoch+1, 100 * correct / total, correct, total))
    return correct / total
'''
txt_name = save_dir + net_name + '_log.txt'

train_acc_list=[]
#test_acc_list=[]
print("===================================Start Training===================================")
for epoch in range(epochs):
    train_acc=train(epoch)
    #test_acc=test()
    with open(txt_name, 'a') as f:
        #f.write(str(train_acc) + ' '+str(test_acc) + '\n')
        f.write(str(train_acc) + '\n')
    train_acc_list.append(train_acc)
    #test_acc_list.append(test_acc)
    #if (epoch+1)%10 == 0:
    #    torch.save(model.state_dict(),save_dir+'{}-{}-{}-{}-{:.3f}-{:.3f}.pt'.format(net_name,epoch+1,batch_size,lr,train_acc,test_acc))
    torch.save(model.state_dict(),save_dir+'{}-{}-{}-{}-{:.3f}.pt'.format(net_name,epoch+1,batch_size,lr,train_acc))
print("=================================Training Finished==================================")
print(train_acc_list)
#print(test_acc_list)


