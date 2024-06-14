


使用 Sophgo BM1684x 实现 yolov5s 训练                
=====================================

**摘要**:<br>
本示例是使用算能SC7系列板卡对Stable-Diffusion（以下简称：SD）模型进行训练的一个说明。<br>
通过插件的形式将Sophgo的设备（目前仅支持SC7系列）接入了PyTorch框架。<br>
本示例就是采用当前SD模型比较流行的diffusers的开源库，使用Sophgo SC7设备进行训练流程的简要说明。<br>
目前仅实验了SD14的LORA训练。 

## 1.提前准备
### 1.1 驱动安装
如果已经安装了驱动可以跳过这一步。
参考[https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/libsophon/guide/html/1_install.html](https://doc.sophgo.com/sdk-docs/v23.07.01/docs_latest_release/docs/libsophon/guide/html/1_install.html)进行驱动的安装。
主要命令如下：
```
安装依赖库，只需要执行一次:
sudo apt install dkms libncurses5
安装 libsophon:
sudo dpkg -i sophon-*.deb
在终端执行如下命令，或者登出再登入当前用户后即可使用 bm-smi 等命令:
source /etc/profile
```
`lisophon`可以从`https://developer.sophgo.com/site/index/material/40/all.html`下载，可以使用[下载资源](#下载资源)中打包好的文件。

### 1.2 环境配置
#### 1.2.1 安装Anaconda（可选）
建议使用Anaconda来进行Python环境的管理，这样可以避免Python环境中各种包的依赖问题。
```bash
wget https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-py311_23.5.0-3-Linux-x86_64.sh
bash Miniconda3-py311_23.5.0-3-Linux-x86_64.sh
```
#### 1.2.2 配置python环境
请确保当前Python是3.9的版本。如果安装了Conda，可以通过下面命令新建一个Python3.10的工作环境（建议）
```bash
conda create -n SD python=3.10
conda activate SD
```

#### 1.2.3 安装PyTorch
当前适配的版本是torch2.1, 请通过下面方式安装PyTorch(仅需要安装CPU版本的PyTorch即可)。
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

#### 1.2.4 安装torch-tpu
`torch-tpu`就是算能设备进入PyTorch框架的插件库，是利用了PyTorch的PRIVATEUSEONE后端对算能设备进行的支持。可以从[下载资源](#下载资源)中获取，
通过下面方式进行安装
```bash
pip install torch_tpu-2.1.0.post1-cp39-cp39-linux_x86_64.whl
```
安装完之后，可以通过下面的python脚本进行检验，
```python
import torch
import torch_tpu
device = "tpu"
batch = 8
sequence = 1024
hidden_size = 768
out_size = 3

inp = torch.rand(batch, sequence, hidden_size).to(device)
ln_net = nn.Linear(hidden_size, out_size).to(device)
out = ln_net(inp)
print(out.cpu())
```
如果之前有过使用cuda版本的PyTorch进行训练的经验，那么tpu的使用与cuda设备的使用是基本一致的，将cuda换成tpu就可以。<br>
不过有一点要注意：使用print打印设备上的Tensor的时候，要把Tensor传回cpu（即`print(out.cpu())`),这主要是相关的接口还没有实现。<br>
以上，就完成了PyTorch的支持。<br>


## 2. 进入训练 

进入 yolov5s文件夹，安装必要环境 `pip3 install -r requirements.txt`<br>  
然后执行训练脚本，如下：<br>
```bash
python3 train_fp16.py --img 640 --epoch 3 --data coco128.yaml --weights yolov5s.pt --workers 1 --batch-size 2 --device tpu --optimizer SGD
python3 train.py --img 640 --epoch 3 --data coco128.yaml --weights yolov5s.pt --workers 1 --batch-size 2 --device tpu --optimizer SGD
```
如果缺少数据集，脚本会自动下载数据集  