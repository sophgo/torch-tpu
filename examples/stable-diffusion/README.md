- [使用 Sophgo BM1684x 实现 Stable Diffusion Lora 微调](#使用-sophgo-bm1684x-实现-stable-diffusion-lora-微调)
  - [1.提前准备](#1提前准备)
    - [1.1 驱动安装](#11-驱动安装)
    - [1.2 环境配置](#12-环境配置)
      - [1.2.1 安装Anaconda（可选）](#121-安装anaconda可选)
      - [1.2.2 配置python环境](#122-配置python环境)
      - [1.2.3 安装Pytorch](#123-安装pytorch)
      - [1.2.4 安装torch-tpu](#124-安装torch-tpu)
      - [1.2.5 安装 diffusers，transformers， accelerate库](#125-安装-diffuserstransformers-accelerate库)
        - [1.2.5.1 安装diffusers](#1251-安装diffusers)
        - [1.2.5.2 安装tansformers](#1252-安装tansformers)
        - [1.2.5.3 安装accelerate](#1253-安装accelerate)
  - [2. 使用diffusers进行文生图的推理测试](#2-使用diffusers进行文生图的推理测试)
    - [2.1 使用预训练参数进行文生图推理](#21-使用预训练参数进行文生图推理)
    - [2.2 使用预训练参数,加载LORA参数进行文生图推理](#22-使用预训练参数加载lora参数进行文生图推理)
  - [3. LORA Finetune训练](#3-lora-finetune训练)
    - [3.1 安装训练lora过程中用到的库](#31-安装训练lora过程中用到的库)
    - [3.2 在训练脚本中添加torch\_tpu支持](#32-在训练脚本中添加torch_tpu支持)
      - [3.2.1 导入`torch_tpu`包](#321-导入torch_tpu包)
      - [3.2.2 将lora权重手动放置到tpu上](#322-将lora权重手动放置到tpu上)
    - [3.3 训练脚本训练](#33-训练脚本训练)
      - [3.3.1 使用 fp32 精度进行训练](#331-使用-fp32-精度进行训练)
      - [3.3.2 使用fp16精度进行混精训练](#332-使用fp16精度进行混精训练)
  - [下载资源](#下载资源)

使用 Sophgo BM1684x 实现 Stable Diffusion Lora 微调               
=====================================

**摘要**:<br>
本示例是使用算能SC7系列板卡对Stable-Diffusion（以下简称：SD）模型进行训练的一个说明。<br>
通过插件的形式将Sophgo的设备（目前仅支持SC7系列）接入了Pytorch框架。<br>
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
请确保当前Python是3.9的版本。如果安装了Conda，可以通过下面命令新建一个Python3.9的工作环境（建议）
```bash
conda create -n SD python=3.9
conda activate SD
```

#### 1.2.3 安装Pytorch
当前适配的版本是torch2.1, 请通过下面方式安装Pytorch(仅需要安装CPU版本的Pytorch即可)。
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cpu
```

#### 1.2.4 安装torch-tpu
`torch-tpu`就是算能设备进入Pytorch框架的插件库，是利用了Pytorch的PRIVATEUSEONE后端对算能设备进行的支持。可以从[下载资源](#下载资源)中获取，
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
如果之前有过使用cuda版本的Pytorch进行训练的经验，那么tpu的使用与cuda设备的使用是基本一致的，将cuda换成tpu就可以。<br>
不过有一点要注意：使用print打印设备上的Tensor的时候，要把Tensor传回cpu（即`print(out.cpu())`),这主要是相关的接口还没有实现。<br>
以上，就完成了Pytorch的支持。<br>

#### 1.2.5 安装 diffusers，transformers， accelerate库
下面开始安装StableDiffusion的相关支持。<br>
这里选用的是huggingface的[diffusers](https://github.com/huggingface/diffusers)解决方案，依赖于
[diffusers]()，[transformers]()，[accelerate]()这三个库。
由于这些代码库处于快速的更新当中，这里仅实验了`diffusers==0.20.0`,`transformers==4.29.1`,`accelerate==0.16.0`版本下的代码，下面采用源码的方式进行安装。
##### 1.2.5.1 安装diffusers
```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers
git checkout v0.20.0
python setup.py build develop
```
##### 1.2.5.2 安装tansformers
```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout v4.29.1
python setup.py build develop
```
##### 1.2.5.3 安装accelerate
下载accelerate
```bash
git clone https://github.com/huggingface/accelerate.git
cd accelerate
git checkout v0.16.0
```
添加对SophgoTpu的支持需要修改一点accelerate的代码，如下所示：
```python
#对于accelerate/src/accelerate/accelerator.py 第374行到382行内容
            if not torch.cuda.is_available() and not parse_flag_from_env("ACCELERATE_USE_MPS_DEVICE"):
                raise ValueError(err.format(mode="fp16", requirement="a GPU"))
            kwargs = self.scaler_handler.to_kwargs() if self.scaler_handler is not None else {}
            if self.distributed_type == DistributedType.FSDP:
                from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

                self.scaler = ShardedGradScaler(**kwargs)
            else:
                self.scaler = torch.cuda.amp.GradScaler(**kwargs)
#=============更改为如下：
            if not torch.cuda.is_available() and not parse_flag_from_env("ACCELERATE_USE_MPS_DEVICE") and not torch_tpu.tpu.is_available():
                raise ValueError(err.format(mode="fp16", requirement="a GPU"))
            kwargs = self.scaler_handler.to_kwargs() if self.scaler_handler is not None else {}
            if self.distributed_type == DistributedType.FSDP:
                from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

                self.scaler = ShardedGradScaler(**kwargs)
            elif self.device.type == "tpu":
                import torch_tpu
                self.scaler = torch_tpu.tpu.amp.GradScaler(**kwargs)
            else:
                self.scaler = torch.cuda.amp.GradScaler(**kwargs)
```
最后，进行安装
```bash
python setup.py build develop
```

## 2. 使用diffusers进行文生图的推理测试
在开始LORA Finetune训练之前，可以先利用diffusers的原生Pipeline在Sophgo的设备上进行推理测试。<br>
### 2.1 使用预训练参数进行文生图推理
下面使用`CompVis/stable-diffusion-v1-4`的预训练模型进行文生图。其中，
- `MODEL_NAME`: 预训练模型。第一次加载时，会根据文件名自动从huggingface代码库下载，
也可以使用附录中下载好的参数"all_in_one/CompVis_SD14_pretrained_weights"。
```python
import torch
import torch_tpu
from diffusers import DiffusionPipeline
torch.manual_seed(42)

device = torch.device("tpu:0")
generator = torch.Generator(device=device)
MODEL_NAME="CompVis/stable-diffusion-v1-4" #or PATH/OF/all_in_one/CompVis_SD14_pretrained_weights
prompt = "cute dragon creature"

pipe = DiffusionPipeline.from_pretrained(MODEL_PATH, revision = None, torch_dtype=torch.float16)
pipe.to(device)
image = pipe(prompt, num_inference_steps=20, generator=generator).images[0]
image.save(f"pokemon_wo_lora.png")
```
### 2.2 使用预训练参数,加载LORA参数进行文生图推理
下面使用`CompVis/stable-diffusion-v1-4`的预训练模型并加载LORA参数进行文生图。其中，
- `lora_weight_path`: lora参数。第一次加载时，会根据文件名自动从huggingface代码库下载，
也可以使用附录中下载好的参数"all_in_one/sayakpau_sd_lora-t4"或者使用算能芯片训练得到的lora参数"all_in_one/SophgoTrained_lora_weight"。
```python
from diffusers import StableDiffusionPipeline
import torch

device = torch.device("tpu:0")
MODEL_NAME="CompVis/stable-diffusion-v1-4" # or PATH/OF/all_in_one/CompVis_SD14_pretrained_weights
lora_weight_path = "sayakpaul/sd-model-finetuned-lora-t4" # or PATH/OF/all_in_one/sayakpau_sd_lora or PATH/OF/all_in_one/SophgoTrained_lora_weight
prompt = "cute dragon creature"

pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
pipe.unet.load_attn_procs(lora_weight_path)
pipe.to(device)

image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("pokemon_w_lora.png")
```

## 3. LORA Finetune训练
### 3.1 安装训练lora过程中用到的库
进入`diffusers/exampels/text_to_image`路径，
```bash
pip install -r requirements.txt
```
### 3.2 在训练脚本中添加torch\_tpu支持
这里采用`train_text_to_image_lora.py`脚本进行训练。对脚本内容进行修改如下
#### 3.2.1 导入`torch_tpu`包
在`train_text_to_image_lora.py`第49行添加:
```python
import torch_tpu
```
#### 3.2.2 将lora权重手动放置到tpu上
将`train_text_to_image_lora.py`第519行
```python
lora_layers = AttnProcsLayers(unet.attn_processors)
```
修改为下面的代码
```
lora_layers = AttnProcsLayers(unet.attn_processors).to(accelerator.device)
```
### 3.3 训练脚本训练
使用下面的命令就可以开始训练了，当前提供了`fp32`和`fp16`两种训练方案。<br>
下面命令中,<br>
- `ACCELERATE_TORCH_DEVICE`: 训练采用的设备, 其中`tpu:0`对应与`bm-smi`看到的0号设备。
- `MODEL_NAME`: 预训练模型。第一次训练时会自动从huggingface代码库下载，这里使用的是"CompVis/stable-diffusion-v1-4"的预训练模型。当指定为模型名称（CompVis/stable-diffusion-v1-4）时会从huggingface自动下载对应模型，或者指定本地模型的路径将会使用本地下载的模型。
- `DATASET_NAME`: 训练数据集。第一次训练时，会自动从huggingface下载。

#### 3.3.1 使用 fp32 精度进行训练
```bash
ulimit -n 65535
ulimit -n
export ACCELERATE_TORCH_DEVICE="tpu:0"
export MODEL_NAME=CompVis/stable-diffusion-v1-4
#export MODEL_NAME="CompVis_SD14_pretrained_weights/133a221b8aa7292a167afc5127cb63fb5005638b"
export DATASET_NAME=lambdalabs/pokemon-blip-captions
#export DATASET_NAME="dataset/0111277fb19b16f696664cde7f0cb90f833dec72db2cc73cfdf87e697f78fe02"

python train_text_to_image_lora.py \
--mixed_precision="no" \
--pretrained_model_name_or_path=$MODEL_NAME \
--dataset_name=$DATASET_NAME --caption_column="text" \
--resolution=512 --random_flip \
--train_batch_size=1 \
--num_train_epochs=10 --checkpointing_steps=500 \
--learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
--seed=42 \
--output_dir="sd-pokemon-model-lora_fp32" \
--validation_prompt="cute dragon creature"
```

#### 3.3.2 使用fp16精度进行混精训练
```bash
ulimit -n 65535
ulimit -n
export ACCELERATE_TORCH_DEVICE="tpu:0"
export MODEL_NAME=CompVis/stable-diffusion-v1-4
#export MODEL_NAME="CompVis_SD14_pretrained_weights/133a221b8aa7292a167afc5127cb63fb5005638b"
export DATASET_NAME=lambdalabs/pokemon-blip-captions
#export DATASET_NAME="dataset/0111277fb19b16f696664cde7f0cb90f833dec72db2cc73cfdf87e697f78fe02"

python train_text_to_image_lora.py \
--mixed_precision="fp16" \
--pretrained_model_name_or_path=$MODEL_NAME \
--dataset_name=$DATASET_NAME --caption_column="text" \
--resolution=512 --random_flip \
--train_batch_size=1 \
--num_train_epochs=10 --checkpointing_steps=500 \
--learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
--seed=42 \
--output_dir="sd-pokemon-model-lora_fp16" \
--validation_prompt="cute dragon creature"
```

## 下载资源
备注：由于网络原因huggingface的官网可能无法访问。这里提供了下载好的[模型、数据集、以及torch_tpu和libsophon的包](https://disk.sophgo.vip/sharing/igJxL3Ymn)。其中，各个文件夹的内容如下：
- CompVis_SD14_pretrained_weights: "CompVis/stable-diffusion-v1-4"的预训练模型参数
- dataset: "lambdalabs/pokemon-blip-captions"数据集
- SophgoTrained_lroa_weight: 使用bm1684x训练得到的lora权重
- sayakpau_sd_lora-t4: "sayakpaul/sd-model-finetuned-lora-t4"训练得到的lora权重
- torch_tpu: torch_tpu的wheel包
- libsophon: libsophon的安装包