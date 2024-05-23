
使用 Sophgo BM1684x 应用 llama-factory 仓库实现 llama2 训练
=====================================

**摘要**:<br>
本示例是使用算能SC7系列板卡对 llama2（以下简称：SD）模型进行训练的一个说明。<br>
通过插件的形式将Sophgo的设备（目前仅支持SC7系列）接入了Pytorch框架。<br>
本示例就是采用当前SD模型比较流行的 llama-factory 的开源库，使用Sophgo SC7设备进行训练流程的简要说明。<br>
目前仅实验了Llama2 7B 13B lora和全量微调。 


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

#### 1.2.5 安装 llama-factory 必要仓库 

进入`torch_tpu/demo/llama-factory`目录，安装 `accelerate`, `peft` 和 `transformers` 仓库。 

```bash
cd accelerate 
pip install -e .
cd ../transformers
pip install -e .
cd ../peft
pip install -e .
```
下载 `llama-factory` 仓库，并checkout到特定的分支  
```bash
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
git checkout b988ce0a0c164213ad2e52efadd6aa5b71fd39c5 
```
安装 `llama-factory` 的额外依赖  
```bash
pip3 install -r requirements.txt
```

## 2. 进入训练

注意：需要提前下载llama2 7b和13b的权重。 

进入 `torch_tpu/demo/llama-factory` 目录，执行如下命令进行llama2 7b lora训练。 

需要注意：llama2 7b需要两颗芯片才可以正常运行。

在工程里面，使用环境变量设置芯片个数和起始芯片id。 

- `CHIPS` 设置芯片起始id，默认为0
- `SOPHGO_SPLIT` 设置芯片个数，默认为1，目前仅支持连续芯片分配

其他变量意义如下：  
- `USE_SOPHGO` 是否使用Sophgo设备，设置为1表示使用Sophgo设备

```bash
export USE_SOPHGO=1
export SOPHGO_SPLIT=2
export CHIPS=0
python src/train_bash.py     --stage sft     --do_train     --model_name_or_path /workspace/torch2onnx/llama-2-7b-chat-hf     --dataset alpaca_gpt4_zh     --template default     --finetuning_type lora     --lora_target q_proj,v_proj     --output_dir path_to_sft_checkpoint     --overwrite_cache     --per_device_train_batch_size 1     --gradient_accumulation_steps 1     --lr_scheduler_type cosine     --logging_steps 1     --save_steps 10    --learning_rate 5e-5     --num_train_epochs 4.0     --plot_loss     --fp16     --overwrite_output_dir --disable_gradient_checkpointing
```
如上环境变量表示：使用Sophgo设备，使用2个芯片，起始芯片id为0。


执行如下命令进行llama2 13b lora训练。 

```bash
export USE_SOPHGO=1
export SOPHGO_SPLIT=4
export CHIPS=0
python src/train_bash.py     --stage sft     --do_train     --model_name_or_path /workspace/llama2-13b-torch  --dataset alpaca_gpt4_zh     --template default     --finetuning_type lora     --lora_target q_proj,v_proj     --output_dir path_to_sft_checkpoint     --overwrite_cache     --per_device_train_batch_size 1     --gradient_accumulation_steps 1     --lr_scheduler_type cosine     --logging_steps 1     --save_steps 10    --learning_rate 5e-5     --num_train_epochs 4.0     --plot_loss     --fp16     --overwrite_output_dir --disable_gradient_checkpointing
```
如上环境变量表示：使用Sophgo设备，使用4个芯片，起始芯片id为0。

执行如下命令进行llama2 7b 全量训练。 

```sh
export USE_SOPHGO=1
export SOPHGO_SPLIT=11
export CHIPS=0
python3 src/train_bash.py --stage sft --do_train --model_name_or_path /workspace/aa/torch2onnx/llama-2-7b-chat-hf --dataset alpaca_gpt4_zh --template default --finetuning_type full --output_dir path_to_sft_full_checkpoint --overwrite_cache --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --lr_scheduler_type cosine --logging_steps 1 --save_steps 100 --learning_rate 5e-5 --num_train_epochs 3.0 --plot_loss --fp16 --overwrite_output_dir --disable_gradient_checkpointing
```

执行如下命令进行llama2 13b 全量训练。 

```sh
export USE_SOPHGO=1
export SOPHGO_SPLIT=21
export CHIPS=0
python3 src/train_bash.py --stage sft --do_train --model_name_or_path /workspace/llama2-13b/llama2-13b-torch/ --dataset alpaca_gpt4_zh --template default --finetuning_type full --output_dir path_to_sft_full_checkpoint --overwrite_cache --per_device_train_batch_size 1 --gradient_accumulation_steps 1 --lr_scheduler_type cosine --logging_steps 1 --save_steps 100 --learning_rate 5e-5 --num_train_epochs 3.0 --plot_loss --fp16 --overwrite_output_dir --disable_gradient_checkpointing
```
