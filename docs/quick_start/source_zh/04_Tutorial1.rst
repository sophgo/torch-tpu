===================
模型训练实例_BM1684X_StableDiffusion
===================

本章节介绍了如何在使用Torch-TPU的前提下，使用LoRA进行Stable-Diffusion的Finetune的一个示例。

本示例使用了开源的代码库Diffusers、Transformers，其展示了Torch-TPU仅需要做少量修改
就可以嵌入现有的基于PyTorch的开源代码库当中的特点。

相关背景知识介绍
==================

StableDiffusion
------------------

Stable diffusion是一种基于潜在扩散模型（Latent Diffusion Models）的文本到图像生成模型，能够根据任意文本输入生成高质量、高分辨率、高逼真的图像。
同时，也可以应用于其他任务，如内补绘制、外补绘制，以及在提示词指导下产生图生图的翻译。

LoRA
------------------

LoRA，英文全称Low-Rank Adaptation of Large Language Models，直译为大语言模型的低阶适应，是一种PEFT（参数高效性微调方法）。

LoRA的基本原理是：在冻结预训练好的模型权重参数的情况下，只训练模型中额外新增的网络层的参数。

由于这些新增参数数量较少，这样不仅 Finetune 的成本显著下降，还能获得和全模型微调类似的效果。
更具体的介绍可以参见 https://arxiv.org/abs/2106.09685。

Diffusers
------------------

Diffusers(https://github.com/huggingface/diffusers)是Huggingface推出了的基于PyTorch的扩散模型工具包集合，其中包含了StableDiffusion模型（StableDiffusion是扩散模型的一种）。

该工具包提供了众多的模型和简单易用的使用方式，使开发者和用户可以轻松进行模型的训练和应用的开发。

Transformers
------------------

Transformers(https://github.com/huggingface/transformers)是Huggingface推出的NLP(自然语言处理)工具包，同样支持PyTorch框架。

Transformers提供了数以千计的预训练模型，支持 100 多种语言的文本分类、信息抽取、问答、摘要、翻译、文本生成。

使用Transformers可以快速的进行这些模型的训练和推理。

开始前的准备
==================

检查环境
------------------

在本节开始之前，首先检查Torch-TPU是否正常，如果异常请参考 开发环境配置 进行正确的环境配置。

本节需要通过源码安装的方式对依赖库进行安装。用户在使用 git clone 命令获取源代码时，可能会遇到速度慢或者拉取失败的问题，请参考以下步骤解决：

（1）通过该网站链接 https://www.ipaddress.com/ 获取 github.com 对应的 ip 地址。

.. figure:: ../assets/4_git_0.png
   :width: 2000px
   :height: 600px
   :scale: 50%
   :align: center
   :alt: SOPHGO LOGO

（2）通过该网站链接 https://www.ipaddress.com/ 获取 github.global.ssl.fastly.net 对应的 IP 地址（有4个 IP ，任选其一）。

.. figure:: ../assets/4_git_1.png
   :width: 2000px
   :height: 600px
   :scale: 50%
   :align: center
   :alt: SOPHGO LOGO

（3）执行下述命令，打开 hosts。

.. code-block:: shell

    $ sudo vim /etc/hosts

（4）在文件末尾添加如下两行，对应的 IP 为以上两步查询获取到的 IP 地址，并保存退出。

.. code-block:: shell

    140.82.113.3 github.com
    151.101.193.194 github.global.ssl.fastly.net    # 任意选个ip

（5）最后执行下述命令，并重启机器。

.. code-block:: shell

    $ sudo /etc/init.d/network-manager restart
    $ sudo reboot

至此，使用 git clone 命令获取源代码失败的问题解决。

接下来将介绍如何安装并配置特定版本的依赖库。

请注意，下面 "Diffusers"、"Transformers"、"accelerate" 三个仓库，应放置在同级目录下。

”Transformers“若不采用源码安装则，无需下载。

下载安装 Diffusers
------------------

Diffusers 采用源码安装的方式，目前支持版本为v0.20.0。可执行下述命令进行安装：

.. code-block:: shell

    $ git clone https://github.com/huggingface/diffusers.git
    $ cd diffusers
    $ git checkout v0.20.0
    $ python setup.py build develop
    $ cd ..


安装 Transformers
------------------

Transformers 目前支持版本为v4.29.1。可以采用源码安装的方式，可执行下述命令进行安装：

.. code-block:: shell

    $ git clone https://github.com/huggingface/transformers.git
    $ cd transformers
    $ git checkout v4.29.1
    $ python setup.py build develop
    $ cd ..

如果因为网络问题无法下载transformers进行源码安装，可以使用下面的命令进行安装：

.. code-block:: shell

    $ pip install transformers==4.29.1 -i https://pypi.tuna.tsinghua.edu.cn/simple


下载安装 accelerate
------------------

accelerate 为 Diffusers 训练和推理的加速依赖库。

accelerate 采用源码安装的方式，目前支持版本为v0.16.0。可执行下述命令进行安装：

.. code-block:: shell

    $ git clone https://github.com/huggingface/accelerate.git
    $ cd accelerate
    $ git checkout v0.16.0

接下来，需要通过添改一部分代码从而实现其对Torch-TPU的支持：

首先，找到 accelerate/src/accelerate/accelerator.py 文件，对应地将其 第374行 到 第382行 内容(如下所示):

.. code-block:: python

    if not torch.cuda.is_available() and not parse_flag_from_env("ACCELERATE_USE_MPS_DEVICE"):
        raise ValueError(err.format(mode="fp16", requirement="a GPU"))
    kwargs = self.scaler_handler.to_kwargs() if self.scaler_handler is not None else {}
    if self.distributed_type == DistributedType.FSDP:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

        self.scaler = ShardedGradScaler(**kwargs)
    else:
        self.scaler = torch.cuda.amp.GradScaler(**kwargs)

修改为:

.. code-block:: python

    import torch_tpu
    if not torch.cuda.is_available() and not parse_flag_from_env("ACCELERATE_USE_MPS_DEVICE") and not torch_tpu.tpu.is_available():
        raise ValueError(err.format(mode="fp16", requirement="a GPU"))
    kwargs = self.scaler_handler.to_kwargs() if self.scaler_handler is not None else {}
    if self.distributed_type == DistributedType.FSDP:
        from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler
        self.scaler = ShardedGradScaler(**kwargs)
    elif self.device.type == "tpu":
        self.scaler = torch_tpu.tpu.amp.GradScaler(**kwargs)
    else:
        self.scaler = torch.cuda.amp.GradScaler(**kwargs)

这一步需要注意修改后代码的缩进格式，这将直接影响能否顺利通过接下来的安装步骤。推荐最好使用 VS Code 等代码编译软件进行修改。
修改完成后，可对照下图所示，检查代码缩进格式是否有误。

.. figure:: ../assets/4_accelerator.png
   :width: 2200px
   :height: 700px
   :scale: 50%
   :align: center
   :alt: SOPHGO LOGO

然后，执行下述命令完成安装：

.. code-block:: shell

    $ python setup.py build develop

进行文生图的推理
==================

完成上述准备工作之后，接下来便可以通过使用 Diffusers 代码库，进行文字图像生成的推理。

需要注意的是，本节示例代码会自动从 https://huggingface.co/CompVis/stable-diffusion-v1-4/tree/main 下载所需模型权重文件，
请提前确保能够正确访问链接，以免下载失败。

使用预训练参数进行文生图推理
------------------

本节将介绍通过 StableDiffusion 进行文生图的 python 示例代码。

首先，为了确保推理结果的一致性，我们需要固定随机种子：

.. code-block:: python

    torch.manual_seed(42)

随后，需要指定推理设备：

.. code-block:: python

    device = torch.device("tpu:0")

该示例中，通过 DiffusionPipeline.from_pretrained 来加载预训练的扩散模型，其会自动从库中下载 Stable-Diffusion v1.4 版本的模型：

.. code-block:: python

    MODEL_NAME="CompVis/stable-diffusion-v1-4"
    pipe = DiffusionPipeline.from_pretrained(MODEL_NAME, revision = None, torch_dtype=torch.float16)

我们想要生成一张可爱的龙宝宝的图片，因此给定提示词为：

.. code-block:: python

    prompt = "cute dragon creature"

在执行代码前，为保证顺利进行，需要先执行以下命令：

.. code-block:: shell

    $ ulimit -n 65535

该示例的完整代码如下：

.. code-block:: python

    import torch
    import torch_tpu
    from diffusers import DiffusionPipeline
    torch.manual_seed(42)

    device = torch.device("tpu:0")
    generator = torch.Generator(device=device)
    MODEL_NAME="CompVis/stable-diffusion-v1-4" #or PATH/OF/all_in_one/CompVis_SD14_pretrained_weights
    prompt = "cute dragon creature"

    pipe = DiffusionPipeline.from_pretrained(MODEL_NAME, revision = None, torch_dtype=torch.float16)
    pipe.to(device)
    image = pipe(prompt, num_inference_steps=20, generator=generator).images[0]
    image.save(f"pokemon.png")


执行示例代码，如下图所示，即为正常运行。

.. figure:: ../assets/without_lora_demo.png
   :width: 2200px
   :height: 400px
   :scale: 50%
   :align: center
   :alt: SOPHGO LOGO

执行如上示例代码成功执行后，会在当前的路径下生成一张名为 “pokemon.png” 的符合我们预设提示词的龙宝宝图片。该节测试结果图片，可参考本节末尾对比图。


使用预训练参数，加载 LoRA 参数进行文生图推理
------------------

本节将介绍在使用 StableDiffusion 预训练参数的基础上，同时加载 LoRA 参数进行文生图的 python 示例代码。

相较于上一节，我们只需要在加载预训练模型后，使模型加载指定 LoRA 参数即可：

.. code-block:: python

    lora_weight_path = "sayakpaul/sd-model-finetuned-lora-t4" 
    pipe.unet.load_attn_procs(lora_weight_path)

该示例的完整代码如下：

.. code-block:: python

    from diffusers import StableDiffusionPipeline
    import torch
    import torch_tpu
    torch.manual_seed(42)

    device = torch.device("tpu:0")
    MODEL_NAME="CompVis/stable-diffusion-v1-4" 
    lora_weight_path = "sayakpaul/sd-model-finetuned-lora-t4" 
    prompt = "cute dragon creature"

    pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    pipe.unet.load_attn_procs(lora_weight_path)
    pipe.to(device)

    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save("pokemon_lora.png")

执行示例代码，如下图所示，即为正常运行。

.. figure:: ../assets/lora_demo.png
   :width: 1900px
   :height: 400px
   :scale: 50%
   :align: center
   :alt: SOPHGO LOGO

执行如上示例代码后，会在当前的路径下生成一张名为 “pokemon_lora.png” 的图片。因为本示例加载的 LoRA 参数是在卡通风格的数据集上训练的，所以相较于未加载 LoRA 参数的示例生成的更贴近卡通形象。

使用预训练参数的前提下，是否加载 Lora 参数的两种推理结果如下图所示。

这里需要注意一点，只要生成的图片有呈现出卡通形象即可判定上述步骤成功执行。

.. figure:: ../assets/4_fp32_fp16.png
   :width: 1600px
   :height: 650px
   :scale: 50%
   :align: center
   :alt: SOPHGO LOGO

进行LoRA Finetune训练
==================

本节将介绍如何基于 Diffusers 库，在 Torch-TPU 支持下实现 LoRA Finetune 训练。

环境准备
------------------

在实现 Finetune 训练前，我们需要根据环境配置需求文件进行相关库文件的安装。lorafinetune的代码位置在 diffusers/examples/text_to_image ，进入该路径，首先进行环境配置。

.. code-block:: shell

    $ pip install -r requirements.txt

Torch-TPU 支持
------------------

使用者需要通过对 Diffusers 提供的示例代码手动添改一部分内容来实现其对Torch-TPU的支持。

首先，找到 train_text_to_image_lora.py 文件，并对其进行如下修改：

（1）在代码 第49行 添加：

.. code-block:: python

    import torch_tpu

（2）在训练过程中，需要将lora权重手动放置到tpu上。对应地将其 第518行 内容（如下所示）：

.. code-block:: python

    lora_layers = AttnProcsLayers(unet.attn_processors)

修改为：

.. code-block:: python

    lora_layers = AttnProcsLayers(unet.attn_processors).to(accelerator.device)

模型训练指令参数说明
------------------

本章节中包含 FP32 和 FP16 两种训练方案。

在此，对本节接下来的两类型训练方案中的部分特定参数进行说明：

（1） 'ACCELERATE_TORCH_DEVICE'

训练采用的设备, 其中`tpu:0`对应与`bm-smi`看到的0号设备。

（2） 'MODEL_NAME'

预训练模型。第一次训练时会自动从huggingface代码库下载，这里使用的是"CompVis/stable-diffusion-v1-4"
的预训练模型。当指定为模型名称（CompVis/stable-diffusion-v1-4）时会从huggingface自动下载对应模型，或者指定本地模型的路径将会使用本地下载的模型。

（3） 'DATASET_NAME'

训练数据集，这里采用 lambdalabs/pokemon-blip-captions（https://huggingface.co/datasets/lambdalabs/pokemon-blip-captions），是一个宝可梦卡通动画的数据集，该数据集的每一个样本都由一张宝可梦图片和对应的描述构成。
第一次训练时，会自动从huggingface下载。

使用 FP32 精度进行训练
------------------

执行下面的shell脚本：

.. code-block:: shell


    $ ulimit -n 65535
    $ ulimit -n
    $ export ACCELERATE_TORCH_DEVICE="tpu:0"
    $ export MODEL_NAME=CompVis/stable-diffusion-v1-4
    $ export DATASET_NAME=lambdalabs/pokemon-blip-captions   

    $ python train_text_to_image_lora.py \
        --mixed_precision="no" \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --dataset_name=$DATASET_NAME --caption_column="text" \
        --resolution=512 --random_flip \
        --train_batch_size=1 \
        --num_train_epochs=1 --checkpointing_steps=500 \
        --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
        --seed=42 \
        --output_dir="sd-pokemon-model-lora_fp32" \
        --validation_prompt="cute dragon creature"

执行如上指令，如下图所示，即为正常运行。

.. figure:: ../assets/fp32_demo.png
   :width: 1600px
   :height: 400px
   :scale: 50%
   :align: center
   :alt: SOPHGO LOGO

执行如上指令，训练完成后，会在当前目录下生成名为 sd-pokemon-model-lora_fp32 的文件夹，其中包含训练得到的lora参数、训练记录logs，以及checkpoint数据。文件结构如下。

.. code-block:: shell

   /text_to_image/sd-pokemon-model-lora_fp32/
   |—— checkpoint-*
   │    ├── optimizer.bin
   │    ├── pytorch_model.bin
   │    └── random_states_0.pkl
   |—— logs
   │    └── text2image-fine-tune
   └── pytorch_lora_weights.safetensors

在完成上述 LoRA Finetune 训练任务后，可以执行以下推理脚本，加载训练好的 lora 参数进行 FP32 精度下的推理。

.. code-block:: python

    from diffusers import StableDiffusionPipeline
    import torch
    import torch_tpu
    torch.manual_seed(42)
    device = torch.device("tpu:0")
    MODEL_NAME="CompVis/stable-diffusion-v1-4" #"/workspace/all_in_one/CompVis_SD14_pretrained_weights/133a221b8aa7292a167afc5127cb63fb5005638b"
    lora_weight_path = "sd-pokemon-model-lora_fp32"
    prompt = "cute dragon creature"
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
    pipe.unet.load_attn_procs(lora_weight_path)
    pipe.to(device)
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save("pokemon_lora.png")

执行如上示例代码后，会在当前的路径下生成一张名为 “pokemon_lora.png” 的图片。该节测试结果图片，可参考本节末尾，FP32与FP16的测试对比图。

使用 FP16 精度进行训练
------------------

执行下面的shell脚本：

.. code-block:: shell


    $ ulimit -n 65535
    $ ulimit -n
    $ export ACCELERATE_TORCH_DEVICE="tpu:0"
    $ export MODEL_NAME=CompVis/stable-diffusion-v1-4
    $ export DATASET_NAME=lambdalabs/pokemon-blip-captions

    $ python train_text_to_image_lora.py \
        --mixed_precision="fp16" \
        --pretrained_model_name_or_path=$MODEL_NAME \
        --dataset_name=$DATASET_NAME --caption_column="text" \
        --resolution=512 --random_flip \
        --train_batch_size=1 \
        --num_train_epochs=1 --checkpointing_steps=500 \
        --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
        --seed=42 \
        --output_dir="sd-pokemon-model-lora_fp16" \
        --validation_prompt="cute dragon creature"

执行如上指令，训练完成后，如 FP32 精度训练一样，会在当前目录下生成名为 sd-pokemon-model-lora_fp16 的文件夹。其中包含训练得到的lora参数、训练记录logs，以及checkpoint数据。文件结构如下。

.. code-block:: shell

   /text_to_image/sd-pokemon-model-lora_fp16/
   |—— checkpoint-*
   │    ├── optimizer.bin
   │    ├── pytorch_model.bin
   │    ├── random_states_0.pkl
   │    └── scaler.pt
   |—— logs
   │    └── text2image-fine-tune
   └── pytorch_lora_weights.safetensors

在完成上述 LoRA Finetune 训练任务后，可以执行以下推理脚本，加载训练好的 lora 参数进行 FP16 精度下的推理。

.. code-block:: python

    from diffusers import StableDiffusionPipeline
    import torch
    import torch_tpu
    torch.manual_seed(42)
    device = torch.device("tpu:0")
    MODEL_NAME="CompVis/stable-diffusion-v1-4" #"/workspace/all_in_one/CompVis_SD14_pretrained_weights/133a221b8aa7292a167afc5127cb63fb5005638b"
    lora_weight_path = "sd-pokemon-model-lora_fp16"
    prompt = "cute dragon creature"
    pipe = StableDiffusionPipeline.from_pretrained(MODEL_NAME, torch_dtype=torch.float16)
    pipe.unet.load_attn_procs(lora_weight_path)
    pipe.to(device)
    image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
    image.save("pokemon_lora.png")


执行如上示例代码后，也会在当前的路径下生成一张名为 “pokemon_lora.png” 的图片。

FP32 和 FP16 两种精度模式下，推理结果如下图所示。左侧图片为 FP32 精度下生成的图片，右侧图片为 FP16 精度下生成的图片。

这里需要注意一点，只要生成的图片有呈现出卡通形象即可判定上述步骤成功执行。

.. figure:: ../assets/4_lora_fp32_fp16.png
   :width: 1600px
   :height: 650px
   :scale: 50%
   :align: center
   :alt: SOPHGO LOGO
