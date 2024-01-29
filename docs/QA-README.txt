按照 《快速入门指南》进行测试。

其中，“模型训练实例_StableDiffusion“一节当中，需要从huggingface中下载模型和数据集。

如果，网络条件不好下载不了模型和数据集。

可以从https://disk.sophgo.vip/sharing/igJxL3Ymn下载。其中，压缩包内各个文件夹对应的内容如下：

    - CompVis_SD14_pretrained_weights: "CompVis/stable-diffusion-v1-4"的预训练模型参数
    - dataset: "lambdalabs/pokemon-blip-captions"数据集
    - SophgoTrained_lroa_weight: 之前使用bm1684x训练得到的lora权重
    - sayakpau_sd_lora-t4: "sayakpaul/sd-model-finetuned-lora-t4"训练得到的lora权重
    - torch_tpu: torch_tpu的wheel包 （已过时，不用）
    - libsophon: libsophon的安装包 （已过时，不用）