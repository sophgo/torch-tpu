[English](./README.md) | [中文](./README_zh.md)

# Qwen2-7B Pretrain Using PAI-Megatron-Patch

## Step1: prepare base torch-tpu environment

You can refer to [README.md](../../README.md) to prepare base torch-tpu environment.


## Step2: Install `torch_tpu` in the docker container

### Option 1: Install `torch_tpu` by pip

```
pip install torch_tpu
```

### Option 2: Install `torch_tpu` from released wheel file

```
<!-- TODO -->
pip install https://github.com/sophgo/torch-tpu/releases/download/v0.1.0/torch_tpu-0.1.0-cp310-cp310-linux_x86_64.whl
```

## Step3: Get PAI-Megatron-Patch

### Option 1: Get code by git apply

- Clone the Pai-Megatron-Patch repository

```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git /workspace/Pai-Megatron-Patch
cd /workspace/Pai-Megatron-Patch
git checkout v0.9.1
```

- Apply the patch
```bash
cp Pai-Megatron-Sophgo.patch  /workspace/Pai-Megatron-Patch/
cd /workspace/Pai-Megatron-Patch
git apply Pai-Megatron-Sophgo.patch
```

### Option 2: Get code by released demo

<!-- TODO: add the link of the released demo -->
- You can also download full PAI-Megatron-Patch in [here](https://github.com/sophgo/torch-tpu/)

## Step4: Install megatron-patch

After you get the PAI-Megatron-Patch, you should  install `megatron-patch` inside Pai-Megatron-Patch directory by running the following command:

```bash
pip install pybind11 transformers==4.41.2 accelerate==0.30.1 datasets netifaces "nnmoduletools>=0.1.1"
# pip install pybind11 transformers==4.41.2 accelerate==0.30.1 datasets netifaces nnmoduletools>=0.1.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
cd /workspace/Pai-Megatron-Patch/PAI-Megatron-LM-240718
python setup.py develop --user
```

You can check if the installation is successful by running the following command:
```bash
python -c "import megatron; print(megatron.__path__)"
```

## Step5: Get dataset and checkpoint

- Download the checkpoint and dataset according to the instructions in `Pai-Megatron-Patch/examples/qwen2`

- Move the checkpoint, dataset folder and script `run_qwen2_train.sh` into Pai-Megatron-Patch directory. Now your directories will be arranged like this:
```
    /workspace/
    ├─ TPU-Megatron-Patch/
    │  ├─ qwen-ckpts/
    │  ├─ qwen-datasets/
    |  ├─ Pai-Megatron-Sophgo.patch
    │  ├─ run_qwen2_train.sh
    │  └─ ...
    ├─ ...
```


## Step6: Run training script

- Set the hyperparameters and run the pretraining script

The arguments are: full model size, layers (0 for full), TP and PP.

To pretrain 7B model with TP=2, run

```bash
source run_qwen2_train.sh 7B 0 2 1
```

To run a smaller test, for example a 1-layer 7B model with TP=2, run

```bash
source run_qwen2_train.sh 7B 1 2 1
```


## Other suggestions


- in cmodel mode, you'd better set these environment variables to get better performance

```bash
export CMODEL_GLOBAL_MEM_SIZE=120000000000
export CMODEL_FAST_EXEC=1
```

## FAQ

- Make sure you have no installation of transformer_engine, apex and megatron-core. If you have, uninstall them

```bash
python -m pip uninstall transformer_engine apex megatron-core -y
```
