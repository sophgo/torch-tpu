[English](./README.md) | [中文](./README_zh.md)

# Qwen2-7B Pretrain Using PAI-Megatron-Patch

## Step1: Set the environment

### Option 1: Check the Linux system environment

Ensure that the IOMMU (Input-Output Memory Management Unit) service on your system is set to translated mode.

You can verify this by running the following command: `sudo dmesg | grep -i iommu`<br>
If the output indicates that the IOMMU type is `Translated`, your environment is correctly configured.
Otherwise, please update your system configuration to enable IOMMU in translated mode.<br>

### Option 2: Prepare base torch-tpu environment

You can refer to [README.md] or UserGuide to prepare base torch-tpu environment.


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
git submodule update
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

## Step4: Install dependency packages

After you get the PAI-Megatron-Patch, you should install dependency packages by running the following command:

```bash
pip install pybind11 transformers==4.41.2 accelerate==0.30.1 datasets netifaces nnmoduletools>=0.1.1 evaluate sacrebleu scikit-learn sqlitedict peft==0.10.0 pytablewriter
# pip install pybind11 transformers==4.41.2 accelerate==0.30.1 datasets netifaces nnmoduletools>=0.1.1 evaluate sacrebleu scikit-learn sqlitedict peft==0.10.0 pytablewriter -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Step5: Get dataset and checkpoint

- Download the dataset according to the instructions in `Pai-Megatron-Patch/examples/qwen2`
- Download the Qwen2-7B checkpoint as reference model from huggingface, and modify `"torch_dtype"` parameter in `config.json` file from `bfloat16` to `float16`.

- Move the checkpoint, dataset folder, script `run_qwen2_train.sh` and script `run_qwen2_evaluation.sh` into Pai-Megatron-Patch directory. Now your directories will be arranged like this:
```
    /workspace/
    ├─ Pai-Megatron-Patch/
    │  ├─ qwen-ckpts/
    |  |  ├─Qwen2-7B
    |  |  └─ ...
    │  ├─ qwen-datasets/
    |  ├─ Pai-Megatron-Sophgo.patch
    │  ├─ run_qwen2_train.sh
    |  ├─ run_qwen2_evaluation.sh
    │  └─ ...
    ├─ ...
```


## Step6: Run training script and save checkpoints

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


## Step7: Convert checkpoints from MegatronCore to HuggingFace format and evaluate checkpoints

- Source the evaluation script environmrnt

```bash
source run_qwen2_evaluation.sh
```

- The arguments that need to be set are:  the path for the HuggingFace checkpoints to be saved, the path for the MegatronCore checkpoints saved during training (default), and the reference model path downloaded from HuggingFace
- Using default arguments and run to get HuggingFace format checkpints:

```bash
mcore_to_hg
```
- Download the evaluation datasets according to the instructions in `Pai-Megatron-Patch/examples/qwen2` or you can directly run:

```bash
cd /workspace
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/evaluation-datasets/ceval.tgz
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/evaluation-datasets/evaluate.tgz
tar -xvzf ceval.tgz
tar -xvzf evaluate.tgz
```

- The arguments that need to be set are: the path for the saved HuggingFace format checkpoints.
- Using default arguments and evaluate:

```bash
evaluate
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

- I ran the script and got an error `ModuleNotFoundError: No module named 'megatron'`:

We have added the command to modify `PYTHONPATH` in the script.  If it still complains that `megatron` cannot be found, try running the following command under `Pai-Megatron-Patch` directory:

```bash
export PYTHONPATH=$(pwd)/PAI-Megatron-LM-240718:$PYTHONPATH
```
