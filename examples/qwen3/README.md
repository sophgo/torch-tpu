[English](./README.md) | [中文](./README_zh.md)

# Fine-tuning Qwen3-8B Using PAI-Megatron-Patch

## Step 1: Set up the Base Environment

### 1.1 Check Linux System Environment
Ensure that the IOMMU (Input-Output Memory Management Unit) service on your system is set to `translated` mode.

You can verify this by running the following command:
```bash
sudo dmesg | grep -i iommu
```
If the output indicates that the IOMMU type is **Translated**, your environment is correctly configured. Otherwise, please update your system configuration to set IOMMU to `translated` mode.

### 1.2 Prepare torch-tpu Environment

You can refer to the root directory's [README.md] or user manual [dist/docs/TORCH_TPU快速入门指南.pdf] to set up the `torch_tpu` environment.


## Step 2: Install torch_tpu in Docker Container
> **Important**: We strongly recommend using "Option 2" to obtain the latest `torch_tpu` installation package to avoid potential failures or performance issues that may occur with "Option 1".

### Option 1: Install torch_tpu via pip

```bash
pip install torch_tpu
```

### Option 2: Install torch_tpu Directly from Release Link (Recommended)
Pull the torch-tpu whl package from the FTP server's `torch_tpu/release_build/latest_release` directory and install it:
```
tar -xvf torch-tpu_*.tar.gz
pip install dist/torch_tpu-*_x86_64.whl --force-reinstall
```

## Step 3: Get PAI-Megatron-Patch

### 3.1 Clone the Repository

```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git /workspace/Pai-Megatron-Patch
cd /workspace/Pai-Megatron-Patch
# Switch to the specified version
git checkout b9fd9c2
git submodule update
```

### 3.2 Apply the Patch

```bash
# Assuming you are currently in the ../examples/qwen3 example directory
cp Pai-Megatron-Sophgo-Qwen3.patch /workspace/Pai-Megatron-Patch/
cd /workspace/Pai-Megatron-Patch
git apply Pai-Megatron-Sophgo-Qwen3.patch
```

## Step 4: Install Dependency Packages

After obtaining PAI-Megatron-Patch, you need to run the following command to install the required dependencies:

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Step 5: Get Base Model and Dataset

### 5.1 Download the Model

Download the `Qwen3-8B-Base` model from the [Hugging Face](https://huggingface.co/) website as the base model.

### 5.2 Download the Dataset

Execute the following commands to download the training dataset:

```bash
pip3 install dfss -i https://pypi.tuna.tsinghua.edu.cn/simple --upgrade
cd /workspace
python3 -m dfss --url=open@sophgo.com:SC11-FP300/LLM-finetune/qwen3-datasets/qwen3-datasets.zip
unzip qwen3-datasets.zip
```

### 5.3 Copy Execution Scripts

Copy the execution scripts from the current directory to the working path:

```bash
# Execute in the ../examples/qwen3 example directory
cp *.sh *.py /workspace/
```

### 5.4 Final File Directory Structure

After completing the above steps, the file directory structure should look like this:

```
/workspace/
├─ Pai-Megatron-Patch/          # PAI-Megatron-Patch code repository
│  └─ Pai-Megatron-Sophgo-Qwen3.patch
├─ qwen3-datasets/              # Training and validation sets
├─ Qwen3-8B-Base/               # Base model in HuggingFace format
├─ run_qwen3_train.sh           # Training script
├─ convert_qwen3_model.sh       # Model conversion script
├─ evalute_qwen3.sh             # Model evaluation script
├─ tgi_evaluate_qwen3.py        # TGI evaluation Python script
└─ ...
```

## Step 6: Fine-tuning Training

### 6.1 Convert Model Format (HuggingFace → MCore)

Before training, you need to convert the HuggingFace format model to MCore format:

```bash
source /workspace/convert_qwen3_model.sh
hf2mcore
```

### 6.2 Execute Fine-tuning Training

The training script defaults to 3000 iterations (approximately 2 epochs) and saves checkpoints every 1000 iterations:

```bash
bash /workspace/run_qwen3_train.sh
```

### 6.3 Convert Model Format (MCore → HuggingFace)

After training is complete, convert the MCore format model back to HuggingFace format for subsequent use:

```bash
source /workspace/convert_qwen3_model.sh
mcore2hf
```

### 6.4 File Directory After Training

After completing training, the following new directories will be generated:

```
/workspace/
├─ Pai-Megatron-Patch/
├─ ...                          # Same as above, omitted
├─ Qwen3-8B-to-mcore-tp2        # Base model in MCore format (TP=2)
├─ Qwen3-8B-sft-mcore           # Checkpoint models saved during training
├─ Qwen3-8B-sft-hf              # Fine-tuned model (HuggingFace format)
└─ ...
```

## Step 7: Evaluate the Model

You can choose any inference framework to evaluate the model according to your needs. This example uses **TGI-TPU** (Text Generation Inference for Sophgo TPU) for inference and uses **ROUGE scores** as the evaluation metric.

### 7.1 Environment Setup

For detailed information, refer to the `text-generation-inference_quick_start_zh.pdf` document in the `LLMs/text-generation-inference/release_build/latest_release` directory on the FTP server to set up the TGI-TPU Docker environment.

You only need to pull the corresponding Docker image, create and enter the container. Reference commands are as follows:

```bash
# Create container
docker run --privileged -itd --restart always \
  --name <CONTAINER_NAME> \
  --shm-size 1g \
  -p 8080:80 \
  -v $(pwd):/workspace \
  -v /dev/:/dev/ \
  -v /opt/tpuv7:/opt/tpuv7 \
  --entrypoint /bin/bash \
  soph_tgi:3.2.0-slim

# Enter container
docker exec -it <CONTAINER_NAME> bash
```

### 7.2 Run Inference and Evaluation

Run the following command in the TGI-TPU container to perform inference and evaluation:

```bash
bash /workspace/evaluate_qwen3.sh 
```

This script will:
- Start the TGI-TPU inference service
- Perform inference using the fine-tuned model
- Calculate ROUGE scores and output the evaluation results table
- Save the inference results as `/workspace/results.json` file
- The final ROUGE scores should be similar to the following:
```
==========================================================================================
                                 ROUGE evaluation results                                 
==========================================================================================
Metric       Recall                    Precision                 F-Measure                
------------------------------------------------------------------------------------------
ROUGE-1      0.5625 (±0.2373)          0.5997 (±0.2921)          0.5296 (±0.2446)         
ROUGE-2      0.3621 (±0.2358)          0.3867 (±0.2571)          0.3433 (±0.2320)         
ROUGE-L      0.4739 (±0.2353)          0.5004 (±0.2696)          0.4435 (±0.2345)         
==========================================================================================
```

## FAQ

### Q1: Dependency Package Conflicts

Ensure that `transformer_engine`, `apex`, and `megatron-core` are not installed. If they are installed, please uninstall them first:

```bash
python -m pip uninstall transformer_engine apex megatron-core -y
```

### Q2: Module 'megatron' Not Found

**Error message**: `ModuleNotFoundError: No module named 'megatron'`

**Solution**: The example script already includes commands to add `PYTHONPATH`. If it still reports that `megatron` cannot be found, please manually run:

```bash
export PYTHONPATH=/workspace/Pai-Megatron-Patch/backends/megatron/Megatron-LM-250624:$PYTHONPATH
```

### Q3: TGI-TPU Inference Errors

If you encounter errors when using TGI-TPU for inference, please first refer to section 4.2.1 of the `text-generation-inference_quick_start_zh.pdf` document to test whether the environment is set up successfully, to rule out TGI-TPU environment setup issues.

### Q4: How to Use Specified Chips

For example, if you have 8 chips and want to use the last 2 chips, i.e., chip6 and chip7, you can add the environment variable `CHIP_MAP=6,7` before the command. This environment variable will specify the chip numbers to be used for distributed training and inference.

---

If you have any other questions, please feel free to submit an Issue or contact the technical support team.
