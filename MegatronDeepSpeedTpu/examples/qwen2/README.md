# Qwen2-7B Pretrain

## Prequisites

- Create a docker container with sophgo/torch_tpu:latest image and make sure the latest version of torch_tpu and the tpuv7-runtime library is installed.

- Clone the Pai-Megatron-Patch repository
```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch
git checkout v0.9.1
```

- Apply the patch
```bash
cp Pai-Megatron-Sophgo.patch  /workspace/Pai-Megatron-Patch/
cd /workspace/Pai-Megatron-Patch
git apply Pai-Megatron-Sophgo.patch
```

- Install MegatronDeepSpeedTpu
```bash
cd MegatronDeepSpeedTpu
python -m pip install -e .
```

- Install the required dependencies
```bash
python -m pip install pybind11 transformers==4.41.2 accelerate==0.30.1 datasets
```

- Make sure you have no installation of transformer_engine, apex and megatron-core. If you have, uninstall them
```bash
python -m pip uninstall transformer_engine apex megatron-core -y
```

- Download the checkpoint and dataset according to the instructions in `Pai-Megatron-Patch/examples/qwen2`

- Move the checkpoint, dataset folder and script `run_qwen2_train.sh` into Pai-Megatron-Patch directory. Now your directories will be arranged like this:
```
    /workspace/
    ├─ Pai-Megatron-Patch/
    │  ├─ qwen-ckpts/
    │  ├─ qwen-datasets/
    |  ├─ Pai-Megatron-Sophgo.patch
    │  ├─ run_qwen2_train.sh
    │  └─ ...
    ├─ ...
```

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