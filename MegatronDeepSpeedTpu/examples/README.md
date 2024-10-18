# Examples for MegatronDeepSpeedTpu

We provide two examples for MegatronDeepSpeedTpu: bert and gpt.

`gen_dist_cmd.py` will generate the distributed training command for you.  You can run the following command to generate the command:

```shell
python gen_dist_cmd.py [bert|gpt]
```

We suppose that the distributed training is running on 4 servers, as the default values in the script.  You can change the values in the script to fit your own settings.

## bert

An example of pretraining bert model.  Training config can be set in `pretrain_bert.py`.

You will need no preparations, just run the generated commands on each server.

Don't forget to change directory to `examples/bert` before running the command.

## gpt

An example of pretraining gpt model.  Training config can be set in `run_distributed_gpt_train.sh`.  The default setting is training GPT-13B.

The gpt example requires the data to be downloaded and processed.  You can generate the data by running the following command:

```shell
python gen_dataset.py [--limit 1000000]
```

The dataset is very big so you can pass the `--limit` argument to limit the size of the dataset.

Then, run the generated commands on each server.
Don't forget to change directory to `examples/gpt` before running the command.

## byobu

Copy-pasting commands to each server is annoying.  You can run the examples using `byobu` simply.

To install `byobu`, run `sudo apt install byobu`.  Then, run the following command on each server:

```shell
byobu new-session -s tpu-train -n DeepSpeed
```

In byobu windows, enter the docker in each server, source the environments and change directory to `experts/[bert|gpt]` on each server.

Configure the ssh connection to each server and run the following command:

```shell
python gen_dist_cmd.py --execute [bert|gpt]
```

The training will be automatically started on each server.

## debug

For debug purpose, we have inserted several debug code into the examples.  By default, it will only print logs when doing forward and backward and losses.
You can set environment variable `DBG_SAVE_ALL=1` to save the inputs and outputs of each layer, the model parameters and the model grads.
You can also set `DBG_SAVE_RESULTS=1`, `DBG_SAVE_PARAMS=1` and `DBG_SAVE_GRADS=1` separately.
You can pass these environment variables in `--extra_flags` to `gen_dist_cmd.py` to generate the command.

When using cuda for reference data, you should also pass `DS_DISABLE_TPU=1` to disable TPU and only use the debugger.  `gen_dist_cmd.py` will generate the correct command for you if you pass `--device cuda`.

For example, to save all the tensors and use cuda for reference data, you can run the following command:

```shell
python gen_dist_cmd.py --device cuda --extra_flags "DBG_SAVE_ALL=1" [bert|gpt]
```