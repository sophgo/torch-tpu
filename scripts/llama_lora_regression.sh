function release_tpu-train() {
    pushd $TPUTRAIN_TOP/scripts > /dev/null || return 1
    trap "popd" RETURN
    bash release.sh || return 1
    return 0
}

function install_requirements() {
    local torch_tpu_dist_path=$(python3 -m pip show torch_tpu | grep Location | awk '{print $2}')
    pushd ${torch_tpu_dist_path}/torch_tpu/demo/LLaMA-2_LoRA_Finetune > /dev/null || return 1
    trap "popd" RETURN
    python3 -m pip install -r requirements.txt || return 1
    return 0
}

function llama_lora_test() {
    pushd $TPUTRAIN_TOP/python/gen_ins > /dev/null || return 1
    trap "popd" RETURN
    python3 llama_lora_full.py || return 1
    return 0
}

function llama_lora_regression() {

    # release_tpu-train || return 1
    install_requirements || return 1
    tpu_apply_all_patch || return 1
    llama_lora_test || return 1
    return 0
}