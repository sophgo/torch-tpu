# function install_requirements() {
#     local torch_tpu_dist_path=$(python3 -m pip show torch_tpu | grep Location | awk '{print $2}')
#     pushd ${torch_tpu_dist_path}/torch_tpu/demo/LLaMA-2_LoRA_Finetune > /dev/null || return 1
#     PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple python3 -m pip install -r requirements.txt || return 1
#     popd > /dev/null
#     return 0
# }

function install_requirements() {
    for repo in accelerate transformers LLaMA-Factory
    do
        PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple python3 -m pip install ${CURRENT_DIR}/../../${repo}/ || return 1
    done
}

function llama_lora_test() {
    pushd $CURRENT_DIR/../python/gen_ins > /dev/null || return 1
    python3 llama_lora_full.py || return 1
    popd > /dev/null
    return 0
}

function llama_lora_regression() {

    CURRENT_DIR=$(dirname ${BASH_SOURCE})
    echo "********************************************"
    echo "[INFO]LLaMA-LoRA regression daily test"
    echo "[INFO]CURRENT_DIR:$CURRENT_DIR"

    SKIP_DOC=true
    bash scripts/release.sh || return 1
    install_requirements || return 1
    # while true
    # do 
    #     install_requirements && break
    # done
    tpu_apply_all_patch || return 1
    llama_lora_test || return 1
    return 0
}