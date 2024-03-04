#!/bin/bash

function gen_gpt3_32ch_all() {
    echo "*********** Gen gpt3 32ch patterns all ***************"
    if [ ! -d "ins_gpt" ]; then
        mkdir -p "ins_gpt"
        echo "文件夹已创建"
    else
        echo "文件夹已存在"
    fi
    cd ins_gpt
    python3 ../python/gen_ins/gpt3block_TP16_fb.py

    #删除冗余文件
    rm -rf LayerNorm_embedded_input FC_QKV Matmul_QK Norm_QK\[Div\]  Get_QK_MASK/ WHERE_ON_QK/ SOFTMAX_QK/ Matmul_QKV Add_atten/ LayerNorm_atten/ FC_mlp0/ SOFTMAX_QK_d_/ WHERE_ON_dqkv_/  Norm_QK\[Div\]_d_/
    rm -rf GDMA_reg.* SDMA_reg.*
    echo "*********** GEN END ***************"
}