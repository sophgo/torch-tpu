get_pytorch_install_dir(){
     pytorch_path=$(python3 -c \
                    "import torch; \
                     import os; \
                     print(os.path.dirname(os.path.realpath(torch.__file__))) \
                    ")
     export PYTORCH_INSTALL_DIR=${pytorch_path}
}

get_tputrain_install_dir(){
    tputrain_path=$(python3 -c \
                   "import torch_tpu; \
                    import os; \
                    print(os.path.dirname(os.path.realpath(torch_tpu.__file__))) \
                   ")
    export TPUTRAIN_INSTALL_TOP=${tputrain_path}
}

prepare_dev_env(){
    export SOC_CROSS=OFF
    get_pytorch_install_dir
    if [ -d "$TPUTRAIN_TOP/build/Release/packages/torch_tpu" ]; then
        export TPUTRAIN_INSTALL_TOP=$TPUTRAIN_TOP/build/Release/packages/torch_tpu
    elif [ -d "$TPUTRAIN_TOP/build/Debug/packages/torch_tpu" ]; then
        export TPUTRAIN_INSTALL_TOP=$TPUTRAIN_TOP/build/Debug/packages/torch_tpu
    else
        echo "Error: Neither 'Release' nor 'Debug' version of torch_tpu found under $TPUTRAIN_TOP/build/"
        return 1
    fi
}

prepare_cross_env(){
    export SOC_CROSS=ON
    export TPUTRAIN_TOP=/workspace/tpu-train/dist/torch_tpu
    export PYTORCH_INSTALL_DIR=/workspace/tpu-train/toolchains_dir/torchwhl/torch
    export ARM_COMPILE=/workspace/tpu-train/toolchains_dir/gcc-arm-10.3-2021.07-x86_64-aarch64-none-linux-gnu
}

prepare_soc_env(){
    export CHIP_ARCH="bm1684x"
    export SOC_CROSS=OFF
    get_pytorch_install_dir
    get_tputrain_install_dir
}

run_env(){
    export LD_LIBRARY_PATH=${PYTORCH_INSTALL_DIR}/lib:${TPUTRAIN_TOP}/lib:${TPUTRAIN_TOP}/libtorch:${LD_LIBRARY_PATH}
}