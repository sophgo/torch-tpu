function copy_file(){
    src_file=$1
    dst_dir=$2
    if [ ! -f $src_file ] ; then
        echo "Not Found $src_file"
        return -1
    fi
    cp $src_file $dst_dir
    echo "Copy $src_file To $dst_dir"
}
function copy_build_binary(){
    dst_dir=$TPUTRAIN_TOP/python/tpu_plugin/lib
    if [ "${CHIP_ARCH}" = "sg2260" ]; then
        echo "chip 2260 tpu-lplugin is 'invalid'"
    else
        if [ ! -d $dst_dir ] ; then
            mkdir -p $dst_dir
        fi
        copy_file $TPUTRAIN_TOP/libtorch_plugin/build/liblibtorch_plugin.so              $dst_dir
        copy_file $TPUTRAIN_TOP/build/sgdnn/libsgdnn.so                                  $dst_dir
        copy_file $TPUTRAIN_TOP/build/firmware_core/lib${CHIP_ARCH}_kernel_module.so     $dst_dir
        copy_file $TPUTRAIN_TOP/third_party/$CHIP_ARCH/lib$CHIP_ARCH.a                   $dst_dir
    fi
}

function clean_python_pack(){
    src_dir=$TPUTRAIN_TOP/python
    pushd $src_dir
    rm -rf build
    rm -rf dist
    rm -rf tpu_plugin.egg-info
    popd

    src_dir=$TPUTRAIN_TOP
    pushd $src_dir
    rm -f out/*.whl
    popd
}

function python_pack(){
    src_dir=$TPUTRAIN_TOP/python
    dst_dir=$TPUTRAIN_TOP/out
    mkdir -p $dst_dir
    pushd $src_dir
    rm -rf build
    rm -f dist/*.whl
    python3 setup.py bdist_wheel --plat-name linux_x86_64
    ret=$?; if [ $ret -ne 0 ]; then return $ret; fi
    if [ -n "$dst_dir" ]; then
        mkdir -p "$dst_dir"
        cp dist/*.whl  "$dst_dir"
    fi
    popd
}

function build_bdist_wheel(){
    copy_build_binary;
    python_pack;
}

function rebuild_bdist_wheel(){
    pushd $TPUTRAIN_TOP/python/tpu_plugin
    rm -rf lib
    popd
    build_bdist_wheel;
}