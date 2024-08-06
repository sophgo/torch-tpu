set -ex
TPUTRAIN_TOP=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)
function command_with_check()
{
    command=$1
    ret=$(eval $command)
    if [ $? -ne 0 ]; then
        echo "Error: $command"
        exit 1
    fi
}


function generate_patch()
{
    package_name=$1
    target_dir=$TPUTRAIN_TOP/torch_tpu/demo/patch/
    echo "Current patch"
    pushd $TPUTRAIN_TOP/../$package_name
    ret=`git diff HEAD remotes/origin/sophgo > $target_dir"${package_name}-Sophgo.patch"`
    popd
}

# generate_patch transformers
# generate_patch accelerate
# generate_patch LLaMA-Factory