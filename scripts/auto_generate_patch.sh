set -ex
TPUTRAIN_TOP=$(cd $(dirname "${BASH_SOURCE[0]}")/.. && pwd)


function generate_patch()
{
    local target_dir=$TPUTRAIN_TOP/torch_tpu/demo/patch
    local package_name=$1
    echo "Current patch:" ${package_name}
    local package_dir=$TPUTRAIN_TOP/../${package_name}
    # we assume that the package is up-to-date in the same directory as torch-tpu
    if [ ! -d ${package_dir} ]; then\
        echo "Error: ${package_dir} not exist"
        return 1
    fi
    pushd ${package_dir} || return 1
    # git fetch origin || return 1
    git reset --hard origin/master || return 1
    git diff HEAD remotes/origin/sophgo > ${target_dir}/${package_name}-Sophgo.patch || return 1
    
    local commit_id=$(git rev-parse HEAD)
    local tag=$(git describe --tags HEAD 2>/dev/null)
    sed -i "s/${package_name}_commit_id/${commit_id}/g" ${target_dir}/README.md || return 1
    sed -i "s/${package_name}_commit_tag/${tag}/g" ${target_dir}/README.md || return 1
    echo "Patch generated for" ${package_name}
    popd
    return 0
}

for package in transformers accelerate LLaMA-Factory
do
    generate_patch $package || exit -1
done