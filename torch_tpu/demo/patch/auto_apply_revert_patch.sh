# auto apply patch to the certain package
# set -ex
export PATCH_TOP=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)

function info()
{
    echo -e "\033[32m[INFO] $1\033[0m"
}

function error()
{
    echo -e "\033[31m[ERROR] $1\033[0m"
}

function warning()
{
    echo -e "\033[33m[WARNING] $1\033[0m"
}

info "export PATCH_TOP=${PATCH_TOP}"

function check_version()
{
    package_name=$1
    version=$2
    # check if package_name exists must match version with version
    if [ -z "$(pip3 show ${package_name} | grep Version | awk '{print $2}')" ]; then
        error "${package_name} not found, please install it first with version ${version}"
        exit 1
    else
        if [ "$(pip3 show ${package_name} | grep Version | awk '{print $2}')" != "${version}" ]; then
            error "${package_name} found, but version mismatch, please install it with version ${version}"
            exit 1
        fi
        info "${package_name} found, version: $(pip3 show ${package_name} | grep Version | awk '{print $2}')"
    fi
}

function check_demo_package_version()
{
    info "check_demo_package_version and auto install if not found"
    # pip3 install accelerate==0.30.1
    check_version accelerate 0.30.1
    # pip3 install transformers==4.41.2
    check_version transformers 4.41.2
    # pip3 install git+https://github.com/hiyouga/LLaMA-Factory.git@v0.8.3
    check_version llamafactory 0.8.3
}

function apply_patch()
{
    check_demo_package_version;
    # apply patch to certain package
    info "apply_patch. Current we have accelerate, transformers, llama-factory patch"
    # patch accelerate
    info "PATCH_TOP=${PATCH_TOP}"   
    accelerate_path=$(pip3 show accelerate | grep Location | awk '{print $2}')
    pushd ${accelerate_path}"/accelerate"
    patch -p3  --dry-run < ${PATCH_TOP}/accelerate-Sophgo.patch
    popd
    # patch transformers
    transformers_path=$(pip3 show transformers | grep Location | awk '{print $2}')
    pushd ${transformers_path}"/transformers"
    patch -p3  --dry-run < ${PATCH_TOP}/transformers-Sophgo.patch
    popd
    # patch llama-factory
    llama_factory_path=$(pip3 show llamafactory | grep Location | awk '{print $2}')
    pushd ${llama_factory_path}"/llamafactory"
    patch -p3  --dry-run < ${PATCH_TOP}/LLaMA-Factory-Sophgo.patch
    popd
    info "apply_patch done"
}

function revert_patch()
{
    check_demo_package_version;
    # revert patch to certain package
    echo "[INFO] revert_patch. Current we have accelerate, transformers, llama-factory patch"
    # revert accelerate
    accelerate_path=$(pip3 show accelerate | grep Location | awk '{print $2}')
    pushd ${accelerate_path}"/accelerate"
    patch -p3 -R < ${TPUTRAIN_TOP_DIR}/accelerate-Sophgo.patch
    popd
    # revert transformers
    transformers_path=$(pip3 show transformers | grep Location | awk '{print $2}')
    pushd ${transformers_path}"/transformers"
    patch -p3 -R < ${TPUTRAIN_TOP_DIR}/transformers-Sophgo.patch
    popd
    # revert llama-factory
    llama_factory_path=$(pip3 show llamafactory | grep Location | awk '{print $2}')
    pushd ${llama_factory_path}"/llamafactory"
    patch -p3 -R < ${TPUTRAIN_TOP_DIR}/LLaMA-Factory-Sophgo.patch
    popd
    echo "[INFO] revert_patch done"
}

# auto apply patch to the certain package


# apply_patch;
check_demo_package_version