#!/bin/bash
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd )"
pushd $CUR_DIR/..

# ------------------------------------------------------------------------------
source scripts/envsetup.sh
export RELEASE_MODE=ON
TORCH_TPU_NONINTERACTIVE=ON python setup.py clean
# bash scripts/auto_generate_patch.sh || exit -1
bdist_wheel || exit -1
mkdir -p release
cp dist/*.whl release/ || exit 1
unset RELEASE_MODE

source scripts/envsetup.sh sg2260
# build_riscv_whl || exit -1
cp release/*.whl dist/
rm -r release

## docs
if [ -z $SKIP_DOC ]; then
    source scripts/release_doc.sh
fi

## package example
pushd $CUR_DIR/../dist
cp -rf $CUR_DIR/../examples .
cp -rf $CUR_DIR/../python/regression .
popd

## src
source scripts/package_src/package_release_src.sh

# ------------------------------------------------------------------------------
release_archive="./dist"
BUILD_PATH=build/torch-tpu
torch_tpu_version="$(grep TORCHTPU_VERSION ${BUILD_PATH}/CMakeCache.txt | cut -d "=" -f2)"
commit=$(git log -1 --pretty=format:"%h")
tar -cvzf "torch-tpu_${torch_tpu_version}_${commit}.tar.gz" ${release_archive}
rm -rf ${release_archive}

popd
