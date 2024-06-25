#!/bin/bash
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd )"
pushd $CUR_DIR/..

# ------------------------------------------------------------------------------
source scripts/envsetup.sh
export RELEASE_MODE=ON
python setup.py clean
bdist_wheel

if [ ! -z $SKIP_DOC ]; then
    source scripts/release_doc.sh
fi

# ------------------------------------------------------------------------------
BUILD_PATH=build/torch-tpu
torch_tpu_version="$(grep TORCHTPU_VERSION ${BUILD_PATH}/CMakeCache.txt | cut -d "=" -f2)"
commit=$(git log -1 --pretty=format:"%h")
tar -cvzf "torch-tpu_${torch_tpu_version}_${commit}.tar.gz" ${release_archive}
rm -rf ${release_archive}

popd
