#!/bin/bash
CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd )"
pushd $CUR_DIR/.. 
release_archive="./dist"
rm -rf ${release_archive}*

# ------------------------------------------------------------------------------
source scripts/envsetup.sh bm1684x stable
new_build
source scripts/release_doc.sh

# ------------------------------------------------------------------------------
BUILD_PATH=build/Release
torch_tpu_version="$(grep TORCHTPU_VERSION ${BUILD_PATH}/CMakeCache.txt | cut -d "=" -f2)"
tar -cvzf "torch-tpu_${torch_tpu_version}.tar.gz" ${release_archive}
rm -rf ${release_archive}

popd