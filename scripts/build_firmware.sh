#!/bin/bash

echo "build firmware!"

if [ "$#" -lt 1 ]; then
  echo "Usage: $0 "device_code_dir1,device_code_dir2" [output_dir]"
  exit -1
fi

DEVICE_DIR=$1
#OUTPUT_DIR=${2:-$TPUKERNEL_TOP/out}
OUTPUT_DIR=${2:-$TRAIN_TOP/out}
mkdir -p ${OUTPUT_DIR}/device

CURRENT_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)


mkdir -p ${OUTPUT_DIR}
IFS=',' read -ra DIRS<<< "${DEVICE_DIR}"
for i in "${DIRS[@]}"; do
  echo "process $i"
  if [ -d "$i" ]; then
    cp -r $i/* ${OUTPUT_DIR}/device
  elif [ -f "$i" ]; then
    cp $i ${OUTPUT_DIR}/device
  fi
done

cp -r ${CURRENT_DIR}/Makefile.device ${OUTPUT_DIR}
cp -r ${CURRENT_DIR}/target-ram.lds ${OUTPUT_DIR}
pushd ${OUTPUT_DIR}
make -f Makefile.device
if [ $? -ne 0 ]; then echo "Failed to make firmware!"; exit -1; fi
#source ${CURRENT_DIR}/version_info.sh
#python3 ${CURRENT_DIR}/firmware_pack.py bm1684x.bin .
#rm -rf bm1684x.bin
#rm -rf build
#rm -rf device
popd
