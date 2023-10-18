How to gen dependency files for bm1684x?
0) cd nntoolchain/TPU1686&&git fetch&&git reset --hard origin/master
   #As nntc:master has not continued to merge latest TPU1686, please directly update 1686
1) Inside docker for nntoolchain:
	cd /workspace/nntoolchain/net_compiler

    export GLOG_v=4
	export GLOG_logtostderr=1
	export GLOG_log_prefix=false
	export FLAGS_log_prefix=false
	export BMLANG_CONFIG="DEBUG=1"
	export BMNETC_CONFIG="DEBUG=1"
	export BMNETU_CONFIG="DEBUG=1"
	export BMRUNTIME_CONFIG="DEBUG=1"
	export CPU_CONFIG="DEBUG=1"
	export DYNAMIC_CONTEXT_DEBUG=1
	source scripts/envsetup.sh
	rebuild_all_nntc
	2
	1
	3
	1


    cd /workspace/nntoolchain/TPU1686
    rm -rf build
    rm -rf build_test
    rm -rf build_runtime
    source scripts/envsetup.sh  bm1684x
	export EXTRA_CONFIG=-DDEBUG=ON
	rebuild_all
	rebuild_test sgdnnn
	rebuild_firmware
	Y


	cd /workspace/nntoolchain
	rm -rf target&&mkdir target
    cp ./net_compiler/out/install/lib/libbmlib.so target/.
    cp ./TPU1686/build_test/firmware_core/libcmodel_firmware.so ./target/
    cp ./TPU1686/build/firmware_core/libfirmware_core.a ./target/

    cd  /workspace/nntoolchain/target
    cp libbmlib.so libbmlib_cmodel.so
    mv libfirmware_core.a libbm1684x.a
    ln -s libbmlib.so libbmlib.so.0
    cd ..

###############################################
2) Inside docker for tpu-train:
   mv /workspace/tpu-train/third_party/bm1684x/README.md  /workspace/tpu-train/third_party/README_bm1684x.md
   rm -rf /workspace/tpu-train/third_party/bm1684x
   cp -r /workspace/nntoolchain/target /workspace/tpu-train/third_party/bm1684x
   mv /workspace/tpu-train/third_party/README_bm1684x.md  /workspace/tpu-train/third_party/bm1684x/README.md
   chmod -R 777 /workspace/tpu-train/third_party/bm1684x/

###############################################
Or) Copy Jump Inside docker for tpu-train:
    cp /workspace/nntoolchain/TPU1686/build_test/firmware_core/libcmodel_firmware.so       /workspace/tpu-train/third_party/bm1684x/libcmodel_firmware.so
    cp /workspace/nntoolchain/TPU1686/build/firmware_core/libfirmware_core.a               /workspace/tpu-train/third_party/bm1684x/libbm1684x.a