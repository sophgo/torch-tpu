
function gen_GeLU()
{
    local ret=0
    echo "##########################################################################"
    echo "Dump GeLU"
    echo ""
    execl=${TEST_DIR:-$SG1684X_TOP/build_test}/test/test_api_sgdnn/test_gelu_multi_core
    [ -f $execl ] || {
        echo "$execl does not exist. You may need to cd into build_test dir."
        exit -1
    }
    execl=$(realpath $execl)

    new_dir GeLU
    if [ -n "$COMPARE" ]; then
        FILE_DUMP_CMD=GeLU $execl --batch 1 --channel c --height 1 --width 128 --compare 1 || ret=$?
    else
        FORBID_CMD_EXECUTE=1 FILE_DUMP_CMD=GeLU $execl --batch 1 --channel 128 --height 1 --width 128 --compare 0 || ret=$?
    fi
    popd > /dev/null

    return $ret
}