#!/bin/bash

function sg2260e_online_regression_test() {
    CURRENT_DIR=$(dirname ${BASH_SOURCE})

    source $CURRENT_DIR/envsetup.sh sg2260e || return -1

    new_clean

    develop_torch_tpu || return -1

    bdist_wheel || return -1
}

sg2260e_online_regression_test
