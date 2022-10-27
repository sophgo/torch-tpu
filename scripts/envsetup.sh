#!/bin/bash

function gettop
{
  local TOPFILE=scripts/envsetup.sh
  if [ -n "$TOP" -a -f "$TOP/$TOPFILE" ] ; then
    # The following circumlocution ensures we remove symlinks from TOP.
    (cd $TOP; PWD= /bin/pwd)
  else
    if [ -f $TOPFILE ] ; then
      # The following circumlocution (repeated below as well) ensures
      # that we record the true directory name and not one that is
      # faked up with symlink names.
      PWD= /bin/pwd
    else
      local HERE=$PWD
      T=
      while [ \( ! \( -f $TOPFILE \) \) -a \( $PWD != "/" \) ]; do
        \cd ..
        T=`PWD= /bin/pwd -P`
      done
      \cd $HERE
      if [ -f "$T/$TOPFILE" ]; then
        echo $T
      fi
    fi
  fi
}

function printpath()
{
  T=$(gettop)
  if [ ! "$T" ]; then
    echo "Couldn't locate the top of the tree.  Try setting TOP." >&2
    return
  fi
  echo $T
}

function set_cmodel_firmware {
     export TPUKERNEL_FIRMWARE_PATH=`realpath $1`
}

TRAIN_TOP=$(gettop)
# printpath
echo "TRAIN_TOP = $TRAIN_TOP"
export TRAIN_TOP
LD_LIBRARY_PATH=
OS_TYPE=`cat /etc/os-release | grep ^"NAME=" | cut -d = -f 2`
if [ -z "$PREBUILT_DIR" ]; then
	export PREBUILT_DIR=$TRAIN_TOP/../bm_prebuilt_toolchains
fi
case ${OS_TYPE[*]:1:-1} in
  "CentOS Linux")
    export LD_LIBRARY_PATH=$PREBUILT_DIR/x86-64-core-i7--glibc--stable/x86_64-buildroot-linux-gnu/lib64:$LD_LIBRARY_PATH
    export LD_LIBRARY_PATH=$PREBUILT_DIR/x86-64-core-i7--glibc--stable/x86_64-buildroot-linux-gnu/sysroot/usr/lib:$LD_LIBRARY_PATH;;
  *) ;;
esac
export LD_LIBRARY_PATH=$PREBUILT_DIR/x86-64-core-i7--glibc--stable/lib:$LD_LIBRARY_PATH

source ${TRAIN_TOP}/scripts/build_helper.sh
