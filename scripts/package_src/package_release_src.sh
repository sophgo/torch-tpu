pushd $TPUTRAIN_TOP/dist


package_src=src
setup_path=$TPUTRAIN_TOP/scripts/package_src/setup.py
readme_path=$TPUTRAIN_TOP/scripts/package_src/README.md
binding_src=$TPUTRAIN_TOP/torch_tpu/csrc/InitTpuBindings.cpp

# ==================================================================
echo "[step-1] package torch-tpu"

keep_file=*.whl
find . -mindepth 1 ! -name "$keep_file" -exec rm -rf {} +;

mkdir $package_src
cp $readme_path     $package_src/

unzip $keep_file
mv torch_tpu        $package_src/
cp $setup_path      $package_src/torch_tpu/
cp $binding_src     $package_src/torch_tpu/
rm $package_src/torch_tpu/*.so
rm -rf *.egg-info
rm -rf *.dist-info

# ==================================================================
echo "[step-2] package third-party include"
src_dir=../third_party
dest_dir=$package_src/third_party
mkdir $dest_dir
find "$src_dir" -type f -name "*.h" -exec cp --parents {} "$dest_dir" \;

# ==================================================================
echo "[step-3] package sgdnn include"
src_dir=../sgdnn
dest_dir=$package_src/sgdnn
mkdir $dest_dir
find "$src_dir" -type f -name "*.h" -exec cp --parents {} "$dest_dir" \;
find "$src_dir" -type f -name "*.hpp" -exec cp --parents {} "$dest_dir" \;
cp ../common/include/* $dest_dir/include

# ==================================================================
echo "[step-4] tar src to torch_tpu_src.tar.gz"
tar -cvzf "torch-tpu_src.tar.gz" ${package_src}
rm -rf ${package_src}

popd