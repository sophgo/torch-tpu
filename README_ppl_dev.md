PPL 算子对接 Torch-TPU
======================

### 注意事项

+ *SG2260E* 运行时默认使用 PPL 算子，*SG2260* 运行时默认使用手写算子；可以通过 USE\_PPL=0 或 USE\_PPL=1 关闭或打开 PPL 算子执行；
+ 切换芯片类型后，一定要删除 build 目录重新编译。

### 如何编译

脚本将依次从以下目录寻找 ppl 程序自动赋值到 `PPL_INSTALL_PATH` 环境变量：

+ tpu-train 同级以 `ppl` 开头的目录；
+ `tpu-train/third_party/ppl` 目录。

所以，可以按以下任一种方式配置 PPL 编译器：

+ 将 PPL Release 包解压到 tpu-train 同级目录；
+ `git lfs pull --include 'third_party/ppl/**/*' --exclude ''` 拉取；
+ 也可以在 `source envsetup.sh` 前手动指定 `PPL_INSTALL_PATH` 环境变量，适用于手动编译 PPL 源码的情况。

之后按照原本 SG2260/SG2260E 编译流程正常编译即可。

### 如何更新 `third_party/ppl`

source 环境变量后，使用 `untar_ppl` 更新，将自动筛选用到的文件。

```shell
untar_ppl ./ppl_v1.6.32-g1f31d7b4-20251030.tar.gz
```

### 如何兼容 SG2260

参考 [torch\_tpu/csrc/ops/my\_ops/RMSNorm.cpp](https://gerrit-ai.sophgo.vip:8443/#/c/152716/2/torch_tpu/csrc/ops/my_ops/RMSNorm.cpp)

+ 需要将 TPUStream 类型的 stream 变量传递给 PPL 算子 host 接口，它会自动转换成所需的 `tpuHandle_t` 或 `tpudnnHandle_t`；
+ SG2260 调用算子要少传递一个 `ppl_module` 参数，要将其放到宏判断中；
+ SG2260 要同时保留手写算子与 PPL 算子调用。


```C++
-    auto kernel = [&](tpuStream_t stream, tpuKernelModule_t ppl_module,
+    auto kernel = [&](TPUStream stream, tpuKernelModule_t ppl_module,

             uint32_t tile_size) -> int {
         if constexpr (std::is_same_v<scalar_t, float>) {
             return rmsnorm_fp32(
-                stream, ppl_module,
+                stream,
+#ifndef BACKEND_SG2260
+                ppl_module,
+#endif
                output_addr, input_addr, scale_addr, bias_addr,
                 eps, (scale_addr != 0), (bias_addr != 0),
                 static_cast<uint32_t>(outer_size),
                 static_cast<uint32_t>(inner_size),
                 tile_size);
...
...

-    tpuStream_t stream = c10_tpu::getCurrentTPUStream().stream();
+    auto stream = c10_tpu::getCurrentTPUStream();

     tpuKernelModule_t ppl_module = getPplModule();
     uint32_t tile_size = inner_size;

...


// 利用 usePPLKernels() 运行时判断是否使用 PPL 算子

#ifdef USING_PPL
+        if (usePPLKernels())
+        {

...
             AT_DISPATCH_FLOATING_TYPES_AND2(
                 at::kHalf, at::kBFloat16, input.scalar_type(), "rmsnorm_forward", [&] {
                     rmsnorm_forward_impl<scalar_t>(
                         reinterpret_cast<uint64_t>(output.data_ptr()),
                         reinterpret_cast<uint64_t>(input.data_ptr()),
                         scale ? reinterpret_cast<uint64_t>(scale->data_ptr()) : 0,
                         bias ? reinterpret_cast<uint64_t>(bias->data_ptr()) : 0,
                         outer_size, inner_size,
                         static_cast<float>(eps));
                 });
+        } else
+#endif
+        {
             auto stream = c10_tpu::getCurrentTPUStream();
             tpudnnStatus_t status = tpudnnRmsNormForwardAsync(
                 stream,
                 tpu::TPUGenerateTpudnnTensor(stream, input),
                 scale.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, scale.value()) : tpudnnUndefinedTensor(),
                 bias.has_value() ? tpu::TPUGenerateTpudnnTensor(stream, bias.value()) : tpudnnUndefinedTensor(),
                 tpu::TPUGenerateTpudnnTensor(stream, output),
                 axis,
                 eps);
             TORCH_CHECK(status == TPUDNN_STATUS_SUCCESS);
+        }
```

