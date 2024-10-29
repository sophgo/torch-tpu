gDNN 算子迁移至 tpuDNN
=======================

### 背景

要解决的问题：

+ Torch-TPU 的实现对于 BM1684X 和 SG2260 存在两套完全独立的管理、调用代码，造成：a) 维护困难，要分别编译调试、release；b）难以扩展支持新的芯片；

+ Torch-TPU firmware 目录中的算子 kernel 实现，大量依赖 TPU1686 代码仓库的实现函数，但是全部都 hard-code 函数签名，binary 放在 third-party 目录。这导致这部分实现事实上无法更新、维护。

```
 ┌─────────────────────────────────────────────┐             ┌───────────────────────────────┐
 │                                             │             │                               │
 │                                             │             │                               │
 │              ┌───────────────────────────┐  │             │  ┌─────────────────────────┐  │
 │              │       Torch-Ops           │  │             │  │       Torch-TPU         │  │
 │              └────┬─────────────────┬────┘  │             │  └─────────┬───────────────┘  │
 │                   │                 │       │             │            │                  │
 │           ┌───────┼─────────────────┼───────┘             │            │                  │
 │           │       │                 │                     │            │                  │
 │           │       v                 v                     │            v                  │
 │ Torch-TPU │  ┌────────────┐  ┌───────────┐                │  ┌─────────────────────────┐  │
 │           │  │   BMLib    │  │  TPUV7Rt  │                │  │       tpuDNN            │  │
 │           │  └────┬───────┘  └──────┬────┘     ────────>  │  └───┬────────────────┬────┘  │
 │           │       │                 │                     │      │                │       │
 │           └───────┼─────────────────┼───────┐             │      │                │       │
 │                   │                 │       │             │      │                │       │
 │                   │                 │       │             │      │                │       │
 │                   v                 v       │             │      v                v       │
 │              ┌───────────────────────────┐  │             │  ┌─────────┐   ┌───────────┐  │
 │              │          firmware         │  │             │  │  BMLib  │   │  TPUV7Rt  │  │
 │              └────┬─────────────────┬────┘  │             │  └───┬─────┘   └──────┬────┘  │
 │                   │                 │       │             │      │                │       │
 └───────────────────┼─────────────────┼───────┘             │      v                v       │
                     │                 │                     │  ┌─────────────────────────┐  │
                     │                 │                     │  │       TPU1686           │  │
                     v                 v                     │  └─────────────────────────┘  │
                ┌───────────────────────────┐                │                               │
                │          TPU1686          │                │                               │
                └───────────────────────────┘                └───────────────────────────────┘
```

### 目标

+ Torch-TPU 的实现与芯片架构、runtime 类型脱钩，只依赖 tpuDNN；

+ 新适配的芯片与算子不再依赖 Torch-TPU 的 kernel 实现，已有算子后续逐渐迁移到 TPU1686 仓库。

### 迁移步骤

首先根据最新 README.md 配置好开发环境，编译 Debug 版本的 TPU1686 和 Torch-TPU。

与算子迁移相关的代码主要有：

+ 单元测试 Python 代码；

+ PyTorch op 适配实现代码；

+ `tpu-train` 仓库下 Sgdnn 实现；

+ tpuDNN 代码。

#### 定位单元测试

在 `tpu-train/python/utest_ops` 目录找到 op 对应的单元测试，如 `python/utest_ops/matmul.py`。在单元测试中确认 SG2260 测试已开启：

```python
    metric_table = ['max_diff','MAE']
    chip_arch_dict = {"bm1684x":1, 'sg2260':0} # 这里 sg2260 字段要改成 1
    epsilon_dict = {'bm1684x':{'f32':1e-6,'f16':1e-2},'sg2260':{'f32': 1e-6, 'f16':1e-2}}
```

在 `tpu-train/python/utest_ops` 目录运行单元测试，在日志中关注：

```python
{'max_diff': 0.00463104248046875, 'MAE': 0.001481811166740954}
```

这样的比对结果，这两个数值代表误差大小，可以看到原始的 TPU 算子是否能与 CPU 计算对齐。

#### 查看原始对接实现

在 `tpu-train/torch_tpu/csrc/ops/native_ops/` 目录中找到对应的 Op 实现文件，如 `tpu-train/torch_tpu/csrc/ops/native_ops/Matmul.cpp`。

在 Op 实现文件中搜索 `TORCH_LIBRARY_IMPL` 可以找出所有的 Op 注册操作，如

```
TORCH_LIBRARY_IMPL ( aten, TPU, m )
{
    m.impl ( "bmm.out", bmm_out_tpu );
}
```

上面的代码注册 `bmm_out_tpu` 函数作为 PyTorch 预定义的 `bmm.out` 实现，`.out` 后缀代表该计算的输出已经预先分配好，不是 `inplace` 操作。

在 Op 实现函数中可以看到 sgdnn 函数调用：

```
auto status = sgdnnMatmul (
              tpu::TPUGetDeviceResource(),
              tpu::TPUGenerateSgdnnTensor ( mat1_ ),
              tpu::TPUGenerateSgdnnTensor ( mat2_ ),
              tpu::TPUGenerateSgdnnTensor ( self ),
              tpu::TPUGenerateSgdnnTensor ( out ) );
TORCH_CHECK ( status == SG_SUCCESS );
```

这部分就是我们要替换成 tpuDNN 实现的代码。

根据名字定位后，还需要通过 gdb 确认。用 gdb 运行单元测试，在实现文件中打断点，看 Op 实现代码是否被调用。

#### tpuDNN 接口定义与实现

在 `sgdnn/src/sgdnn.cpp` 文件中找到 sgdnn 接口实现，将其迁移到 tpuDNN 中。如：

```cpp
tpu_status_t sgdnnMatmul ( tpu_resource_t resource ,
                          SgdnnTensor_t left,
                          SgdnnTensor_t right,
                          SgdnnTensor_t bias,
                          SgdnnTensor_t output ,
                          bool non_blocking )
```

将函数签名拷贝到 `TPU1686/tpuDNN/include/tpuDNNTensor.h` 中，并做相应修改：

+ `tpu_status_t` 返回值类型改为 `tpudnnStatus_t`；
+ `SgdnnTensor_t` 改为 `tpudnnTensor_t`；
+ `tpu_resource_t` 改为 `tpudnnHandle_t`；
+ `sgdnn` 函数前缀改为 `tpudnn`，在函数结尾加上 `Async` 后缀；
+ `non-blocking` 参数去掉。

修改后的函数签名如下：

```cpp
tpudnnStatus_t tpudnnMatmulAsync(
    tpudnnHandle_t handle,
    tpudnnTensor_t left,
    tpudnnTensor_t right,
    tpudnnTensor_t bias,
    tpudnnTensor_t output);
```

然后在 `TPU1686/tpuDNN/src/tensor/` 目录创建 Op 同名源文件，参考 `TPU1686/tpuDNN/src/tensor/matmul.cpp` 包含基本头文件，将上述函数签名拷贝进去；将 `sgdnn` 的函数实现内容拷贝进去作为新的 `tpuDNN` 接口实现。

`tpuDNN` 实现有两大原则：

+ 尽量跳过 `tpu-train/firmware` 目录中定义的算子接口和参数结构体；

+ BM1684X 和 SG2260 尽量使用同一套调用代码，把原本 `sgdnn` 实现中的 `#ifdef BACKEND_1684X/BACKEND_SG2260` 宏判断全部去掉。

以下是已知需要做的修改，可以用于参考检查：

+ 加上 `auto pimpl = static_cast<TPUDNNImpl *>(handle);`，获得 tpuDNN 实现类指针；
+ 格式变化，两空格缩进变 4 空格缩进；
+ 所有 Sgdnn 开头的类型、帮助函数全部替换成 tpuDNN 的版本；
+ `BACKEND_1684X` 的宏定义替换成 `__bm1684x__`；
+ 注意 tpuDNN 传入的地址全部为虚拟地址，解析 `tensor.addr` 的全部要加上 `pimpl->virtToPhys` 调用进行转换；
+ 用到 runtime 接口申请内存的，全部替换成 tpuDNN 内存分配；`tpu_device_mem_t` 地址类型全部换成 `void *` 或转换后的 `uint64_t` 物理地址；
+ `sgdnnTPUKernelLaunch` 函数和 `sgdnnTPUKernelLaunchMultiCore` 函数统一替换成 `pimpl->launchKernel` 调用；
+ `tpuDNN` 传入的数据类型全部用 `tpudnnDataType_t` 表示，需要使用到 `sg_data_type_t` 数据类型的，`tpudnnDataType_t` 可以直接强制转换成 `sg_data_type_t`。
+ `#ifdef USING_PERF_MODE` 宏去掉，`printf` 和 `std::cout` 打印全部换成 LOG 调用，后续通过 Log level 控制打印输出；
+ 源文件加入 `TPU1686/tpuDNN/src/tensor/CMakeLists.txt` 中的编译列表。

修改完成后，执行 `rebuild_TPU1686` 编译；也可以在 `TPU1686/build` 目录执行 `make` 编译，这样不会每次从头编译，更方便定位编译问题。

#### Torch-TPU 调用修改

在 Op 实现文件中替换 `sgdnn` 实现成新的 `tpuDNN` 调用，如前述 `tpu-train/torch_tpu/csrc/ops/native_ops/Matmul.cpp` 中替换为：

```cpp
  auto stream = c10_tpu::getCurrentTPUStream();
  auto status = tpudnnMatmulAsync(
    stream,
    tpu::TPUGenerateTpudnnTensor(stream, self_),
    tpu::TPUGenerateTpudnnTensor(stream, mat2_),
    tpudnnUndefinedTensor(),
    tpu::TPUGenerateTpudnnTensor(stream, out));
  TORCH_CHECK ( status == TPUDNN_STATUS_SUCCESS );
```

然后调用 `python setup.py develop` 编译后，执行前述单元测试验证正确性。

#### 测试 1684X

SG2260 的实现通过后，还需要进行 BM1684X 的实现验证。需要退出当前终端，重新 `source scripts/envsetup.sh bm1684x local`。

+ `rm -rvf build` 删除 SG2260 编译；
+ `rebuild_TPU1686` 编译 BM1684X 版本的 `tpuDNN`；
+ 调用 `python setup.py develop` 编译后，执行前述单元测试验证在 BM1684X 的正确性。

#### 代码提交

功能正常后，首先提交 TPU1686 的代码，通过回归并合入。*合入之后*编译更新 `tpu-train` repo 下两个芯片对应的 `tpuDNN.so`，和 `torch-tpu` 代码修改一起提交回归。

```bash
cp -rv TPU1686/tpuDNN/include/* tpu-train/third_party/tpuDNN/include/

# Build tpuDNN for bm1684x
cp build/tpuDNN/src/libtpudnn.so ../tpu-train/third_party/tpuDNN/bm1684x_lib/libtpudnn.so

# Build tpuDNN for sg2260
cp build/tpuDNN/src/libtpudnn.so ../tpu-train/third_party/tpuDNN/sg2260_lib/libtpudnn.so
```

### 参考

+ 必读：[Facilitating New Backend Integration by PrivateUse1](https://pytorch.org/tutorials/advanced/privateuseone.html)；
+ [PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/)；
+ [提交示例](https://gerrit-ai.sophgo.vip:8443/#/c/116475/)，及其对应[TPU1686 提交](https://gerrit-ai.sophgo.vip:8443/#/c/116476/)。
