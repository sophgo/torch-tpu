# 测试用例说明

# 实现规范

## pytest 规范

所有的测试前缀为 `test_`

实现过程需要参数化，避免出现参数写在函数内的方式：

```py
@pytest.mark.parametrize("input_shape", [(1, 1), (2, 2)])
def test_shape(input_shape):
    pass
```

> `input_shape` 会由 pytest 根据 parametrize 的参数自动注入

而不是

```py
def test_shape():
    input_shape = (1, 1)
    pass
```

多个 `@pytest.mark.parametrize` 可以叠加，会组成笛卡尔积：

```py
@pytest.mark.parametrize("input_shape, eps", [(1, 1), (2, 2)])
def test_shape(input_shape, eps):
    pass

@pytest.mark.parametrize("input_shape, eps", [(1, 1), (2, 2)])
@pytest.mark.parametrize("with_scale, with_bias", [(True, True), (False, True)])
def test_shape(input_shape, eps, with_scale, with_bias):
    pass
```

## 测试文件撰写规范

- 一个测试文件只测试一个算子，不同的 my_ops 算子需要不同的测试函数（xxx 和 xxx_fp8 算两个函数）
- 一个测试文件内包含：
  - 一个 ground truth 的 cpu 实现： `xxx_cpu`
  - 一个 tpu 实现： `xxx_tpu`
  - 1-n 个 `test_` 前缀函数
  - （可选）一个对比函数： `cmp_xxx`，用于实现多个 `test_` 但有相同的比对逻辑时复用。
- 使用 `assert` 而不是 `print`

如：

```py
assert torch.allclose(out_cpu, out_tpu)
```

如果有复杂的判断逻辑，可以用 `AssertionError` 抛出：

```py
if ...:
    ...
    ...
    raise AssertionError("out_cpu != out_tpu")
```

## 构建错误单测

对一些 bug，推荐使用 `pytest` 构建单测，在单测内说明情况，方便积累

```py
# reproduce topk precision bugs @2025.07.21
@pytest.mark.parametrize(
    "rand_batch,repeat_batch,n,groups,k",
    [
        (128, 1, 256, 32, 2),
    ],
)
def test_mid_topk(rand_batch, repeat_batch, n, groups, k, device="tpu"):
    ...
```

# 约定大于配置

> 约定大于配置:提供一些功能，但是不强制要求使用。
<!-- 
## LLM 标识

对 LLM 算子，在对应的测试中建议添加 `@pytest.mark.llm` 标记，方便后续统计。 -->

## 性能分析

性能分析默认关闭，在具体测试某个 case 时，可以手动开启：

```sh
pytest -k test_rmsnorm --enable_profile
```

在测试函数中，可以添加 `profiler` 参数，pytest 会自动注入 `profiler` 对象：

```py
def test_rmsnorm(profiler):
    with profiler.profile(buffer_size=1024, trace_level=2):
        pass
```

## 区分优先级

通过 `pytest.mark.priority_high` 标记高优先级，通过 `pytest.mark.priority_medium` 标记中等优先级，通过 `pytest.mark.priority_low` 标记低优先级：

```py
@pytest.mark.priority_high
def test_rmsnorm():
    pass
```

在回归中，pr 回归跑高优先级，保证快速通过回归：

```sh
pytest --priority high
```

daily 回归跑全量，每日发包前应该跑全量通过才能有对应的 daily 包：

```sh
pytest
```

- priority_high: 高优先级，会优先执行
- priority_medium: 中等优先级，会根据情况执行
- priority_low: 低优先级，不会执行

默认 `pytest` 全跑，可以指定某一类优先级 `pytest --priority high`

不做标记的测试函数默认都跑。实践中，一般将部分耗时不常用算子标记为低优先级，优先保证日常回归快速完成。
