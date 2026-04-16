# MXFP8 Mock 验证方案

> 目标：通过 mock 算子跑通 DeepSeek3 MXFP8 训练流程，验证 msadapter + MindSpore 路径的可行性。

---

## 一、背景

DeepSeek3 MXFP8 训练依赖 9 个 `torch_npu` 专有算子 + 1 个自定义 dtype。msadapter 当前均未适配。在 MindSpore 补齐真实算子之前，用 mock 实现填充接口，使端到端流程跑通。

### 算子缺失全景

| 分类 | 算子 | aclnn 内核 | MindSpore 现状 |
|------|------|-----------|---------------|
| 量化 | `npu_dynamic_mx_quant` | `aclnnDynamicMxQuant` | 缺失 |
| 量化 | `npu_dynamic_mx_quant_with_dual_axis` | `aclnnDynamicMxQuantWithDualAxis` | 缺失 |
| 量化 | `npu_grouped_dynamic_mx_quant` | `aclnnGroupedDynamicMxQuant` | 缺失 |
| 矩阵乘 | `npu_quant_matmul` | `aclnnQuantMatmulV5` | **已有** `gen.quant_matmul` |
| 矩阵乘 | `npu_add_quant_matmul_` | `aclnnQuantBatchMatmulInplaceAdd` | 缺失 |
| 矩阵乘 | `npu_grouped_matmul` (MXFP8) | `aclnnGroupedMatmulV5` | 已有 V4，缺 MXFP8 参数 |
| 矩阵乘 | `npu_add_quant_gmm_` | `aclnnQuantGroupedMatmulInplaceAdd` | 缺失 |
| 通算融合 | `npu_all_gather_quant_mm` | `aclnnAllGatherMatmulV2` | 已有 V1，缺量化参数 |
| 通算融合 | `npu_quant_mm_reduce_scatter` | `aclnnMatmulReduceScatterV2` | 已有 V1，缺量化参数 |
| dtype | `float8_e8m0fnu` | `ACL_FLOAT8_E8M0` | 缺失 |

---

## 二、Mock 策略

### 核心原则

- **保持接口签名完全一致**：MindSpeed 代码零修改
- **保持 tensor shape/dtype 正确**：下游数据结构（`Float8Tensor2D`、`QuantTensorMeta`）能正确构造
- **数值近似即可**：用 BF16 matmul 替代 FP8 量化 matmul，流程验证不追求数值精度
- **每个 mock 首次调用打 warning 日志**：便于确认覆盖范围

### Mock 行为表

| 算子 | Mock 行为 | 输出 |
|------|----------|------|
| `npu_dynamic_mx_quant` | `_safe_cast(tensor, fp8_dtype)` + 全 1 scale | `(data[fp8], scale[e8m0])` |
| `npu_dynamic_mx_quant_with_dual_axis` | 两次单轴 mock | `(col_data, col_scale, row_data, row_scale)` |
| `npu_grouped_dynamic_mx_quant` | 复用单轴 mock | `(data[fp8], scale[e8m0])` |
| `npu_quant_matmul` | 忽略 scale，BF16 `ops.matmul` | `output[bf16]` |
| `npu_add_quant_matmul_` | `main_grad += matmul(x1, x2)` 原地写 | `main_grad` |
| `npu_grouped_matmul` [MXFP8] | cast BF16 后调 `grouped_matmul_v4`（忽略 scale_dtype） | `tuple[tensor]` |
| `npu_add_quant_gmm_` | `main_grad += matmul(x1, x2)` 原地写 | `main_grad` |
| `npu_all_gather_quant_mm` | cast BF16 后调 `ops.all_gather_matmul` | `(output, gather_out, None)` |
| `npu_quant_mm_reduce_scatter` | cast BF16 后调 `ops.matmul_reduce_scatter` | `(output, None)` |
| `float8_e8m0fnu` | `ms.int8` 占位 | — |

### `_safe_cast` 防护

`tensor.to(float8_e4m3fn)` 在 MindSpore 上可能不支持 cast。`_safe_cast` 捕获异常后 fallback 到原 dtype：

```python
def _safe_cast(tensor, dst_type):
    try:
        return tensor.to(dst_type)
    except (TypeError, RuntimeError):
        return tensor  # keep original dtype
```

---

## 三、文件改动

### 3.1 新增文件

**`msadapter/msa_thirdparty/torch_npu/mxfp8_mock.py`**

9 个 mock 函数 + `float8_e8m0fnu` 常量 + `_safe_cast` 工具函数。

```python
# dtype 占位
float8_e8m0fnu = getattr(ms, 'float8_e8m0fnu', ms.int8)

# 量化 mock 示例
def npu_dynamic_mx_quant(tensor, axis=-1, round_mode="rint",
                         dst_type=ms.float8_e4m3fn, block_size=32,
                         scale_alg=None):
    data = _safe_cast(tensor, dst_type)
    scale_shape = list(tensor.shape)
    dim = axis if axis >= 0 else len(scale_shape) + axis
    scale_shape[dim] = math.ceil(scale_shape[dim] / block_size)
    scale = ops.ones(scale_shape, dtype=float8_e8m0fnu)
    return data, scale

# matmul mock 示例
def npu_quant_matmul(x1, x2, scale, offset=None, pertoken_scale=None,
                     bias=None, output_dtype=None, ...):
    out_dtype = output_dtype if output_dtype is not None else ms.bfloat16
    return ops.matmul(x1.to(out_dtype), x2.to(out_dtype))
```

### 3.2 修改文件

**`msadapter/msa_thirdparty/torch_npu/__init__.py`**

变更点：

1. 新增 `hifloat8` / `HiFloat8Tensor` 占位（防止 MindSpeed `constants.py` import 报错）
2. 新增 mock import 块
3. `npu_grouped_matmul` 签名扩展 `scale_dtype`/`per_token_scale_dtype` 参数

```python
# HiFloat8 占位
hifloat8 = getattr(ms, 'hifloat8', None)
class HiFloat8Tensor:
    @staticmethod
    def to_hifloat8(tensor):
        return tensor

# MXFP8 mock import
from .mxfp8_mock import (
    float8_e8m0fnu,
    npu_dynamic_mx_quant,
    npu_dynamic_mx_quant_with_dual_axis,
    npu_grouped_dynamic_mx_quant,
    npu_quant_matmul,
    npu_add_quant_matmul_,
    npu_grouped_matmul_mxfp8 as _grouped_matmul_mxfp8,
    npu_add_quant_gmm_,
    npu_all_gather_quant_mm,
    npu_quant_mm_reduce_scatter,
)

# npu_grouped_matmul 扩展签名
def npu_grouped_matmul(x, weight, ..., scale_dtype=None, per_token_scale_dtype=None):
    return _grouped_matmul_mxfp8(x, weight, ..., scale_dtype=scale_dtype,
                                  per_token_scale_dtype=per_token_scale_dtype)
```

---

## 四、端到端流程验证

### 4.1 调用链覆盖

```
pretrain_gpt.py
  ↓
MindSpeed patch 注册（纯 Python setattr）                    ← 不涉及 mock
  ↓
fp8_autocast → FP8GlobalStateManager                        ← 不涉及 mock
  ↓
TEColumnParallelLinear.forward
  → MXFP8MatMul.apply
    → torch_npu.npu_dynamic_mx_quant_with_dual_axis         ← mock ✓
    → torch_npu.npu_quant_matmul                             ← mock ✓
  ↓
MXFP8MatMul.backward
  → torch_npu.npu_dynamic_mx_quant_with_dual_axis           ← mock ✓
  → torch_npu.npu_quant_matmul (dx)                          ← mock ✓
  → torch_npu.npu_quant_matmul (dw)                          ← mock ✓
  → torch_npu.npu_add_quant_matmul_ (梯度融合)                ← mock ✓
  ↓
Float8Tensor2D.release
  → data.untyped_storage().resize_(0)                        ← MindSpore 原生支持 ✓
  ↓
MoE MXFP8GMMFunction
  → torch_npu.npu_dynamic_mx_quant                           ← mock ✓
  → torch_npu.npu_grouped_matmul (scale_dtype=e8m0fnu)       ← mock ✓
  → torch_npu.npu_grouped_dynamic_mx_quant                   ← mock ✓
  → torch_npu.npu_add_quant_gmm_                             ← mock ✓
  ↓
MC2 通算融合（多卡）
  → torch_npu.npu_all_gather_quant_mm                        ← mock ✓
  → torch_npu.npu_quant_mm_reduce_scatter                    ← mock ✓
```

### 4.2 已确认可行的机制

| 机制 | 确认来源 |
|------|---------|
| `torch.autograd.Function` ctx 任意属性 | MindSpore C++ `FunctionBase` tp_setattro=nullptr，子类有 `__dict__` |
| `_Function.save_for_backward` 无类型检查 | `_grad_function.py:49` 直接赋值 |
| `untyped_storage().resize_(0)` | `storage_py_reg.cc:322` 注册了 `resize_` |
| MindSpeed monkey-patch 与 msadapter 不冲突 | patch 操作 Megatron 模块路径，非 `torch.*` |
| `create_dummy` 空壳模块 | 操作 `transformer_engine.*` 路径，不被 msadapter 拦截 |

### 4.3 需运行时验证的点

| 点 | 风险 | 应对 |
|----|------|------|
| `tensor.to(float8_e4m3fn)` | MindSpore 可能不支持该 cast | `_safe_cast` fallback 到原 dtype |
| `ops.matmul` 接收 FP8 tensor | FP8 输入可能不支持 matmul | mock 中先 `.to(bf16)` 再 matmul |
| `.t()` / `.T` 是 view 还是 copy | MindSpore 可能返回 copy | 功能正确但显存翻倍 |
| HCCL 通信域初始化 | 多卡场景需 msrun 正确启动 | 先单卡验证，再多卡 |

---

## 五、验证步骤

### Step 1：单卡 smoke test

```bash
# 设置环境
export PYTHONPATH=.../msadapter:.../msadapter/msa_thirdparty:$PYTHONPATH

# 单卡、小 seq_len、少量 step
python pretrain_gpt.py \
    --fp8-format e4m3 \
    --fp8-recipe mxfp8 \
    --transformer-impl transformer_engine \
    --micro-batch-size 1 \
    --seq-length 128 \
    --train-iters 2 \
    ...
```

预期输出：
- `[MXFP8 MOCK]` 日志出现 9 次（每个算子首次调用）
- forward/backward 完成无报错
- loss 值有输出（数值不要求准确）

### Step 2：检查 mock 覆盖

```python
# 训练结束后检查
from torch_npu.mxfp8_mock import _warned
print("Mocked operators hit:", _warned)
# 预期: {'npu_dynamic_mx_quant', 'npu_dynamic_mx_quant_with_dual_axis',
#         'npu_quant_matmul', 'npu_grouped_matmul[MXFP8]', ...}
```

### Step 3：多卡验证（8 卡）

`torchrun` 和 `msrun` 均可使用：

```bash
# 方式一：torchrun（原始脚本不用改）
torchrun --nproc_per_node=8 \
    pretrain_gpt.py \
    --fp8-recipe mxfp8 \
    --tensor-model-parallel-size 4 \
    --expert-model-parallel-size 8 \
    --train-iters 2 \
    ...

# 方式二：msrun
msrun --worker_num 8 --local_worker_num 8 \
    pretrain_gpt.py \
    ...
```

重点关注通算融合 mock 是否正常（`npu_all_gather_quant_mm` / `npu_quant_mm_reduce_scatter`）。

### Step 4：逐个替换为真实算子

替换顺序建议（按依赖关系 + 验证难度）：

| 优先级 | 算子 | 原因 |
|--------|------|------|
| P0 | `float8_e8m0fnu` dtype | 所有算子的前置依赖 |
| P1 | `npu_quant_matmul` | MindSpore 已有 `gen.quant_matmul`，直接映射 |
| P1 | `npu_dynamic_mx_quant` | 最基础的量化算子 |
| P1 | `npu_dynamic_mx_quant_with_dual_axis` | Dense Linear 核心路径 |
| P2 | `npu_grouped_matmul` 扩展 | MoE 核心路径 |
| P2 | `npu_grouped_dynamic_mx_quant` | MoE backward dw |
| P3 | `npu_add_quant_matmul_` | 梯度融合优化 |
| P3 | `npu_add_quant_gmm_` | MoE 梯度融合优化 |
| P4 | `npu_all_gather_quant_mm` | MC2 通算融合 |
| P4 | `npu_quant_mm_reduce_scatter` | MC2 通算融合 |

替换方式：修改 `torch_npu/__init__.py` 的 import 来源：

```python
# mock → real 示例
# from .mxfp8_mock import npu_quant_matmul      # mock
from .mxfp8_real import npu_quant_matmul          # real (调 gen.quant_matmul)
```

---

## 六、分布式启动：torchrun 兼容性

### 不需要手动改启动脚本

msadapter 的 `init_process_group` 内置了 torchrun 自动适配机制：

```
torchrun 启动
  → 设置 RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
  → 脚本调用 torch.distributed.init_process_group("nccl")
    → msadapter.distributed.init_process_group
      → "nccl" 自动转换为 "hccl"
      → 检测 MS_ROLE 环境变量不存在（非 msrun 启动）
      → rank=0 自动启动 MindSpore scheduler 子进程
      → 自动设置 MS_WORKER_NUM, MS_ROLE, MS_NODE_ID 等环境变量
      → 调用 mindspore.communication.init(backend_name="hccl")
```

关键代码位于 `msadapter/distributed/distributed_c10d.py:1560-1592`：

```python
if not os.getenv("MS_ROLE"):
    # Not call from msrun, should call a subprocess for scheduler.
    if rank == 0:
        sched_p = subprocess.Popen([sys.executable, str(script),
            "--rank_id", str(rank), "--rank_size", str(world_size),
            "--distributed_init_method", str(init_method)])
    os.environ["MS_WORKER_NUM"] = str(world_size)
    os.environ["MS_ROLE"] = "MS_WORKER"
    os.environ["MS_NODE_ID"] = str(rank)
    os.environ["MS_SCHED_HOST"] = str(master_addr)
    os.environ["MS_SCHED_PORT"] = str(master_port)
init(backend_name=backend)
```

### 启动方式对照

| 启动方式 | 是否支持 | 原理 |
|---------|---------|------|
| `torchrun --nproc_per_node=8` | 支持 | msadapter 自动补全 MindSpore 环境变量 + 启动 scheduler |
| `msrun --worker_num 8` | 支持 | 原生 MindSpore 启动，`MS_ROLE` 已设置 |
| 脚本中 `init_process_group("nccl")` | 支持 | `nccl` → `hccl` 自动转换 |

### DeepSeek3 脚本无需修改

原始脚本 `8k_fp8_sbh_8p.sh` 中的 torchrun 启动命令可直接使用，msadapter 在 `init_process_group` 内部透明处理所有适配。

---

## 七、机制可行性结论

| 维度 | 结论 |
|------|------|
| msadapter import 代理 | `torch` → `msadapter`、`torch_npu` → `msa_thirdparty/torch_npu`，路径畅通 |
| MindSpeed monkey-patch | 操作 Megatron/TE 模块，与 msadapter 代理互不干扰 |
| `torch.autograd.Function` | MindSpore `_Function` 完整支持，ctx 可存任意属性 |
| `untyped_storage().resize_(0)` | MindSpore C++ 层已注册，显存释放机制可用 |
| MoE GroupedMatMul | 现有 `grouped_matmul_v4` 可接受 BF16 输入做 mock |
| 通算融合 | 非量化版 `all_gather_matmul` / `matmul_reduce_scatter` 可做 mock 降级 |

**结论：基于 mock 跑通 DeepSeek3 MXFP8 端到端流程可行。**
