# dequantize_k_cache_paged：CUDA C++ 实现

这是针对 **SGLang DSA** 的实现草案：按 token slot 索引 gather，将 FP8 E4M3 乘每 128 个元素一个 FP32 scale，输出 BF16，并复制 BF16 RoPE。附带固定版本的上游 Triton 源码、正确性验证和性能对比脚本。

**当前交付状态：完成源码和主机侧检查；环境没有 PyTorch、CUDA GPU 或 nvcc，未编译 CUDA、未执行 GPU 验证、没有实测加速比。请先运行 `validate.py`。**

## 对齐的上游

- 仓库：[sgl-project/sglang](https://github.com/sgl-project/sglang)
- Commit：`d4e3c163c95afe5f549116c9c73a6f0e6a4830a7`
- 文件：[python/sglang/kernels/ops/attention/dsa/dequant_k_cache.py](https://github.com/sgl-project/sglang/blob/d4e3c163c95afe5f549116c9c73a6f0e6a4830a7/python/sglang/kernels/ops/attention/dsa/dequant_k_cache.py)
- 函数：`dequantize_k_cache_paged` / `_dequantize_k_cache_paged_kernel`

`upstream_dsa.py` 保留该版本文件的原始内容，供直接比较。**DSv4 目录也有同名函数，但其 UE8M0/page 布局不适用于本实现。** 本实现不属于上游已合入的 CUDA 算子。

## 输入输出契约

```text
输入 dtype: torch.float8_e4m3fn（承载整行原始字节，包含 scale/RoPE 字节）
输入 shape: [slots, 1, packed_bytes] 或 [pages, page_size, 1, packed_bytes]
每行字节: [nope 个 FP8 | nope/128 个 FP32 scale | rope 个 BF16]
索引: [N] CUDA int32/int64，表示展平后的 token slot
输出: [N, 1, nope + rope] CUDA bfloat16
```

| NoPE / RoPE | FP8 字节 | Scale 字节 | RoPE 字节 | 输入每行 | 输出每行 |
|---|---:|---:|---:|---:|---:|
| 128 / 0 | 128 | 4 | 0 | 132 B | 256 B |
| 256 / 0 | 256 | 8 | 0 | 264 B | 512 B |
| 512 / 64 | 512 | 16 | 128 | 656 B | 1152 B |

- `nope` 为正的 128 倍数，`rope` 为 0 或 64，`group_size` 固定为 128。
- 反量化语义是 `BF16_RNE(FP32(fp8) * FP32(scale))`，scale 为乘数。
- RoPE 按原始 BF16 位复制，不乘 scale。
- 支持乱序、重复索引、输出 token 数大于 cache slot 数；这对 prefix sharing 很重要。
- 与上游一样，正常路径要求索引在 `[0, slots)`。`-1` 不是 sentinel。调试时传 `check_indices=True` 做同步边界检查。
- 支持可 `view(-1, packed_bytes)` 的跨行 stride；最后一维必须连续，每行地址至少 4 字节对齐。支持正 stride 的索引。
- `out` 必须是指定形状、连续、独立存储的 BF16 张量，至少 8 字节对齐；自动选择向量宽度时会考虑对齐。
- 推理用途，无 autograd、`torch.compile` 自定义算子注册。CUDA Graph 需提前编译和预热，并建议传入固定 `out`。

## 使用

建议在现有 SGLang CUDA 环境中运行。构建需要 CUDA-enabled PyTorch（带 `float8_e4m3fn`）、匹配的 CUDA Toolkit/nvcc、支持 C++17 的编译器以及 Ninja。可采用 PyTorch 2.5+ / CUDA 12.4+ 作为待验证的起点；本包未认证具体版本组合。CUDA 执行路径无需安装 SGLang 或 Triton；对比脚本需要 Triton。

```bash
cd dequant_k_cache_paged_cuda
python -m pip install ninja
# 可选：限制编译资源；设置与你的设备匹配的 TORCH_CUDA_ARCH_LIST
MAX_JOBS=4 python validate.py
```

首次调用通过 `torch.utils.cpp_extension.load` 编译，之后使用 PyTorch 的扩展缓存。

```python
from kv_dequant import build, dequantize_k_cache_paged

build()  # 可放在模型初始化阶段，避免首次请求承担编译延迟
output = dequantize_k_cache_paged(quant_k_cache, page_table_1_flattened)

# 重复调用时复用输出，避免分配混入测量：
dequantize_k_cache_paged(quant_k_cache, page_table_1_flattened, out=output)
```

替换 SGLang 的调用点时，使 `kv_dequant.py` 和 `dequant_cuda.cu` 位于同一可导入目录，再将原 DSA 函数导入替换为上述导入。保持 `group_size=128`，先核对实际部署分支的字节布局。这里没有修改或部署任何现有服务。

## 核心实现与性能判断

`dequant_cuda.cu` 中，一个 warp 负责一个输出 token，一个 CTA 处理 4 或 8 个 token。一个 kernel 完成 gather、NoPE 反量化和 RoPE 复制，使用 64 位行地址。线程每次读取 4/8/16 个 FP8，使用成对 FP8 转换和向量化 BF16 写回。

对 512+64 布局，上游固定版本每 token 启动 5 个 Triton program；本实现默认每 CTA 处理 4 个 token，CTA 数由 `5*N` 降至 `ceil(N/4)`。**CTA 数减少不等于耗时同比降低。** 小 N 的并行度、随机 gather、数据是否命中 L2、指令吞吐及寄存器占用都可能改变结果；这版 CUDA 也可能在部分形状上更慢。

FP8 先转 FP16 再转 FP32：所有有限 E4M3 数都能由 FP16 精确表示，因此该步骤没有额外舍入误差。乘法在 FP32 进行，最终用 BF16 round-to-nearest-even。未使用 `--use_fast_math`。NoPE NaN 的 payload 不保证与 Triton 相同；RoPE 原始位保留。极端 scale（例如 FP32 subnormal/溢出）仍需按实际应用数值域补测。

自动向量宽度只是起始配置：512+64 的连续行通常选 16，256+0 选 8，128+0 选 4。可用 `vec=4/8/16`、`warps=4/8` 比较。不能仅从 CUDA/Triton 语言选择推断加速倍数。

## 正确性验证

```bash
python check_host.py                       # 无 GPU 的格式/地址模型检查
python validate.py                         # CUDA vs PyTorch vs 上游 Triton
python validate.py --skip-triton           # 仅验证 CUDA vs PyTorch
compute-sanitizer --tool memcheck python validate.py --skip-triton
```

GPU 验证覆盖不同 NoPE/RoPE、非整 CTA 的 N、空输出、4D paged view、padding stride、int32/int64、乱序/重复索引、所有 FP8 编码、非 2 幂 scale、两种 block 大小、各向量宽度、非默认 stream 和 CUDA Graph replay。有限 NoPE 结果逐位比较；NoPE NaN 允许 payload 不同；RoPE 逐位比较。

`check_host.py` 只解析 Python，并验证 FP8→FP16 精确性及线程/地址模型，不能替代 CUDA 编译或 GPU 正确性验证。已执行结果记录在 `VALIDATION.txt`。

## CUDA vs Triton 性能对比

```bash
# 默认：预分配输出，在 CUDA Graph 中反复执行相同输入，用 CUDA events 计时。
python benchmark.py --mode graph --csv results_graph.csv

# 同时报告包含分配和 Python 开销的接口耗时（同步 wall clock）。
python benchmark.py --mode both --pattern sequential random shared --csv results.csv

# 仅 NoPE 的布局：
python benchmark.py --nope 256 --rope 0 --csv results_256.csv

# 固定维度/规模，比较另一个 CUDA launch 配置：
python benchmark.py --tokens 4096 32768 131072 --vec 8 --warps 8 --csv results_v8w8.csv

# 若要对比你的 SGLang 源码版本，指定其 DSA 文件路径；需维持同一 kernel 接口：
python benchmark.py --upstream-file /path/to/sglang/python/sglang/kernels/ops/attention/dsa/dequant_k_cache.py
```

输出 `CUDA_us`、`Triton_us` 和 `Triton/CUDA`。该比值大于 1 表示 CUDA 更快。CSV 配套 metadata JSON 记录 GPU、PyTorch/CUDA/Triton 版本、参数和上游文件 SHA256。

测量前先验证结果并预热；两条路径交替计时，报告多次测量中位数。Graph 路径两边都预分配输出，并直接调用同一个上游 Triton kernel；eager 路径两边都调用会分配输出的公开接口。

**默认测量是复用地址的稳定态，包含 L2 缓存效果，没有强制清空 L2。** 小 N 或 shared 索引可能完全命中缓存。`logical_GB/s` 仅按每 token 的逻辑输入/输出字节估算，并非实测 HBM 带宽。512+64、int32 索引时为每 token `656 + 1152 + 4 = 1812` B；物理访存、重复读取和缓存命中会改变实际流量。要分析生产收益，需要使用真实索引分布、上下游 kernel 顺序及 Nsight Compute 测量，不能把此比值直接当作端到端生成速度提升。

## 文件

| 文件 | 用途 |
|---|---|
| `dequant_cuda.cu` | CUDA kernel、dispatch、PyTorch C++ 绑定 |
| `kv_dequant.py` | JIT 构建与同名 Python 接口 |
| `reference.py` | 输入打包、PyTorch 参考、上游预分配输出启动器 |
| `validate.py` | GPU 正确性验证 |
| `benchmark.py` | CUDA/Triton 对比与 CSV 输出 |
| `upstream_dsa.py` | 固定 commit 的原始 SGLang 参考源码 |
| `check_host.py` / `VALIDATION.txt` | 主机侧检查及已执行结果 |
| `PROVENANCE.json` / `NOTICE` / `LICENSE` | 来源、状态与许可 |
