# dequantize_k_cache_paged：CUDA C++ 实现

这是针对 **SGLang DSA** 的实现草案：按 token slot 索引 gather，将 FP8 E4M3 乘每 128 个元素一个 FP32 scale，输出 BF16，并复制 BF16 RoPE。附带固定版本的上游 Triton 源码、正确性验证和性能对比脚本。

**当前交付状态：已在一块 RTX 3070 Ti Laptop GPU (sm86) 上编译、验证并实测。CUDA 与 PyTorch 参考逐位一致；与 Triton 的对比见下文（上游 Triton kernel 在 sm86 无法编译，对比使用等价的 sm86 适配版）。**

## 对齐的上游

- 仓库：[sgl-project/sglang](https://github.com/sgl-project/sglang)
- Commit：`d4e3c163c95afe5f549116c9c73a6f0e6a4830a7`
- 文件：[python/sglang/kernels/ops/attention/dsa/dequant_k_cache.py](https://github.com/sgl-project/sglang/blob/d4e3c163c95afe5f549116c9c73a6f0e6a4830a7/python/sglang/kernels/ops/attention/dsa/dequant_k_cache.py)
- 函数：`dequantize_k_cache_paged` / `_dequantize_k_cache_paged_kernel`

`upstream_dsa.py` 保留该版本文件的原始内容，供直接比较。**DSv4 目录也有同名函数，但其 UE8M0/page 布局不适用于本实现。** 本实现不属于上游已合入的 CUDA 算子。

**sm86 (Ampere) 注意**：Triton 3.6 在 compute capability < 89 上拒绝编译 `fp8e4nv`（E4M3）kernel，固定版上游 kernel 在此类 GPU 上无法运行。`upstream_sm86.py` 是它的 sm86 适配版：仅把 FP8 load 换成 uint8 load 加一个精确的软件 E4M3→FP32 位解码，grid/mask/scale/RoPE/舍入语义完全一致；对比双方在此 GPU 上都走软件解码。sm89+ 上应继续用 `upstream_dsa.py`。

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

`dequant_cuda.cu` 中，一个 warp 负责一个输出 token，一个 CTA 处理 4 或 8 个 token（`warps=0` 自动选 8）。一个 kernel 完成 gather、NoPE 反量化和 RoPE 复制，使用 64 位行地址。线程每次读取 4/8/16 个 FP8，成对转换后向量化 BF16 写回。

对 512+64 布局，上游固定版本每 token 启动 5 个 Triton program；本实现每 CTA 处理 8 个 token，CTA 数由 `5*N` 降至 `ceil(N/8)`。**CTA 数减少不等于耗时同比降低**；小 N 的并行度、随机 gather、L2 命中、指令吞吐及寄存器占用都可能改变结果。

### sm86 实测后的优化（2026-10-19）

原版直接调用 `cuda_fp8.h` 的 `__nv_cvt_fp8x2_to_halfraw2`。在 sm86 上它是软件模拟，且含**变长的 subnormal 规范化 while 循环**——小 N 时 kernel 是指令/延迟受限的，这一路径成了主要瓶颈。优化分两轮：

**第一轮（解码与访存策略）**
- **直接 E4M3→FP32 位构造**：normal 码 `sign | ((exp4+120)<<23) | (mant<<20)` 一步拼出 FP32 位型；subnormal 用 `(1+m/8)·2⁻⁶ − 2⁻⁶` 的常数修正；`0x7F/0xFF` 映射为 canonical NaN。无循环、无 FP16 中转，每个值的解码指令数大约减半。**这是最大单项收益：N=2048 时 13.4µs → 6.7µs（约 2 倍）**。
- **`__float22bfloat162_rn` 成对打包** BF16 输出，一条指令替代两次转换加拼包。
- **`__ldcs` 流式读**：gather 行单次消费，evict-first 不污染 L2。
- `warps=0` 自动选 8（所有实测尺寸上 8-token CTA 最快或持平）。

**第二轮（形状特化与写策略自适应）**
- **形状特化 kernel**：nope/rope 编译期常量化（覆盖 128/256/384/512/768/1024 × rope 0/64），列循环、scale 组号（`lane>>3`）、RoPE 偏移全部成为常量，gather 首次访存的地址链显著缩短。N=2048 再 -9~14%。非标准形状/显式 vec/warps 自动回落到通用 kernel。
- **按 L2 容量自适应写策略**：输出总字节数小于 L2（本机 4MiB）时用 `__stwt`（write-through，免去脏行回写管理），超过则用 `__stcs`（流式 write-back）。交叉点恰为 L2 容量；两侧各赢 1~8%。L2 大小通过 `cudaDeviceGetAttribute` 查询并按设备缓存。

**实验过并否决的方向**（都有数据支撑）：SIMD 快路径解码（4 字节整体测试 + PRMT 展开 + bf16 位型对一次构造，指令更少但随机数据下近半数字节含 subnormal，warp 内分支混合执行反而慢 40%）；512 线程 CTA（小 N 慢 5~15%）；16B 向量化 RoPE 拷贝（正确实现后各尺寸持平或更差；一次早期测量显示的收益来自只拷一半数据的 bug）。

反量化语义保持 `BF16_RNE(FP32(fp8) · FP32(scale))`：`e4m3_to_f32` 对全部有限值精确，乘法 FP32 RNE，最终一次 BF16 RNE。未使用 `--use_fast_math`。NoPE NaN 的 payload 不保证与 Triton 相同；RoPE 原始位保留。

自动向量宽度：512+64 的连续行选 16，256+0 选 8，128+0 选 4。可用 `vec=4/8/16`、`warps=4/8` 复算（显式参数会绕过形状特化路径）。

## 正确性验证

```bash
python check_host.py                       # 无 GPU 的格式/地址模型检查
python validate.py --upstream-file upstream_sm86.py   # sm86: CUDA vs PyTorch vs Triton 适配版
python validate.py --skip-triton           # 仅验证 CUDA vs PyTorch
compute-sanitizer --tool memcheck python validate.py --skip-triton   # 需要 compute-sanitizer
```

sm89+ 上可用 `python validate.py` 直接对比固定版上游 Triton。

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

## 目标场景实测（topk=2048 / pool 200k / 输入 1–8k，sm86）

`bench_dsa.py` 固定该场景：每次调用 gather `topk=2048` 个随机 slot（稀疏注意力选中的 token）加最近 `L` 个连续 slot（当前输入块），`pool=200000`，512+64 布局，int32 索引。graph 模式为 kernel 纯耗时（CUDA Graph 内 50 次连发取中位数）；eager 模式含 Python 封装与输出分配。

```bash
python bench_dsa.py --mode both --csv results_dsa_optimized.csv
```

RTX 3070 Ti Laptop (sm86)，torch 2.11.0+cu130，Triton 对比为 `upstream_sm86.py`（见上文说明）：

| L(输入) | N=gather 数 | CUDA µs (graph) | Triton µs (graph) | Triton/CUDA | CUDA µs (eager) | Triton µs (eager) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2048 | 5.8 | 16.6 | **2.89×** | 6.8 | 21.3 |
| 1 | 2049 | 5.8 | 15.6 | 2.70× | 7.4 | 23.3 |
| 128 | 2176 | 6.7 | 16.4 | 2.44× | 7.3 | 22.0 |
| 512 | 2560 | 11.6 | 20.1 | 1.74× | 11.1 | 26.2 |
| 1024 | 3072 | 14.4 | 24.6 | 1.71× | 15.4 | 24.7 |
| 2048 | 4096 | 18.5 | 30.5 | 1.64× | 19.8 | 33.0 |
| 4096 | 6144 | 28.0 | 50.5 | 1.80× | 29.2 | 52.3 |
| 8192 | 10240 | 46.8 | 94.4 | **2.02×** | 48.1 | 96.0 |

优化历程（graph 模式，N=2048/2176/3072/4096/6144/10240）：最初版本 13.4/12.2/16.9/21.3/30.0/47.9 µs → 第一轮（解码重写+流式提示）→ 7.3/8.0/13.8/18.7/28.1/46.9 µs → 第二轮（形状特化+写策略自适应）→ 5.8/6.7/14.4/18.5/28.0/46.8 µs。合计小 N 提升约 **2.3 倍**；N≥4096 达到 ~397 GB/s 逻辑带宽（448 GB/s 峰值的 89%），已接近 HBM 带宽墙。eager 模式下 Triton 额外承担其 Python 启动开销，比值进一步扩大到 ~1.6–3.2×。

测量前先验证结果并预热；两条路径交替计时，报告多次测量中位数。Graph 路径两边都预分配输出，并直接调用同一个 Triton kernel；eager 路径两边都调用会分配输出的公开接口。

**默认测量是复用地址的稳定态，包含 L2 缓存效果，没有强制清空 L2。** 小 N 或 shared 索引可能完全命中缓存。`logical_GB/s` 仅按每 token 的逻辑输入/输出字节估算，并非实测 HBM 带宽。512+64、int32 索引时为每 token `656 + 1152 + 4 = 1812` B；物理访存、重复读取和缓存命中会改变实际流量。要分析生产收益，需要使用真实索引分布、上下游 kernel 顺序及 Nsight Compute 测量，不能把此比值直接当作端到端生成速度提升。

## 文件

| 文件 | 用途 |
|---|---|
| `dequant_cuda.cu` | CUDA kernel、dispatch、PyTorch C++ 绑定 |
| `kv_dequant.py` | JIT 构建与同名 Python 接口 |
| `reference.py` | 输入打包、PyTorch 参考、上游预分配输出启动器 |
| `validate.py` | GPU 正确性验证 |
| `benchmark.py` | CUDA/Triton 对比与 CSV 输出 |
| `bench_dsa.py` | topk/pool/输入长度目标场景对比与 CSV 输出 |
| `upstream_dsa.py` | 固定 commit 的原始 SGLang 参考源码（需 sm89+） |
| `upstream_sm86.py` | 上游 kernel 的 sm86 适配版（uint8 load + 软件 E4M3 解码） |
| `results_dsa_optimized.csv` | 目标场景最终实测数据（含 metadata JSON） |
| `check_host.py` / `VALIDATION.txt` | 主机侧检查及已执行结果 |
| `PROVENANCE.json` / `NOTICE` / `LICENSE` | 来源、状态与许可 |
