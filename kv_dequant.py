"""CUDA C++ replacement for SGLang's DSA dequantize_k_cache_paged.

No Triton/SGLang dependency for execution. Inference only; no autograd rule.
"""
from functools import lru_cache
from pathlib import Path
import hashlib

import torch
from torch.utils.cpp_extension import load


def infer_dims(width: int) -> tuple[int, int]:
    for rope in (0, 64):
        scaled = (width - 2 * rope) * 128
        if scaled % 132 == 0:
            nope = scaled // 132
            if nope > 0 and nope % 128 == 0:
                return nope, rope
    raise ValueError(f"Unsupported packed DSA KV row width: {width}")


@lru_cache(maxsize=1)
def build():
    """Compile/load once. Call before CUDA graph capture or timing."""
    source = Path(__file__).with_name("dequant_cuda.cu")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    return load(
        name=f"dsa_dequant_cuda_{digest}",
        sources=[str(source)],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--ftz=false", "--fmad=false", "-lineinfo"],
        verbose=False,
    )


def dequantize_k_cache_paged(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    group_size: int = 128,
    *,
    out: torch.Tensor | None = None,
    vec: int = 0,
    warps: int = 0,
    check_indices: bool = False,
) -> torch.Tensor:
    """Gather token slots, scale E4M3 per 128 elements, and copy BF16 RoPE.

    cache: [slots, 1, packed_bytes] or [pages, page_size, 1, packed_bytes]
           (a [slots, packed_bytes] view is also accepted).
    indices: [N], CUDA int32/int64, flattened TOKEN SLOT indices.
    out: optional contiguous [N, 1, nope + rope] BF16 buffer.
    vec: FP8 elements/thread/load, 0 chooses an alignment-safe default.
    warps: tokens per CTA; 0 (auto) uses 8, or pass 4/8 to compare.
    check_indices: debug-only bounds validation; synchronizes the host.

    With check_indices=False indices MUST be in [0, slots); -1 is not a
    sentinel. Duplicates and N > slots are supported. No implicit tensor
    copies, device transfers, or device synchronizations in the fast path.
    """
    if group_size != 128:
        raise ValueError("This DSA layout has one FP32 scale per 128 FP8 elements")
    if quant_k_cache.ndim not in (2, 3, 4):
        raise ValueError("cache must be 2D, 3D, or 4D")
    if quant_k_cache.ndim >= 3 and quant_k_cache.shape[-2] != 1:
        raise ValueError("DSA cache requires a singleton KV-head dimension")
    if quant_k_cache.dtype != torch.float8_e4m3fn or not quant_k_cache.is_cuda:
        raise ValueError("cache must be CUDA float8_e4m3fn")
    if page_table_1_flattened.ndim != 1:
        raise ValueError("indices must be 1D")
    if page_table_1_flattened.dtype not in (torch.int32, torch.int64):
        raise ValueError("indices must be int32/int64")
    if page_table_1_flattened.device != quant_k_cache.device:
        raise ValueError("cache and indices must be on the same CUDA device")
    nope, rope = infer_dims(quant_k_cache.shape[-1])
    cache = quant_k_cache.view(-1, quant_k_cache.shape[-1])
    n = page_table_1_flattened.numel()
    if check_indices and n:
        lo, hi = torch.aminmax(page_table_1_flattened)
        if lo.item() < 0 or hi.item() >= cache.shape[0]:
            raise IndexError(f"token slot index must lie in [0, {cache.shape[0]})")
    if out is None:
        out = torch.empty((n, 1, nope + rope), dtype=torch.bfloat16, device=cache.device)
    build().run(cache, page_table_1_flattened, out, nope, rope, vec, warps)
    return out
