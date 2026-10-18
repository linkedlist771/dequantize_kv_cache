"""Input generation and independent PyTorch arithmetic reference (GPU required)."""
import importlib.util
from pathlib import Path
import sys

import torch

from kv_dequant import infer_dims


def make_cache(slots, nope=512, rope=64, *, device="cuda", padding=0, all_codes=False):
    if nope <= 0 or nope % 128 or rope not in (0, 64) or padding % 4:
        raise ValueError("invalid packed-cache dimensions or padding")
    groups = nope // 128
    width = nope + groups * 4 + rope * 2
    # uint8 is used for packing: scale/RoPE bytes must never be numerically
    # converted to FP8 even though the outer container has FP8 dtype.
    storage = torch.empty((slots, width + padding), dtype=torch.uint8, device=device)
    raw = storage[:, :width]
    if all_codes:
        codes = (torch.arange(slots * nope, device=device) % 256).to(torch.uint8).view(slots, nope)
    else:
        codes = torch.randint(0, 256, (slots, nope), dtype=torch.uint8, device=device)
        codes = torch.where((codes & 127) == 127, 126, codes).to(torch.uint8)
    raw[:, :nope].copy_(codes)
    scales = torch.exp2(torch.rand((slots, groups), dtype=torch.float32, device=device) * 16 - 8)
    raw[:, nope:nope + 4 * groups].copy_(scales.view(torch.uint8))
    if rope:
        raw[:, nope + 4 * groups:].copy_(
            torch.randint(0, 256, (slots, 2 * rope), dtype=torch.uint8, device=device)
        )
    return raw.view(torch.float8_e4m3fn).unsqueeze(1)


def torch_reference(cache, indices):
    """Explicit gather + FP32 multiply + BF16 rounding, plus raw RoPE copy."""
    nope, rope = infer_dims(cache.shape[-1])
    groups = nope // 128
    raw = cache.view(-1, cache.shape[-1]).view(torch.uint8)
    # uint8 gather works even if float8 index_select isn't implemented by PyTorch.
    picked = raw.index_select(0, indices.to(torch.int64))
    q = picked[:, :nope].view(torch.float8_e4m3fn).to(torch.float32)
    scale = picked[:, nope:nope + 4 * groups].view(torch.float32)
    result = torch.empty((indices.numel(), 1, nope + rope), dtype=torch.bfloat16, device=cache.device)
    result[:, 0, :nope].copy_((q.view(-1, groups, 128) * scale[:, :, None]).flatten(1).to(torch.bfloat16))
    if rope:
        result[:, 0, nope:].view(torch.uint8).copy_(picked[:, nope + 4 * groups:])
    return result


def assert_matches(actual, expected, nope):
    """All finite NoPE bits must match; NaN payloads may differ. RoPE is bit exact."""
    a, b = actual[:, 0, :nope], expected[:, 0, :nope]
    same = (a.view(torch.int16) == b.view(torch.int16)) | (a.isnan() & b.isnan())
    if not bool(same.all()):
        mismatch = (~same).nonzero()[0].tolist()
        i, j = mismatch
        raise AssertionError(f"NoPE mismatch [{i}, {j}]: {a[i,j].item()} != {b[i,j].item()}")
    if not torch.equal(actual[:, 0, nope:].view(torch.int16), expected[:, 0, nope:].view(torch.int16)):
        raise AssertionError("RoPE raw bits differ")


def load_upstream(path=None):
    path = Path(path) if path else Path(__file__).with_name("upstream_dsa.py")
    spec = importlib.util.spec_from_file_location("_dsa_upstream_for_comparison", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def upstream_out_runner(module, cache, indices, out):
    """Launch the UNMODIFIED upstream kernel with a preallocated output.

    Only the Python allocating wrapper is replaced for fair kernel-only timing.
    The upstream kernel assumes contiguous indices.
    """
    if not indices.is_contiguous():
        raise ValueError("upstream kernel requires contiguous indices")
    nope, rope = infer_dims(cache.shape[-1])
    groups = nope // 128
    flat = cache.view(-1, cache.shape[-1])
    q = flat[:, :nope]
    s = flat[:, nope:nope + 4 * groups].view(torch.float32)
    r = flat[:, nope + 4 * groups:].view(torch.bfloat16)
    grid = (indices.numel(), groups + (rope + 127) // 128)

    def run():
        if indices.numel():
            module._dequantize_k_cache_paged_kernel[grid](
                out, q, s, r, indices,
                out.stride(0), q.stride(0), s.stride(0), r.stride(0),
                NUM_NOPE_BLOCKS=groups, GROUP_SIZE=128, DIM_NOPE=nope, DIM_ROPE=rope,
            )
        return out
    return run
