"""sm86 (Ampere) adaptation of the pinned upstream SGLang Triton kernel.

Triton refuses to compile `fp8e4nv` (E4M3) kernels on compute capability < 89:
the stock `_dequantize_k_cache_paged_kernel` cannot run on e.g. a RTX 3070 at
all ("type fp8e4nv not supported in this architecture"). This file keeps the
upstream kernel byte-for-byte identical in structure (same grid, same
per-(token, 128-block) program decomposition, same masks, same scale loads,
same RoPE copy, same bf16 rounding) and only replaces the fp8 load + hardware
conversion with a uint8 load plus an exact software decode of E4M3 into fp32:

  exp>=1 : bits = sign | ((exp+120)<<23) | (mant<<20)      (127-7 = 120)
  exp==0 : (1+m/8)*2^-6 - 2^-6 == m*2^-9  via the exp=1 pattern, exact
  exp==15 & mant==7 : NaN (e4m3fn has no infinity)

Every finite value decodes bit-exactly like the hardware/CUDA path, so the
arithmetic after the load (fp32 multiply, bf16 RNE store) is unchanged.
Both sides pay a software FP8 decode on this GPU; on sm89+ the stock kernel
should be used instead.
"""
import torch
import triton
import triton.language as tl

# Signal to reference.upstream_out_runner that the q operand must be uint8.
UINT8_NOPE_INPUT = True


def _infer_dsa_dims(dim_quant: int, tile_size: int = 128) -> tuple[int, int]:
    """Recover the NoPE and RoPE widths from a packed DSA cache row."""
    for dim_rope in (0, 64):
        scaled_width = (dim_quant - 2 * dim_rope) * tile_size
        if scaled_width % (tile_size + 4):
            continue
        dim_nope = scaled_width // (tile_size + 4)
        if dim_nope > 0 and dim_nope % tile_size == 0:
            return dim_nope, dim_rope
    raise ValueError(f"Unsupported packed DSA KV row width: {dim_quant}")


@triton.jit
def _fp8e4m3fn_to_f32(u):
    # u: uint8 tile -> exact fp32 value of the E4M3FN encoding.
    u = u.to(tl.uint32)
    sign = (u & 0x80) << 24
    exp = (u >> 3) & 0xF
    mant = (u & 0x7) << 20
    normal = (sign | ((exp + 120) << 23) | mant).to(tl.float32, bitcast=True)
    # exp==0: magnitude (1+m/8)*2^-6 - 2^-6 = m*2^-9 exactly; apply the sign
    # to the subtracted magnitude, not to the subtrahend.
    sub_mag = ((121 << 23) | mant).to(tl.float32, bitcast=True) - 0.015625
    sub = (sign | sub_mag.to(tl.uint32, bitcast=True)).to(tl.float32, bitcast=True)
    out = tl.where(exp == 0, sub, normal)
    # 0x7F/0xFF are the only NaN encodings of e4m3fn.
    return tl.where((exp == 15) & ((u & 0x7) == 7), float("nan"), out)


@triton.jit
def _dequantize_k_cache_paged_kernel(
    output_ptr,
    input_nope_q_ptr,
    input_nope_s_ptr,
    input_rope_ptr,
    page_table_1_ptr,
    output_stride_0: int,
    input_nope_q_stride_0: int,
    input_nope_s_stride_0: int,
    input_rope_stride_0: int,
    NUM_NOPE_BLOCKS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    DIM_ROPE: tl.constexpr,
):
    token_id = tl.program_id(0).to(tl.int64)
    token_id_paged = tl.load(page_table_1_ptr + token_id).to(tl.int64)
    raw_block_id = tl.program_id(1)

    if raw_block_id < NUM_NOPE_BLOCKS:
        # a. dequant nope
        effective_block_id = raw_block_id

        offs_q = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs_q < DIM_NOPE
        ptr_q = input_nope_q_ptr + token_id_paged * input_nope_q_stride_0 + offs_q
        ptr_s = (
            input_nope_s_ptr
            + token_id_paged * input_nope_s_stride_0
            + effective_block_id
        )

        y_q = _fp8e4m3fn_to_f32(tl.load(ptr_q, mask=mask, other=0))
        y_s = tl.load(ptr_s)

        y = (y_q * y_s).to(output_ptr.dtype.element_ty)

        dst_ptr = output_ptr + token_id * output_stride_0 + offs_q
        tl.store(dst_ptr, y, mask=mask)
    else:
        # b. copy rope
        effective_block_id = raw_block_id - NUM_NOPE_BLOCKS

        offs = effective_block_id * GROUP_SIZE + tl.arange(0, GROUP_SIZE)
        mask = offs < DIM_ROPE

        src_ptr = input_rope_ptr + token_id_paged * input_rope_stride_0 + offs
        dst_ptr = output_ptr + token_id * output_stride_0 + DIM_NOPE + offs

        data = tl.load(src_ptr, mask=mask).to(tl.bfloat16)
        tl.store(dst_ptr, data, mask=mask)


def dequantize_k_cache_paged(
    quant_k_cache: torch.Tensor,
    page_table_1_flattened: torch.Tensor,
    group_size: int = 128,
) -> torch.Tensor:
    """Same contract as upstream, but q rows are passed to Triton as uint8."""
    dim_quant = quant_k_cache.shape[-1]
    dim_nope, dim_rope = _infer_dsa_dims(dim_quant)
    quant_k_cache = quant_k_cache.view((-1, dim_quant))

    num_tokens = page_table_1_flattened.shape[0]
    assert quant_k_cache.dtype == torch.float8_e4m3fn
    num_tiles = dim_nope // group_size

    output = torch.empty(
        (num_tokens, 1, dim_nope + dim_rope),
        dtype=torch.bfloat16,
        device=quant_k_cache.device,
    )

    num_blocks_per_token = num_tiles + triton.cdiv(dim_rope, group_size)

    assert dim_nope % group_size == 0

    input_nope_q = quant_k_cache[:, :dim_nope].view(torch.uint8)
    input_nope_s = quant_k_cache[:, dim_nope : dim_nope + num_tiles * 4].view(
        torch.float32
    )
    input_rope = quant_k_cache[:, dim_nope + num_tiles * 4 :].view(torch.bfloat16)

    _dequantize_k_cache_paged_kernel[(num_tokens, num_blocks_per_token)](
        output,
        input_nope_q,
        input_nope_s,
        input_rope,
        page_table_1_flattened,
        output.stride(0),
        input_nope_q.stride(0),
        input_nope_s.stride(0),
        input_rope.stride(0),
        NUM_NOPE_BLOCKS=num_tiles,
        GROUP_SIZE=group_size,
        DIM_NOPE=dim_nope,
        DIM_ROPE=dim_rope,
    )

    return output
