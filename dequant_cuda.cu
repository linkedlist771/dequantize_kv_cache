// SPDX-License-Identifier: Apache-2.0
// CUDA implementation of the SGLang DSA packed-cache contract.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

// Direct E4M3FN -> FP32 bit construction. Exact for every finite code, no
// FP16 detour and no variable-length subnormal normalization loop (the
// cuda_fp8.h emulation loop made small-N launches instruction bound).
//   normal    : bits = sign | ((exp4 + 120) << 23) | (mant << 20)
//   subnormal : (1+m/8)*2^-6 - 2^-6 == m*2^-9, sign applied after the
//               subtraction so negative values stay exact
//   0x7F/0xFF: e4m3fn NaN -> canonical NaN (payload unspecified, as upstream)
__device__ __forceinline__ float e4m3_to_f32(uint32_t u) {
  const uint32_t mag = u & 0x7Fu;
  const uint32_t sign = (u & 0x80u) << 24;
  float f = __uint_as_float(((mag << 20) + 0x3C000000u) | sign);
  if (mag < 8u) {
    f = __uint_as_float(sign | __float_as_uint(
        __uint_as_float(0x3C800000u | (mag << 20)) - 0.015625f));
  }
  if (mag == 0x7Fu) f = __uint_as_float(0x7FC00000u);
  return f;
}

// Scale in FP32 and one round-to-nearest-even BF16 conversion per value;
// __float22bfloat162_rn packs the pair in a single instruction.
__device__ __forceinline__ uint32_t dequant_pair(uint16_t packed, float scale) {
  const float x = __fmul_rn(e4m3_to_f32(packed & 0xFFu), scale);
  const float y = __fmul_rn(e4m3_to_f32(packed >> 8), scale);
  const __nv_bfloat162 r = __float22bfloat162_rn(make_float2(x, y));
  return *reinterpret_cast<const uint32_t*>(&r);
}

// Gathered cache rows and the output are streamed once per call. Loads are
// always evict-first; the store policy adapts to the working set (below).
template <typename T>
__device__ __forceinline__ T ld_stream(const T* p) { return __ldcs(p); }
// ST_WB: evict-first streaming store (best when the output does not fit L2).
// ST_WT: write-through store (measured faster while the whole output stays
//        inside L2: no dirty-line write-back management to fight the gather).
enum { ST_WB = 0, ST_WT = 1 };
template <int STMODE, typename T>
__device__ __forceinline__ void st_stream(T* p, T v) {
  if constexpr (STMODE == ST_WB) __stcs(p, v);
  else __stwt(p, v);
}

template <int VEC, int STMODE>
__device__ __forceinline__ void convert_vector(
    const uint8_t* src, uint16_t* dst, float scale) {
  uint32_t q[VEC / 4];
  if constexpr (VEC == 4) {
    q[0] = ld_stream(reinterpret_cast<const uint32_t*>(src));
  } else if constexpr (VEC == 8) {
    const uint2 v = ld_stream(reinterpret_cast<const uint2*>(src));
    q[0] = v.x; q[1] = v.y;
  } else {
    const uint4 v = ld_stream(reinterpret_cast<const uint4*>(src));
    q[0] = v.x; q[1] = v.y; q[2] = v.z; q[3] = v.w;
  }
  uint32_t result[VEC / 2];
  #pragma unroll
  for (int i = 0; i < VEC / 4; ++i) {
    result[2 * i] = dequant_pair(static_cast<uint16_t>(q[i]), scale);
    result[2 * i + 1] = dequant_pair(static_cast<uint16_t>(q[i] >> 16), scale);
  }
  if constexpr (VEC == 4) {
    st_stream<STMODE>(reinterpret_cast<uint2*>(dst), make_uint2(result[0], result[1]));
  } else {
    st_stream<STMODE>(reinterpret_cast<uint4*>(dst),
                      make_uint4(result[0], result[1], result[2], result[3]));
    if constexpr (VEC == 16) {
      st_stream<STMODE>(reinterpret_cast<uint4*>(dst + 8),
                        make_uint4(result[4], result[5], result[6], result[7]));
    }
  }
}

// Generic kernel: runtime NoPE/RoPE widths (any 128-multiple NoPE, RoPE 0/64).
template <typename Index, int VEC, int WARPS, int STMODE>
__global__ void dequant_paged_kernel(
    const uint8_t* __restrict__ cache,
    const Index* __restrict__ indices,
    uint16_t* __restrict__ out,
    int64_t count, int64_t cache_stride, int64_t index_stride,
    int64_t nope, int64_t rope) {
  const int lane = threadIdx.x & 31;
  const int64_t token = static_cast<int64_t>(blockIdx.x) * WARPS + threadIdx.x / 32;
  if (token >= count) return;

  // indices are flattened TOKEN SLOT ids, not page ids. Duplicate ids are valid.
  // As in upstream, bounds are the caller's responsibility.
  const int64_t slot = static_cast<int64_t>(indices[token * index_stride]);
  const uint8_t* row = cache + slot * cache_stride;
  uint16_t* dst = out + token * (nope + rope);
  for (int64_t col = lane * VEC; col < nope; col += 32 * VEC) {
    // VEC divides 128, so one vector cannot cross a quantization group.
    const float scale = *reinterpret_cast<const float*>(row + nope + (col / 128) * 4);
    convert_vector<VEC, STMODE>(row + col, dst + col, scale);
  }
  // Preserve all RoPE BF16 bits, including signed zero and NaN payloads.
  if (rope != 0) {
    const uint8_t* src_rope = row + nope + (nope / 128) * 4;
    for (int col = lane * 2; col < rope; col += 64) {
      st_stream<STMODE>(reinterpret_cast<uint32_t*>(dst + nope + col),
                        *reinterpret_cast<const uint32_t*>(src_rope + col * 2));
    }
  }
}

// Shape-specialized kernel: compile-time NoPE/RoPE turns the column loop,
// scale-group index and RoPE offset into constants (the address chain to the
// first gather load gets measurably shorter; -14% at N=2048 on sm86).
template <typename Index, int NOPE, int ROPE, int VEC, int WARPS, int STMODE>
__global__ void dequant_paged_spec(
    const uint8_t* __restrict__ cache,
    const Index* __restrict__ indices,
    uint16_t* __restrict__ out,
    int64_t count, int64_t cache_stride, int64_t index_stride) {
  constexpr int64_t kRopeOff = NOPE + (NOPE / 128) * 4;
  constexpr int64_t kOutW = NOPE + ROPE;
  const int lane = threadIdx.x & 31;
  const int64_t token = static_cast<int64_t>(blockIdx.x) * WARPS + threadIdx.x / 32;
  if (token >= count) return;
  const int64_t slot = static_cast<int64_t>(indices[token * index_stride]);
  const uint8_t* row = cache + slot * cache_stride;
  uint16_t* dst = out + token * kOutW;
  #pragma unroll
  for (int col = lane * VEC; col < NOPE; col += 32 * VEC) {
    const float scale = *reinterpret_cast<const float*>(row + NOPE + (col / 128) * 4);
    convert_vector<VEC, STMODE>(row + col, dst + col, scale);
  }
  if (ROPE != 0) {
    const uint8_t* src_rope = row + kRopeOff;
    #pragma unroll
    for (int col = lane * 2; col < ROPE; col += 64) {
      st_stream<STMODE>(reinterpret_cast<uint32_t*>(dst + NOPE + col),
                        *reinterpret_cast<const uint32_t*>(src_rope + col * 2));
    }
  }
}

constexpr int natural_vec(int64_t nope) {
  return nope >= 512 ? 16 : nope >= 256 ? 8 : 4;
}

// L2 size in bytes, cached per device (the attribute query is cheap but not
// free, and run() is on the latency-critical path).
int l2_cache_bytes(int device) {
  static int cached[64] = {};
  if (device >= 0 && device < 64) {
    if (!cached[device]) {
      int v = 0;
      cudaDeviceGetAttribute(&v, cudaDevAttrL2CacheSize, device);
      cached[device] = v > 0 ? v : 1;
    }
    return cached[device];
  }
  return 4 << 20;  // conservative default
}

template <typename Index, int STMODE>
void launch_spec(int64_t nope, int64_t rope, const at::Tensor& cache,
                 const at::Tensor& indices, const at::Tensor& out,
                 cudaStream_t stream) {
  const int64_t n = indices.numel();
  const auto blocks = static_cast<unsigned>((n + 7) / 8);
  const auto* src = reinterpret_cast<const uint8_t*>(cache.data_ptr());
  auto* dst = reinterpret_cast<uint16_t*>(out.data_ptr());
  const auto* idx = indices.data_ptr<Index>();
  const int64_t cs = cache.stride(0), is = indices.stride(0);
#define SPEC_K(NP, RP)                                                        \
  dequant_paged_spec<Index, NP, RP, natural_vec(NP), 8, STMODE>               \
      <<<blocks, 256, 0, stream>>>(src, idx, dst, n, cs, is)
  switch (nope * 2 + (rope != 0 ? 1 : 0)) {
    case 128 * 2: SPEC_K(128, 0); break;   case 128 * 2 + 1: SPEC_K(128, 64); break;
    case 256 * 2: SPEC_K(256, 0); break;   case 256 * 2 + 1: SPEC_K(256, 64); break;
    case 384 * 2: SPEC_K(384, 0); break;   case 384 * 2 + 1: SPEC_K(384, 64); break;
    case 512 * 2: SPEC_K(512, 0); break;   case 512 * 2 + 1: SPEC_K(512, 64); break;
    case 768 * 2: SPEC_K(768, 0); break;   case 768 * 2 + 1: SPEC_K(768, 64); break;
    case 1024 * 2: SPEC_K(1024, 0); break; case 1024 * 2 + 1: SPEC_K(1024, 64); break;
    default: TORCH_CHECK(false, "unhandled specialized shape");
  }
#undef SPEC_K
}

template <typename Index, int STMODE>
void launch_generic(int vec, int warps, const at::Tensor& cache, const at::Tensor& indices,
                    const at::Tensor& out, int64_t nope, int64_t rope, cudaStream_t stream) {
  const int64_t n = indices.numel();
  const auto blocks = static_cast<unsigned int>((n + warps - 1) / warps);
  const auto* src = reinterpret_cast<const uint8_t*>(cache.data_ptr());
  auto* dst = reinterpret_cast<uint16_t*>(out.data_ptr());
  const auto* idx = indices.data_ptr<Index>();
  if (warps == 4) {
    switch (vec) {
      case 4: dequant_paged_kernel<Index, 4, 4, STMODE><<<blocks, 128, 0, stream>>>(src, idx, dst, n, cache.stride(0), indices.stride(0), nope, rope); break;
      case 8: dequant_paged_kernel<Index, 8, 4, STMODE><<<blocks, 128, 0, stream>>>(src, idx, dst, n, cache.stride(0), indices.stride(0), nope, rope); break;
      default: dequant_paged_kernel<Index, 16, 4, STMODE><<<blocks, 128, 0, stream>>>(src, idx, dst, n, cache.stride(0), indices.stride(0), nope, rope); break;
    }
  } else {
    switch (vec) {
      case 4: dequant_paged_kernel<Index, 4, 8, STMODE><<<blocks, 256, 0, stream>>>(src, idx, dst, n, cache.stride(0), indices.stride(0), nope, rope); break;
      case 8: dequant_paged_kernel<Index, 8, 8, STMODE><<<blocks, 256, 0, stream>>>(src, idx, dst, n, cache.stride(0), indices.stride(0), nope, rope); break;
      default: dequant_paged_kernel<Index, 16, 8, STMODE><<<blocks, 256, 0, stream>>>(src, idx, dst, n, cache.stride(0), indices.stride(0), nope, rope); break;
    }
  }
}

void run(const at::Tensor& cache, const at::Tensor& indices, const at::Tensor& out,
         int64_t nope, int64_t rope, int vec, int warps) {
  TORCH_CHECK(cache.is_cuda() && indices.is_cuda() && out.is_cuda(), "all tensors must be CUDA");
  TORCH_CHECK(cache.device() == indices.device() && cache.device() == out.device(), "device mismatch");
  TORCH_CHECK(cache.scalar_type() == at::ScalarType::Float8_e4m3fn, "cache must be float8_e4m3fn");
  TORCH_CHECK(indices.scalar_type() == at::kInt || indices.scalar_type() == at::kLong, "indices must be int32/int64");
  TORCH_CHECK(out.scalar_type() == at::kBFloat16, "out must be bfloat16");
  TORCH_CHECK(nope > 0 && nope % 128 == 0 && (rope == 0 || rope == 64), "invalid DSA dimensions");
  TORCH_CHECK(cache.dim() == 2 && cache.stride(1) == 1, "cache must be a row-strided 2D view");
  TORCH_CHECK(cache.size(1) == nope + nope / 128 * 4 + rope * 2, "packed width mismatch");
  TORCH_CHECK(cache.stride(0) >= cache.size(1) && cache.stride(0) % 4 == 0, "invalid cache row stride");
  TORCH_CHECK(indices.dim() == 1 && indices.stride(0) > 0, "indices must be a 1D positive-stride tensor");
  TORCH_CHECK(out.dim() == 3 && out.size(0) == indices.numel() && out.size(1) == 1 &&
              out.size(2) == nope + rope && out.is_contiguous(), "invalid output shape/strides");
  TORCH_CHECK(!out.is_alias_of(cache) && !out.is_alias_of(indices), "out must not share input storage");
  TORCH_CHECK(warps == 0 || warps == 4 || warps == 8, "warps must be 0(auto), 4, or 8");
  TORCH_CHECK(vec == 0 || vec == 4 || vec == 8 || vec == 16, "vec must be 0(auto), 4, 8, or 16");
  if (indices.numel() == 0) return;
  TORCH_CHECK(cache.size(0) > 0, "cannot gather from an empty cache");
  if (warps == 0) {
    // 8-token CTAs measured fastest (or tied) at every tested size; 4 stays
    // available for explicit tuning (and then the generic kernel runs).
    warps = 8;
  }
  TORCH_CHECK((indices.numel() + warps - 1) / warps <= std::numeric_limits<int>::max(), "grid is too large");

  const uintptr_t src_addr = reinterpret_cast<uintptr_t>(cache.data_ptr());
  const uintptr_t dst_addr = reinterpret_cast<uintptr_t>(out.data_ptr());
  auto aligned = [&](int v) {
    return src_addr % v == 0 && cache.stride(0) % v == 0 &&
           dst_addr % std::min(2 * v, 16) == 0;
  };
  if (vec == 0) {
    vec = nope >= 512 && aligned(16) ? 16 : nope >= 256 && aligned(8) ? 8 : 4;
  }
  TORCH_CHECK(aligned(vec), "requested vector width needs aligned cache rows and output");

  const c10::cuda::CUDAGuard guard(cache.device());
  const cudaStream_t stream = c10::cuda::getCurrentCUDAStream(cache.get_device()).stream();
  // While the whole output lives inside L2, write-through beats evict-first
  // stores; once it exceeds L2, streaming write-back wins (crossover sits at
  // the L2 capacity on every measured size).
  const int stmode = indices.numel() * (nope + rope) * 2 < l2_cache_bytes(cache.get_device())
                         ? ST_WT : ST_WB;
  const bool shape_known =
      (nope == 128 || nope == 256 || nope == 384 || nope == 512 || nope == 768 || nope == 1024) &&
      (rope == 0 || rope == 64);
  const bool use_spec = shape_known && warps == 8 && vec == natural_vec(nope);
  if (indices.scalar_type() == at::kInt) {
    if (use_spec) {
      if (stmode == ST_WT) launch_spec<int32_t, ST_WT>(nope, rope, cache, indices, out, stream);
      else launch_spec<int32_t, ST_WB>(nope, rope, cache, indices, out, stream);
    } else if (stmode == ST_WT) {
      launch_generic<int32_t, ST_WT>(vec, warps, cache, indices, out, nope, rope, stream);
    } else {
      launch_generic<int32_t, ST_WB>(vec, warps, cache, indices, out, nope, rope, stream);
    }
  } else {
    if (use_spec) {
      if (stmode == ST_WT) launch_spec<int64_t, ST_WT>(nope, rope, cache, indices, out, stream);
      else launch_spec<int64_t, ST_WB>(nope, rope, cache, indices, out, stream);
    } else if (stmode == ST_WT) {
      launch_generic<int64_t, ST_WT>(vec, warps, cache, indices, out, nope, rope, stream);
    } else {
      launch_generic<int64_t, ST_WB>(vec, warps, cache, indices, out, nope, rope, stream);
    }
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "DSA FP8 paged gather/dequantize into BF16 (CUDA)");
}
