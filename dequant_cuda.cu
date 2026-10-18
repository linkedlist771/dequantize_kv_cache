// SPDX-License-Identifier: Apache-2.0
// CUDA implementation of the SGLang DSA packed-cache contract.
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

// E4M3 -> half is exact for every finite FP8 value. Scaling is deliberately
// FP32, followed by one round-to-nearest-even BF16 conversion.
__device__ __forceinline__ uint32_t dequant_pair(uint16_t packed, float scale) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(packed, __NV_E4M3);
  const float x = __half2float(__ushort_as_half(h.x));
  const float y = __half2float(__ushort_as_half(h.y));
  const uint32_t lo = __bfloat16_as_ushort(__float2bfloat16_rn(__fmul_rn(x, scale)));
  const uint32_t hi = __bfloat16_as_ushort(__float2bfloat16_rn(__fmul_rn(y, scale)));
  return lo | (hi << 16);
}

template <int VEC>
__device__ __forceinline__ void convert_vector(
    const uint8_t* src, uint16_t* dst, float scale) {
  uint32_t q[VEC / 4];
  if constexpr (VEC == 4) {
    q[0] = *reinterpret_cast<const uint32_t*>(src);
  } else if constexpr (VEC == 8) {
    const uint2 v = *reinterpret_cast<const uint2*>(src);
    q[0] = v.x; q[1] = v.y;
  } else {
    const uint4 v = *reinterpret_cast<const uint4*>(src);
    q[0] = v.x; q[1] = v.y; q[2] = v.z; q[3] = v.w;
  }
  uint32_t result[VEC / 2];
  #pragma unroll
  for (int i = 0; i < VEC / 4; ++i) {
    result[2 * i] = dequant_pair(static_cast<uint16_t>(q[i]), scale);
    result[2 * i + 1] = dequant_pair(static_cast<uint16_t>(q[i] >> 16), scale);
  }
  if constexpr (VEC == 4) {
    *reinterpret_cast<uint2*>(dst) = make_uint2(result[0], result[1]);
  } else {
    *reinterpret_cast<uint4*>(dst) =
        make_uint4(result[0], result[1], result[2], result[3]);
    if constexpr (VEC == 16) {
      *reinterpret_cast<uint4*>(dst + 8) =
          make_uint4(result[4], result[5], result[6], result[7]);
    }
  }
}

template <typename Index, int VEC, int WARPS>
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
    convert_vector<VEC>(row + col, dst + col, scale);
  }
  // Preserve all RoPE BF16 bits, including signed zero and NaN payloads.
  if (rope != 0) {
    const uint8_t* src_rope = row + nope + (nope / 128) * 4;
    for (int col = lane * 2; col < rope; col += 64) {
      *reinterpret_cast<uint32_t*>(dst + nope + col) =
          *reinterpret_cast<const uint32_t*>(src_rope + col * 2);
    }
  }
}

template <typename Index, int VEC>
void launch(int warps, const at::Tensor& cache, const at::Tensor& indices,
            const at::Tensor& out, int64_t nope, int64_t rope, cudaStream_t stream) {
  const int64_t n = indices.numel();
  const auto blocks = static_cast<unsigned int>((n + warps - 1) / warps);
  const auto* src = reinterpret_cast<const uint8_t*>(cache.data_ptr());
  auto* dst = reinterpret_cast<uint16_t*>(out.data_ptr());
  if (warps == 4) {
    dequant_paged_kernel<Index, VEC, 4><<<blocks, 128, 0, stream>>>(
        src, indices.data_ptr<Index>(), dst, n, cache.stride(0), indices.stride(0), nope, rope);
  } else {
    dequant_paged_kernel<Index, VEC, 8><<<blocks, 256, 0, stream>>>(
        src, indices.data_ptr<Index>(), dst, n, cache.stride(0), indices.stride(0), nope, rope);
  }
}

template <typename Index>
void dispatch(int vec, int warps, const at::Tensor& cache, const at::Tensor& indices,
              const at::Tensor& out, int64_t nope, int64_t rope, cudaStream_t stream) {
  switch (vec) {
    case 4: launch<Index, 4>(warps, cache, indices, out, nope, rope, stream); break;
    case 8: launch<Index, 8>(warps, cache, indices, out, nope, rope, stream); break;
    case 16: launch<Index, 16>(warps, cache, indices, out, nope, rope, stream); break;
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
  TORCH_CHECK(warps == 4 || warps == 8, "warps must be 4 or 8");
  TORCH_CHECK(vec == 0 || vec == 4 || vec == 8 || vec == 16, "vec must be 0(auto), 4, 8, or 16");
  if (indices.numel() == 0) return;
  TORCH_CHECK(cache.size(0) > 0, "cannot gather from an empty cache");
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
  if (indices.scalar_type() == at::kInt) {
    dispatch<int32_t>(vec, warps, cache, indices, out, nope, rope, stream);
  } else {
    dispatch<int64_t>(vec, warps, cache, indices, out, nope, rope, stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &run, "DSA FP8 paged gather/dequantize into BF16 (CUDA)");
}
