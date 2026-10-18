"""GPU correctness checks, including upstream Triton comparison."""
import argparse
import torch

from kv_dequant import build, dequantize_k_cache_paged as cuda_dequant
from reference import make_cache, torch_reference, assert_matches, load_upstream, upstream_out_runner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-triton", action="store_true")
    parser.add_argument("--upstream-file")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU, CUDA-enabled PyTorch and nvcc are required")
    torch.manual_seed(20261018)
    build()
    upstream = None if args.skip_triton else load_upstream(args.upstream_file)
    checked = 0
    for nope, rope in [(128, 0), (128, 64), (256, 0), (256, 64), (384, 64), (512, 64), (1024, 0)]:
        for padding in (0, 4):
            cache = make_cache(40, nope, rope, padding=padding)
            for n in (0, 1, 3, 7, 32, 97):
                for dtype in (torch.int32, torch.int64):
                    ids = torch.randint(0, 40, (n,), device="cuda", dtype=dtype)
                    if n >= 3:
                        ids[:3] = torch.tensor([39, 0, 39], device="cuda", dtype=dtype)
                    expected = torch_reference(cache, ids)
                    # 4D page view, including rows separated by padding bytes.
                    paged = cache.view(5, 8, 1, cache.shape[-1])
                    for warps in (4, 8):
                        for vec in (0, 4, 8, 16):
                            if vec and (cache.data_ptr() % vec or cache.stride(0) % vec):
                                continue
                            actual = cuda_dequant(paged, ids, vec=vec, warps=warps)
                            assert_matches(actual, expected, nope)
                            checked += 1
                    if upstream and n:
                        baseline = torch.empty_like(expected)
                        upstream_out_runner(upstream, paged, ids, baseline)()
                        assert_matches(baseline, expected, nope)

    # Exhaust all 256 FP8 encodings. Non-power-of-two and signed scales test
    # FP32 multiplication and BF16 rounding, not just a dtype conversion.
    for scale in (1.0, 0.1, -1.3, 257.125, 0.0):
        cache = make_cache(8, 512, 64, all_codes=True)
        cache.view(8, -1)[:, 512:528].view(torch.float32).fill_(scale)
        ids = torch.tensor([7, 0, 3, 3, 2, 5, 1, 7, 6, 4, 0], device="cuda", dtype=torch.int64)
        expected = torch_reference(cache, ids)
        for vec in (4, 8, 16):
            assert_matches(cuda_dequant(cache, ids, vec=vec), expected, 512)
            checked += 1
        if upstream:
            baseline = upstream.dequantize_k_cache_paged(cache, ids)
            assert_matches(baseline, expected, 512)

    # Large-N launch (multi-MiB streaming working set) with auto and explicit
    # CTA sizes.
    cache = make_cache(8192, 512, 64)
    ids = torch.randint(0, 8192, (6000,), device="cuda", dtype=torch.int64)
    expected = torch_reference(cache, ids)
    for warps in (0, 4, 8):
        assert_matches(cuda_dequant(cache, ids, warps=warps), expected, 512)
        checked += 1

    # Positive-stride ids; current-stream ordering; preallocated output; graph.
    cache = make_cache(64, 512, 64)
    ids = torch.arange(128, dtype=torch.int64, device="cuda")[::2] % 64
    id_storage = torch.zeros(128, dtype=torch.int64, device="cuda")
    id_storage[::2] = ids
    ids = id_storage[::2]
    out = torch.empty((64, 1, 576), dtype=torch.bfloat16, device="cuda")
    expected = torch_reference(cache, ids)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        out.fill_(float("nan"))
        returned = cuda_dequant(cache, ids, out=out)
        assert returned is out
    torch.cuda.current_stream().wait_stream(stream)
    assert_matches(out, expected, 512)
    # Warm first, so no extension load/cuda initialization happens in capture.
    cuda_dequant(cache, ids, out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cuda_dequant(cache, ids, out=out)
    id_storage[::2].fill_(31)
    graph.replay()
    assert_matches(out, torch_reference(cache, ids), 512)

    for bad in (-1, 64):
        invalid = torch.tensor([bad], device="cuda", dtype=torch.int32)
        try:
            cuda_dequant(cache, invalid, check_indices=True)
        except IndexError:
            pass
        else:
            raise AssertionError("debug index check did not reject invalid index")
    torch.cuda.synchronize()
    print(f"PASS: {checked} CUDA variants + dtype codes + stream/graph + invalid-index checks")
    print("Triton comparison:", "SKIPPED" if args.skip_triton else "PASS")


if __name__ == "__main__":
    main()
