"""Compare CUDA C++ against the pinned, unmodified SGLang Triton kernel.

graph: preallocated output, GPU event timing, repeated workload (warm cache).
eager: allocating public APIs, synchronized wall-clock timing (includes host).
All reported timings are produced on the machine running this script.
"""
import argparse
import csv
from functools import partial
import hashlib
import json
from pathlib import Path
import platform
import statistics
import time

import torch
import triton

from kv_dequant import build, dequantize_k_cache_paged as cuda_dequant
from reference import make_cache, torch_reference, assert_matches, load_upstream, upstream_out_runner


def capture_runner(fn, inner):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            fn()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph


def graph_sample(graph, inner):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / inner


def eager_sample(fn, inner):
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(inner):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1e6 / inner


def paired_times(cuda_fn, triton_fn, mode, inner, samples):
    if mode == "graph":
        runners = [capture_runner(cuda_fn, inner), capture_runner(triton_fn, inner)]
        measure = lambda i: graph_sample(runners[i], inner)
    else:
        for _ in range(5):
            cuda_fn()
            triton_fn()
        runners = [cuda_fn, triton_fn]
        measure = lambda i: eager_sample(runners[i], inner)
    data = [[], []]
    # Alternate measurement order to reduce systematic clock/thermal bias.
    for sample in range(samples):
        for i in ((0, 1) if sample % 2 == 0 else (1, 0)):
            data[i].append(measure(i))
    return [statistics.median(values) for values in data]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens", type=int, nargs="+", default=[1, 32, 512, 4096, 32768, 131072])
    p.add_argument("--pool", type=int, default=262144)
    p.add_argument("--nope", type=int, default=512)
    p.add_argument("--rope", type=int, default=64)
    p.add_argument("--pattern", nargs="+", choices=["sequential", "random", "shared"], default=["sequential", "random"])
    p.add_argument("--index-dtype", choices=["int32", "int64"], default="int32")
    p.add_argument("--vec", type=int, choices=[0, 4, 8, 16], default=0)
    p.add_argument("--warps", type=int, choices=[4, 8], default=4)
    p.add_argument("--mode", choices=["graph", "eager", "both"], default="graph")
    p.add_argument("--inner", type=int, default=50)
    p.add_argument("--samples", type=int, default=9)
    p.add_argument("--upstream-file", type=Path)
    p.add_argument("--csv", type=Path)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required")
    if min(args.tokens + [args.pool, args.inner, args.samples]) <= 0:
        p.error("tokens, pool, inner and samples must be positive")
    torch.manual_seed(20261018)
    build()
    upstream_path = args.upstream_file or Path(__file__).with_name("upstream_dsa.py")
    upstream = load_upstream(upstream_path)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    metadata = {
        "gpu": props.name, "sm": f"{props.major}.{props.minor}",
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "triton": triton.__version__, "python": platform.python_version(),
        "upstream_path": str(upstream_path.resolve()),
        "upstream_sha256": hashlib.sha256(upstream_path.read_bytes()).hexdigest(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "cache_policy": "same input/output addresses reused; warm steady-state, not forced HBM traffic",
    }
    print(json.dumps(metadata, ensure_ascii=False))
    cache = make_cache(args.pool, args.nope, args.rope)
    dtype = getattr(torch, args.index_dtype)
    modes = ["graph", "eager"] if args.mode == "both" else [args.mode]
    rows = []
    print("mode  pattern      N        CUDA_us  Triton_us  Triton/CUDA  CUDA_logical_GB/s")
    for pattern in args.pattern:
        for n in args.tokens:
            if pattern == "sequential":
                ids = (torch.arange(n, dtype=torch.int64, device="cuda") % args.pool).to(dtype)
            elif pattern == "random":
                ids = torch.randint(args.pool, (n,), dtype=dtype, device="cuda")
            else:
                ids = torch.randint(min(args.pool, 128), (n,), dtype=dtype, device="cuda")
            out_cuda = torch.empty((n, 1, args.nope + args.rope), dtype=torch.bfloat16, device="cuda")
            out_triton = torch.empty_like(out_cuda)
            cuda_out_fn = partial(cuda_dequant, cache, ids, out=out_cuda, vec=args.vec, warps=args.warps)
            triton_out_fn = upstream_out_runner(upstream, cache, ids, out_triton)
            cuda_out_fn()
            triton_out_fn()
            expected = torch_reference(cache, ids)
            assert_matches(out_cuda, expected, args.nope)
            assert_matches(out_triton, expected, args.nope)
            del expected
            for mode in modes:
                if mode == "graph":
                    cuda_fn, triton_fn = cuda_out_fn, triton_out_fn
                else:
                    cuda_fn = partial(cuda_dequant, cache, ids, vec=args.vec, warps=args.warps)
                    triton_fn = partial(upstream.dequantize_k_cache_paged, cache, ids)
                cuda_us, triton_us = paired_times(cuda_fn, triton_fn, mode, args.inner, args.samples)
                # Unique logical input bytes + output bytes + one index per token.
                # This is not measured HBM traffic: caching/reuse and instruction
                # duplication change physical traffic substantially.
                logical_bytes = n * (cache.shape[-1] + 2 * (args.nope + args.rope) + ids.element_size())
                row = dict(mode=mode, pattern=pattern, tokens=n, pool=args.pool,
                           nope=args.nope, rope=args.rope, vec=args.vec, warps=args.warps,
                           cuda_us=cuda_us, triton_us=triton_us,
                           speedup=triton_us / cuda_us,
                           cuda_logical_GBs=logical_bytes / (cuda_us * 1000),
                           triton_logical_GBs=logical_bytes / (triton_us * 1000))
                rows.append(row)
                print(f"{mode:5} {pattern:10} {n:8d} {cuda_us:10.3f} {triton_us:10.3f} "
                      f"{row['speedup']:12.3f} {row['cuda_logical_GBs']:18.2f}")
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        args.csv.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
