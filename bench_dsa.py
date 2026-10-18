"""DSA scenario benchmark: CUDA vs Triton gather/dequant.

Workload: one decode/prefill step gathers `--topk` scattered token slots
(the sparse-attention top-k selection out of the whole pool) plus the last
`L` slots (the current 1..8k input chunk). Pool defaults to 200k slots,
topk to 2048, matching the target deployment scenario.

Timing methodology is shared with benchmark.py: preallocated outputs, CUDA
graph replay of `--inner` back-to-back launches, event timing, alternating
measurement order, median of `--samples`.
"""
import argparse
import csv
import hashlib
import json
from functools import partial
from pathlib import Path
import platform

import torch
import triton

from kv_dequant import build, dequantize_k_cache_paged as cuda_dequant
from reference import (
    make_cache, torch_reference, assert_matches, load_upstream, upstream_out_runner,
)
from benchmark import paired_times


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=int, default=200000)
    p.add_argument("--topk", type=int, default=2048)
    p.add_argument("--chunks", type=int, nargs="+", default=[0, 1, 128, 512, 1024, 2048, 4096, 8192],
                   help="input-chunk lengths L; gathered N = topk + L")
    p.add_argument("--nope", type=int, default=512)
    p.add_argument("--rope", type=int, default=64)
    p.add_argument("--index-dtype", choices=["int32", "int64"], default="int32")
    p.add_argument("--vec", type=int, choices=[0, 4, 8, 16], default=0)
    p.add_argument("--warps", type=int, choices=[0, 4, 8], default=0)
    p.add_argument("--mode", choices=["graph", "eager", "both"], default="graph")
    p.add_argument("--inner", type=int, default=50)
    p.add_argument("--samples", type=int, default=9)
    p.add_argument("--upstream-file", type=Path, default=Path(__file__).with_name("upstream_sm86.py"))
    p.add_argument("--csv", type=Path)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required")
    torch.manual_seed(20261018)
    build()
    upstream = load_upstream(args.upstream_file)
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    metadata = {
        "gpu": props.name, "sm": f"{props.major}.{props.minor}",
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "triton": triton.__version__, "python": platform.python_version(),
        "upstream_path": str(args.upstream_file.resolve()),
        "upstream_sha256": hashlib.sha256(args.upstream_file.read_bytes()).hexdigest(),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "index_pattern": "first topk ids uniform over [0, pool-L); last L ids = pool-L..pool-1",
    }
    print(json.dumps(metadata, ensure_ascii=False))
    cache = make_cache(args.pool, args.nope, args.rope)
    dtype = getattr(torch, args.index_dtype)
    modes = ["graph", "eager"] if args.mode == "both" else [args.mode]
    rows = []
    print("mode  L(chunk)    N      CUDA_us  Triton_us  Triton/CUDA  CUDA_logical_GB/s")
    for chunk in args.chunks:
        n = args.topk + chunk
        ids = torch.empty((n,), dtype=dtype, device="cuda")
        if args.topk:
            ids[:args.topk] = torch.randint(args.pool - chunk, (args.topk,), device="cuda")
        if chunk:
            ids[args.topk:] = torch.arange(args.pool - chunk, args.pool, device="cuda")
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
            logical_bytes = n * (cache.shape[-1] + 2 * (args.nope + args.rope) + ids.element_size())
            row = dict(mode=mode, chunk=chunk, tokens=n, pool=args.pool,
                       nope=args.nope, rope=args.rope, vec=args.vec, warps=args.warps,
                       cuda_us=cuda_us, triton_us=triton_us,
                       speedup=triton_us / cuda_us,
                       cuda_logical_GBs=logical_bytes / (cuda_us * 1000),
                       triton_logical_GBs=logical_bytes / (triton_us * 1000))
            rows.append(row)
            print(f"{mode:5} {chunk:8d} {n:8d} {cuda_us:10.3f} {triton_us:10.3f} "
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
