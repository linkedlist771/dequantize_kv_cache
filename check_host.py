"""No CUDA/PyTorch needed. Address-model and numerical-format checks ONLY.

This does not execute/compile dequant_cuda.cu and cannot establish GPU correctness.
"""
import ast
from collections import Counter
import math
from pathlib import Path
import struct


def decode_e4m3(byte):
    sign = -1.0 if byte & 128 else 1.0
    exp = (byte >> 3) & 15
    frac = byte & 7
    if exp == 15 and frac == 7:
        return float("nan")
    return sign * (math.ldexp(frac / 8, -6) if exp == 0 else math.ldexp(1 + frac / 8, exp - 7))


def main():
    base = Path(__file__).parent
    # Parse all Python files, including the exact upstream snapshot.
    files = sorted(base.glob("*.py"))
    for file in files:
        ast.parse(file.read_text(), filename=str(file))

    # Verify the numerical justification for FP8 -> FP16 -> FP32.
    finite = 0
    for byte in range(256):
        value = decode_e4m3(byte)
        if math.isnan(value):
            continue
        via_half = struct.unpack("<e", struct.pack("<e", value))[0]
        assert struct.pack("<f", value) == struct.pack("<f", via_half)
        finite += 1

    layouts = 0
    for nope in (128, 256, 384, 512, 640, 1024, 4096):
        for rope in (0, 64):
            width = nope + nope // 128 * 4 + 2 * rope
            for padding in (0, 4, 8, 16):
                stride = width + padding
                for vec in (4, 8, 16):
                    if stride % vec:
                        continue
                    written = []
                    for lane in range(32):
                        for col in range(lane * vec, nope, 32 * vec):
                            assert col + vec <= nope
                            assert col // 128 == (col + vec - 1) // 128
                            scale_start = nope + col // 128 * 4
                            assert nope <= scale_start < nope + nope // 128 * 4
                            assert scale_start % 4 == 0
                            for slot in (0, 1, 17, 2**32 + 1):
                                assert (slot * stride + col) % vec == 0
                            assert (2 * col) % min(2 * vec, 16) == 0
                            written.extend(range(col, col + vec))
                        for col in range(lane * 2, rope, 64):
                            assert (nope + nope // 128 * 4 + col * 2) % 4 == 0
                            written.extend((nope + col, nope + col + 1))
                    assert Counter(written) == Counter(range(nope + rope))
                    layouts += 1
    schedules = 0
    for n in (0, 1, 3, 4, 5, 7, 8, 9, 31, 97, 4097):
        for warps in (4, 8):
            rows = [block * warps + warp
                    for block in range((n + warps - 1) // warps)
                    for warp in range(warps) if block * warps + warp < n]
            assert rows == list(range(n))
            schedules += 1
    print(f"PASS: AST parse ({len(files)} Python files)")
    print(f"PASS: {finite} finite E4M3 encodings are exactly representable in FP16")
    print(f"PASS: address model ({layouts} vector/layout cases, {schedules} row schedules)")
    print("NOT CHECKED: CUDA compilation, GPU memory safety, numerical GPU parity, performance")


if __name__ == "__main__":
    main()
