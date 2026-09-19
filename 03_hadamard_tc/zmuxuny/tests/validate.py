import itertools
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
from reference import (
    assert_packed,
    cast,
    hadamard,
    quantize,
    read_packed,
    read_tensor,
    write_tensor,
)


def main():
    cases = list(
        itertools.product([1, 2], [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], [0, 1])
    )
    maxerr = {1: 0.0, 2: 0.0}
    with tempfile.TemporaryDirectory(prefix="had-test-") as tmp:
        t = Path(tmp)
        for i, (dtype, d, fmt) in enumerate(cases):
            rng = np.random.default_rng(i)
            x = rng.normal(size=(19, d)).astype(np.float32)
            x[0] = 0
            x[1] = 0
            x[1, 0] = 8
            signs = i % 2 == 0
            stochastic = i % 3 == 0
            norm = i % 4 != 0
            write_tensor(t / "input", x, dtype)
            x, _ = read_tensor(t / "input")
            command = [
                str(ROOT / "build/hadamard"),
                "--input",
                str(t / "input"),
                "--output",
                str(t / "output"),
                "--packed",
                str(t / "packed"),
                "--unfused_packed",
                str(t / "unfused"),
                "--log",
                str(t / "log"),
                "--format",
                ["mxfp8", "nvfp4"][fmt],
                "--rounding",
                "stochastic" if stochastic else "nearest",
                "--normalize",
                str(int(norm)),
                "--random_sign",
                str(int(signs)),
                "--repeats",
                "3",
                "--tc_output",
                str(t / "tc"),
            ]
            p = subprocess.run(command, capture_output=True, text=True)
            assert p.returncode == 0, (dtype, d, fmt, p.stdout, p.stderr)
            y, _ = read_tensor(t / "output")
            ref = cast(hadamard(x, norm, signs), dtype)
            e = float(np.abs(y - ref).max())
            maxerr[dtype] = max(maxerr[dtype], e)
            assert e < (1e-2 if dtype == 1 else 5e-2), (dtype, d, e)
            expected = quantize(y, fmt, 32 if fmt == 0 else 16, stochastic=stochastic)
            assert_packed(read_packed(t / "packed"), expected)
            assert (t / "packed").read_bytes() == (t / "unfused").read_bytes()
            if dtype == 1 and d >= 16:
                tc, _ = read_tensor(t / "tc")
                assert np.max(np.abs(tc - ref)) < 1e-2
        write_tensor(t / "input", np.ones((2, 63)), 1)
        p = subprocess.run(command, capture_output=True, text=True)
        assert p.returncode != 0 and "power of two" in p.stderr
    result = {
        "status": "PASS",
        "passed": len(cases),
        "invalid_shape_cases": 1,
        "max_abs_error_fp16": maxerr[1],
        "max_abs_error_bf16": maxerr[2],
        "fused_packed_equal": True,
        "oracle": "NumPy float64 dense Sylvester matrix",
    }
    (ROOT / "results/correctness.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
