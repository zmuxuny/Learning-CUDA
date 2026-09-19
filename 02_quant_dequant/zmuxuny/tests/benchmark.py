"""Serial benchmark driver; keeps individual trials and reports medians."""

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import tempfile
from generate import generate
from reference import write_tensor

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trials", type=int, default=3)
    p.add_argument("--repeats", type=int, default=100)
    a = p.parse_args()
    records = []
    with tempfile.TemporaryDirectory(prefix="quant-bench-") as tmp:
        t = Path(tmp)
        for distribution in ["uniform", "normal", "outliers"]:
            for dtype in [0, 1]:
                write_tensor(t / "input", generate(1024, 1024, distribution), dtype)
                for fmt in ["mxfp8", "nvfp4"]:
                    for mode in ["block", "tensor"]:
                        (t / "config").write_text(
                            f'format = "{fmt}"\nscale_mode = "{mode}"\noutput_type = "fp32"\n'
                        )
                        trials = []
                        for _ in range(a.trials):
                            subprocess.run(
                                [
                                    str(ROOT / "build/quantize"),
                                    "--input",
                                    str(t / "input"),
                                    "--config",
                                    str(t / "config"),
                                    "--packed",
                                    str(t / "packed"),
                                    "--output",
                                    str(t / "out"),
                                    "--log",
                                    str(t / "log"),
                                    "--repeats",
                                    str(a.repeats),
                                ],
                                check=True,
                                capture_output=True,
                            )
                            trials.append(json.loads((t / "log").read_text()))
                        record = {
                            "distribution": distribution,
                            "input_dtype": ["fp32", "fp16"][dtype],
                            "format": fmt,
                            "scale_mode": mode,
                            "trials": trials,
                        }
                        record["median"] = {
                            k: statistics.median(v[k] for v in trials)
                            for k in trials[0]
                            if isinstance(trials[0][k], (float, int))
                        }
                        records.append(record)
                        print(
                            distribution,
                            dtype,
                            fmt,
                            mode,
                            "quant_ms",
                            record["median"]["quant_ms"],
                            flush=True,
                        )
    (ROOT / "results/benchmark.json").write_text(json.dumps(records, indent=2) + "\n")


if __name__ == "__main__":
    main()
