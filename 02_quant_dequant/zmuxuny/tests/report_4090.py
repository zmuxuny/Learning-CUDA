"""Rebuild the same-device before/after report from archived measurements."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/4090d"
HAD = ROOT.parent.name == "03_hadamard_tc"


def read(path):
    return json.loads((DATA / path).read_text())


def main():
    before, after = read("before/benchmark.json"), read("after/benchmark.json")
    extended = read("extended.json")
    correctness = read("after/correctness.json")
    profiling = read("after/profiling.json")
    baseline = "0a5e40b" if HAD else "73b66de"
    lines = [
        "# RTX 4090 D 优化实验",
        "\n本页为第一轮已归档结果。最新实现与第二轮对比见 [REPORT_TUNING.md](REPORT_TUNING.md)。",
        "",
        f"实测日期：2026-09-19。基线为本题首版提交 `{baseline}`（与实际构建的本地集成快照 `deba5cb` 对应目录完全相同），优化前后均在同一台 RTX 4090 D 24 GiB 上执行。CUDA 12.8、驱动 570.124.06、GCC 13，编译目标 `sm_89`，使用 `-O3 --fmad=false -lineinfo`。环境和基线可执行文件 SHA256 见 [environment.json](results/4090d/environment.json)。",
        "",
        "## 测量方法",
        "",
        "常规基准每配置运行 3 个独立进程，每进程预热 3 次、CUDA event 重复 100 次，取时间中位数。大张量每配置同样 3 次，重复 30 次；优化前后进程交替运行以减少温度与时钟漂移的影响。GPU 未锁频。表内加速比统一按两个时间中位数之比计算。",
        "",
        "kernel 计时不含文件 I/O、CPU 参考、设备分配和传输；NVFP4 的计时包含全局 amax 清零、归约和输出阶段。输入驻留设备且重复访问，常规尺寸可能受 L2 缓存加速，因此另测单个输入即为 128 MiB 的大张量。有效 GB/s 是逻辑读写量除以时间，不作为 DRAM 利用率。",
        "",
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if not HAD:
        lines += [
            "## 实现变化",
            "",
            "- 默认 32/16 元素块将块 amax、scale 编码、元素量化与 packed 写出合并为一个 kernel，输入只读一遍。整块对齐的行使用连续一维 tile；非整块和奇数列保留按行索引，NVFP4 每个字节仅一个线程写入。",
            "- FP8 编码从 FP32 位域提取指数、尾数和舍入残数；FP8/FP4 解码直接构造浮点位域。NVFP4 最近舍入直接比较 scale 乘以精确 FP4 中点，避免再做一次除法，并穷举所有正 E4M3 scale 下的中点及相邻 ULP 验证；随机舍入保留原商与概率。MXFP8 使用精确二次幂倒数乘法，E8M0 最小 scale 的特殊情况仍用除法。随机舍入、最近偶数舍入和文件格式保持一致。",
            "- 全局 amax 先归约到每个 warp，再通过共享内存归约到整个 CTA，将原子更新从每 warp 一次降至每 CTA 一次。",
            "- 默认格式的反量化专门化索引，NVFP4 一次读取一个字节、输出两个值。自定义块长沿用通用 kernel。",
            "",
            "## 常规尺寸：1024×1024，标准正态",
            "",
            "| 输入 | 格式 | 量化前/后 μs | 量化加速 | 反量化前/后 μs | 反量化加速 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        pairs = [
            (b, a)
            for b, a in zip(before, after)
            if a["distribution"] == "normal" and a["scale_mode"] == "block"
        ]
        for b, a in pairs:
            x, y = b["median"], a["median"]
            lines.append(
                f"| {a['input_dtype']} | {a['format']} | {x['quant_ms']*1000:.3f} / {y['quant_ms']*1000:.3f} | {x['quant_ms']/y['quant_ms']:.2f}× | {x['dequant_ms']*1000:.3f} / {y['dequant_ms']*1000:.3f} | {x['dequant_ms']/y['dequant_ms']:.2f}× |"
            )
        for ax, key, title in zip(
            axes, ["quant_ms", "dequant_ms"], ["Quantize", "Dequantize"]
        ):
            x = np.arange(len(pairs))
            ax.bar(
                x - 0.18,
                [b["median"][key] * 1000 for b, a in pairs],
                0.36,
                label="Before",
                color="#94a3b8",
            )
            ax.bar(
                x + 0.18,
                [a["median"][key] * 1000 for b, a in pairs],
                0.36,
                label="After",
                color="#2563eb",
            )
            ax.set_xticks(x, [a["input_dtype"] + "\n" + a["format"] for b, a in pairs])
            ax.set_title(title)
        lines += [
            "",
            "所有 24 组常规基准的 MAE、MSE 和最大绝对误差与基线一致。完整均匀分布、正态、异常值，以及 per-tensor/per-block 和 FP16/FP32 数据见 `before/benchmark.json`、`after/benchmark.json`。",
            "",
            "## 大张量：输入 128 MiB",
            "\n本轮沿用格式预设：MXFP8 反量化输出 FP16，NVFP4 反量化输出 FP32；同一行的优化前后输出类型相同。第二轮改为统一 FP32 输出，反量化耗时不能跨报告直接比较。",
            "",
            "| 输入形状 / dtype | 格式 | 量化前/后 μs | 加速 | 反量化前/后 μs | 加速 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for r in extended:
            x, y = r["median"]["before"], r["median"]["after"]
            lines.append(
                f"| {r['rows']}×{r['cols']} / {r['dtype']} | {r['format']} | {x['quant_ms']*1000:.2f} / {y['quant_ms']*1000:.2f} | {x['quant_ms']/y['quant_ms']:.2f}× | {x['dequant_ms']*1000:.2f} / {y['dequant_ms']*1000:.2f} | {x['dequant_ms']/y['dequant_ms']:.2f}× |"
            )
        lines += [
            "",
            "## 正确性",
            "",
            f"独立 NumPy 码本 oracle 通过 **{correctness['passed']} 组测试及 3 组无效文件测试**。新增超过 65535 行的单列张量、全块极小 FP32 和大数，覆盖新网格索引及 E8M0 倒数边界。packed data、scales 和重建值逐项比较。",
        ]
    else:
        lines += [
            "## 算法与实现",
            "",
            "原始 WMMA 对照对整个 D×D Hadamard 矩阵做稠密乘法，复杂度为 O(D²)。新路径利用 `H_D = H_(D/16) ⊗ H_16`：把连续 16 个值作为一个 segment，两条 `mma.sync.m16n8k16` 完成 16 个 segment 的 H16 变换，其余维度在 FP32 寄存器内通过 shuffle 与蝶形加减合并。无需在全局内存构造完整 Hadamard 矩阵，整体仍为 O(D log D)。",
            "",
            "FP16 和 BF16 各使用原生对应类型的 MMA 指令，均 FP32 累加。D≤256 时一个 warp 同时覆盖多行；D>256 时一个 warp 覆盖一行。lane/寄存器布局依据 [NVIDIA PTX m16n8k16 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float)，实现位于 `src/tensor_core.cuh`。目标要求 `ARCH>=80`；默认 sm_75 构建保留蝶形和 FP16 稠密 WMMA。",
            "",
            "Tensor Core 融合直接从 MMA 的 FP32 寄存器计算块 scale 和 packed data，省去中间张量写回/重读。保留与输入 dtype 一致的中间舍入，独立比较“Tensor Core 变换→量化”与“Tensor Core 融合”的完整 packed 结果。蝶形和 MMA 的浮点求和顺序不同，各自与 float64 参考校验，而不要求两个算法的 packed 字节相同。",
            "",
            "NVFP4 保留变换后全局 amax 预遍历，表中时间包含这一遍、清零和第二次变换。预遍历采用 CTA 内分层归约减少原子更新；MXFP8 只需单 kernel。蝶形融合同时加入格式专门化与精确二次幂缩放。",
            "",
            "## 常规尺寸：8192 行，FP16",
            "",
            "| D | 格式 | 初版蝶形融合 μs | 新蝶形融合 μs | 新 Tensor Core 融合 μs | TC 相对初版 | TC 相对新蝶形 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for b, a in zip(before, after):
            if a["dtype"] != "fp16" or a["rows"] != 8192:
                continue
            x, y = b["median"], a["median"]
            lines.append(
                f"| {a['dim']} | {a['format']} | {x['fused_ms']*1000:.3f} | {y['fused_ms']*1000:.3f} | {y['factorized_tc_fused_ms']*1000:.3f} | {x['fused_ms']/y['factorized_tc_fused_ms']:.2f}× | {y['fused_ms']/y['factorized_tc_fused_ms']:.2f}× |"
            )
        for ax, fmt in zip(axes, ["mxfp8", "nvfp4"]):
            pairs = [
                (b, a)
                for b, a in zip(before, after)
                if a["dtype"] == "fp16" and a["rows"] == 8192 and a["format"] == fmt
            ]
            x = np.arange(len(pairs))
            for offset, key, label, color, which in [
                (-0.25, "fused_ms", "Before butterfly", "#94a3b8", 0),
                (0, "fused_ms", "After butterfly", "#60a5fa", 1),
                (0.25, "factorized_tc_fused_ms", "After Tensor Core", "#2563eb", 1),
            ]:
                ax.bar(
                    x + offset,
                    [pair[which]["median"][key] * 1000 for pair in pairs],
                    0.25,
                    label=label,
                    color=color,
                )
            ax.set_xticks(x, [a["dim"] for b, a in pairs])
            ax.set_xlabel("Head dimension")
            ax.set_title(fmt.upper() + " fused pipeline")
        lines += [
            "",
            "## 单独变换：8192 行",
            "",
            "| dtype | D | 蝶形 μs | 新 Tensor Core μs | TC 相对蝶形 | 原稠密 WMMA μs |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for a in after:
            if a["rows"] != 8192 or a["format"] != "mxfp8":
                continue
            m = a["median"]
            dense = f"{m['tensor_core_ms']*1000:.3f}" if "tensor_core_ms" in m else "—"
            lines.append(
                f"| {a['dtype']} | {a['dim']} | {m['hadamard_ms']*1000:.3f} | {m['factorized_tc_ms']*1000:.3f} | {m['hadamard_ms']/m['factorized_tc_ms']:.2f}× | {dense} |"
            )
        lines += [
            "",
            "新 Tensor Core 的主要收益出现在融合输出；单独变换不保证所有尺寸都优于蝶形。小尺寸启动开销占比较大。全部 BF16、32 行小批量、三次原始结果均保存在 JSON。",
            "",
            "## 大张量：65536×1024，输入 128 MiB",
            "",
            "| dtype | 格式 | 初版融合 μs | 新蝶形融合 μs | 新 TC 融合 μs | TC 相对初版 | 新 TC 非融合/融合 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in extended:
            x, y = r["median"]["before"], r["median"]["after"]
            lines.append(
                f"| {r['dtype']} | {r['format']} | {x['fused_ms']*1000:.2f} | {y['fused_ms']*1000:.2f} | {y['factorized_tc_fused_ms']*1000:.2f} | {x['fused_ms']/y['factorized_tc_fused_ms']:.2f}× | {y['factorized_tc_unfused_ms']/y['factorized_tc_fused_ms']:.2f}× |"
            )
        lines += [
            "",
            "## 正确性",
            "",
            f"蝶形路径通过 {correctness['passed']} 组测试，Tensor Core 分解及其融合另通过 {correctness['factorized_tc_cases']} 组独立 oracle 校验，包含 FP16/BF16、D=16…1024、非整 CTA 行数、随机符号、两种舍入和归一化选项。Tensor Core 最大绝对误差：FP16 **{correctness['factorized_tc_max_abs_error_fp16']:.9g}**，BF16 **{correctness['factorized_tc_max_abs_error_bf16']:.9g}**。两种后端各自融合/非融合 packed 逐字节相同。",
        ]
    for ax in axes:
        ax.set_ylabel("Latency (µs)")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(DATA / "comparison.png", dpi=180)
    plt.close(fig)
    lines += [
        "",
        "![同一 GPU 优化前后延迟](results/4090d/comparison.png)",
        "",
        "## Profiler / Sanitizer",
        "",
    ]
    assert all(
        profiling[k].get("returncode") == 0
        for k in ["nsys_mxfp8", "nsys_nvfp4", "nsys_stats_mxfp8", "nsys_stats_nvfp4"]
    )
    checks = [
        (k, v)
        for k, v in profiling.items()
        if k.startswith(("memcheck", "racecheck", "synccheck"))
    ]
    assert all(v.get("returncode") == 0 for k, v in checks), checks
    lines += [
        f"Compute Sanitizer 的 memcheck、racecheck、synccheck 共 **{len(checks)} 次检查全部通过**；原始命令及输出见 `results/4090d/after/*check_*.txt`，状态汇总见 [profiling.json](results/4090d/after/profiling.json)。",
        "",
        "Nsight Systems 成功采集 MXFP8/NVFP4 两组 CUDA 时间线，原始 `.nsys-rep` 可在 GUI 打开，kernel/API 汇总见 `nsys_stats_*.csv`。具体 kernel 耗时、调用次数及瓶颈分析见 [PROFILING.md](PROFILING.md)。",
        "",
        "Nsight Compute 返回 `ERR_NVGPUCTRPERM`，该容器所在宿主机限制硬件计数器访问；保留 `ncu.txt` 原始输出。本报告的性能结论使用无插桩 CUDA event 基准，时间线用于确认执行步骤，未把工具失败计作通过。",
        "",
        "## 复现",
        "",
        "```bash",
        "make clean",
        "make ARCH=89 NVCC=/usr/local/cuda/bin/nvcc HOSTCXX=g++",
        "python3 tests/validate.py",
        "python3 tests/benchmark.py --trials 3 --repeats 100",
        "python3 tests/profile_native.py",
        "# 分别从基线提交和当前提交构建两个可执行文件：",
        "python3 tests/benchmark_extended.py --before /path/to/baseline/"
        + ("hadamard" if HAD else "quantize")
        + " --after build/"
        + ("hadamard" if HAD else "quantize"),
        "# 归档 JSON 后重建本报告和图：",
        "python3 tests/report_4090.py",
        "```",
        "",
        "基线与新版本原始 JSON 分别保存在 `results/4090d/before/` 和 `after/`，大张量交替试验见 `extended.json`。原 RTX 3060 Laptop 数据保留在 `results/` 根目录和原报告中。",
    ]
    lines += [
        "",
        "## 旧 Toolkit 兼容验证",
        "",
        "另在本机 RTX 3060 Laptop / CUDA 11.5 / GCC 10.5 上通过相同正确性测试；题目 2 编译目标 sm_75，题目 3 编译目标 sm_86 并验证新 Tensor Core 路径。结果见 [compatibility_3060.json](results/compatibility_3060.json)。",
    ]
    (ROOT / "REPORT_4090D.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
