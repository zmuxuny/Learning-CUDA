"""Regenerate the second-round report from archived, interleaved measurements."""

import json
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HADAMARD = ROOT.parent.name == "03_hadamard_tc"
DATA = ROOT / "results/4090d/tuning"


def main():
    result = json.loads((DATA / "comparison.json").read_text())
    records = result["records"]
    baseline = "e7c2386" if HADAMARD else "1f259ef"
    text = [
        "# RTX 4090 D 第二轮优化\n",
        f"实测日期：2026-09-19。本轮基线是第一轮已经提交的版本 `{baseline}`，不是最初版本。RTX 4090 D 24 GiB、CUDA 12.8、GCC 13、`sm_89`；编译保留 `--fmad=false`。默认程序继续使用软件 FP8/FP4 编码，可选 `_native` 程序使用 Ada 原生 E4M3 最近偶数转换。两者分别测试、分别计时。\n",
        "## 测量与正确性\n",
        "每组使用同一份 seed=42 正态输入，旧版、当前软件版、原生转换对照轮换执行顺序，各运行 3 个独立进程，报告中位数；每进程预热 3 次，小中张量重复 100 次，大张量重复 30 次。CUDA event 时间包括对应流程的全部 GPU 步骤，NVFP4 包括清零与全局最大值归约，不包括 CPU 参考、I/O、分配与传输。GPU 未锁频。大张量输入为 128 MiB，超过 72 MiB L2；小张量的结果可能受缓存影响。\n",
        (
            "[完整原始试验](results/4090d/tuning/comparison.json) 保留各次测量、可执行文件 SHA256 和 packed 文件 SHA256；每个配置的所有程序、所有重复均检查 packed 文件一致。题目 2 本轮统一使用 FP32 反量化输出；第一轮大张量报告沿用预设，MXFP8 输出 FP16、NVFP4 输出 FP32，所以不能跨两份报告直接相除反量化耗时。\n"
            if not HADAMARD
            else "[完整原始试验](results/4090d/tuning/comparison.json) 保留各次测量、可执行文件 SHA256，以及蝶形和分解 Tensor Core 的 packed 文件 SHA256；每个配置的所有程序、所有重复均检查对应路径的 packed 文件一致。\n"
        ),
    ]
    for filename, label in [
        ("correctness.json", "当前软件版"),
        ("correctness_native.json", "原生 FP8 对照"),
        ("compatibility_3060.json", "RTX 3060 / CUDA 11.5 兼容版"),
    ]:
        c = json.loads((DATA / filename).read_text())
        text.append(
            f"- {label}：{c['passed']} 组正确性测试通过；详见 [{filename}](results/4090d/tuning/{filename})。"
        )
    if HADAMARD:
        text.append(
            "- 每个完整测试集包含 60 组分解 MMA 检查，覆盖 FP16/BF16、符号随机化、归一化、随机舍入与不满 CTA 的尾行；另检查 NVFP4 中间结果保留方案。BF16 转换额外比较 326,400 个有限 FP32 位模式，标量和双元素转换均逐位一致。\n"
        )
    else:
        text.append(
            "- 每个测试集另含 3 组无效文件检查；边界覆盖所有 FP8 中点及相邻 ULP、所有正 E4M3 scale 下的 FP4 中点及相邻 ULP、奇数列、自定义块长、超过 65535 行、极大/极小输入和随机舍入。\n"
        )
    text += [
        "\n软件 E4M3 最近偶数舍入在正规数区间使用 `bits + 0x7ffff + retained_lsb` 后移位；尾数进位自然进入指数。次正规区间以精确二次幂缩放后做整数最近偶数转换，符号与负零单独保留。随机舍入仍按原概率与元素索引计算。该变更减少了量化和融合输出的标量指令，已由独立 NumPy 码本验证。\n"
    ]
    if not HADAMARD:
        text += [
            "\n软件版另测均匀分布、正态分布和异常值输入，覆盖 FP32/FP16、两种格式及 block/tensor 缩放，共 24 组配置；[完整数据](results/4090d/tuning/benchmark_modes.json) 保留每次耗时、CPU/GPU 含传输时间和误差指标。所有误差指标与第一轮相同。\n"
        ]
    text += ["\n## 完整流程性能\n"]
    if HADAMARD:
        text += [
            "表内均为分解 Tensor Core 路径，单位 μs。融合耗时包括 NVFP4 的完整 amax 预遍历；默认文件输出仍为蝶形路径，需用 `--factorized_tc_packed` 导出分解路径。\n",
            "| 行数×维度 / dtype | 格式 | 单独变换：旧 / 新 | 融合：旧 / 软件 / 原生 FP8 | 软件融合加速 | 原生融合加速 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for r in records:
            m = r["median"]
            b, s, n = [m[k] for k in ["before", "after", "native"]]
            if r["rows"] == 32:
                continue
            text.append(
                f"| {r['rows']}×{r['cols']} / {r['dtype']} | {r['format']} | {b['factorized_tc_ms']*1000:.2f} / {s['factorized_tc_ms']*1000:.2f} | {b['factorized_tc_fused_ms']*1000:.2f} / {s['factorized_tc_fused_ms']*1000:.2f} / {n['factorized_tc_fused_ms']*1000:.2f} | {b['factorized_tc_fused_ms']/s['factorized_tc_fused_ms']:.2f}× | {b['factorized_tc_fused_ms']/n['factorized_tc_fused_ms']:.2f}× |"
            )
        text += [
            "\n## 设计与消融\n",
            "BF16 在 Ampere 及更新架构使用原生 RN 转换，输出相邻元素用 `cvt.rn.bf16x2.f32` 打包；CPU 与 sm_75 保留整数位运算实现。MMA 仍采用 H16 小块乘法和 FP32 蝶形分解。普通量化、非融合对照同步使用题目 2 的向量化量化和 amax 归约，因此融合加速比的分母也随之变快。融合并非每种形状都优于非融合，原始 JSON 同时提供 `factorized_tc_unfused_ms` 与 `factorized_tc_fused_ms`。\n",
            "[线程块扫描](results/4090d/tuning/tune_mma.csv) 比较 1/2/4/8/16 个 warp。多数尺寸原来的 4 warp 已较好，大尺寸增加 warp 并没有稳定的统一收益，因此默认保持 4 warp。该扫描在软件编码下完成，比较变换输出和 packed 字节，不将噪声级差异用于复杂形状调度。\n",
            "`--materialized_compare 1` 是 NVFP4 的对照路径：第一次变换同时保存中间张量和 amax，再向量量化，避免第二次 Hadamard，但增加中间写回/读取。每次执行都与同一 MMA 的非融合结果比较 packed 字节；原始 JSON 的 `factorized_tc_materialized_ms` 保存测量。该方案在本次扫描中没有稳定的整体优势，保持为显式对照选项。\n",
        ]
    else:
        text += [
            "单位 μs，‘量化’为完整 GPU 量化流程；反量化统一输出 FP32。\n",
            "| 输入 / dtype | 格式 | 量化：旧 / 软件 / 原生 FP8 | 软件量化加速 | 反量化：旧 / 软件 | 反量化加速 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for r in records:
            m = r["median"]
            b, s, n = [m[k] for k in ["before", "after", "native"]]
            text.append(
                f"| {r['rows']}×{r['cols']} / {r['dtype']} | {r['format']} | {b['quant_ms']*1000:.2f} / {s['quant_ms']*1000:.2f} / {n['quant_ms']*1000:.2f} | {b['quant_ms']/s['quant_ms']:.2f}× | {b['dequant_ms']*1000:.2f} / {s['dequant_ms']*1000:.2f} | {b['dequant_ms']/s['dequant_ms']:.2f}× |"
            )
        text += [
            "\n## 设计与消融\n",
            "默认块对齐输入改为每线程处理 4 个元素、128 线程/CTA。FP32 使用 float4 读取，FP16 使用打包向量读取；在更小的 warp 子组中计算块 amax，打包后连续写出。非整块、奇数列和自定义块长仍使用已验证的通用路径。随机数继续按元素线性索引生成，线程映射变化不会改变随机舍入结果。\n",
            "反量化 FP32 输出采用每线程 4 元素、256 线程/CTA；FP16/BF16 输出采用每线程 8 元素、128 线程/CTA。全局 amax 使用每线程 8 元素的向量加载、256 线程/CTA，最多 512 个 CTA，最后每 CTA 只做一次原子最大值更新；不满向量的尾部单独读取。\n",
            "[量化扫描](results/4090d/tuning/tune_vector.csv)、[反量化扫描](results/4090d/tuning/tune_dequant.csv)、[amax 扫描](results/4090d/tuning/tune_max.csv) 保留不同向量宽度、线程块和归约网格的 3 次测量。量化扫描的 kernel 时间不包含 NVFP4 全局归约；正文表格包含它，二者加速比不能混用。\n",
            "[复制带宽对照](results/4090d/tuning/copy_ceiling.csv) 在 128 MiB 输入上测得 SM 向量复制约 924 GB/s、cudaMemcpy D2D 约 945 GB/s，按读+写逻辑字节计算。FP32 MXFP8 大张量量化已接近该逻辑带宽数量级，进一步减少编码计算的收益较小；这不是硬件 DRAM 利用率或绝对性能上限。只有 64 MiB 的 FP16 amax 微基准输入可放进 L2，不能将其加速直接外推到 128 MiB 完整流程。\n",
        ]
    text += [
        "\n## 可选原生 FP8 对照\n",
        "默认 `make` 仍构建软件 E4M3/E2M1 编码。额外执行 `make native ARCH=89` 可生成独立的 `_native` 程序，要求 CUDA >=12.1 和 sm_89 或更新；两路可以同时存在。原生对照使用 `cvt.rn.satfinite.e4m3x2.f32` 一次转换两个相邻值，低/高字节顺序与 packed 格式一致。随机舍入沿用软件编码，NVFP4 的 E2M1 数据仍由软件编码，只有 E4M3 scale 可用原生转换。日志 `fp8_encoding` 明确区分两路。\n",
        "指令与架构依据：[NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cvt)。这个对照不替代题目 2 的软件实现，也不是 FP8/FP4 Tensor Core 矩阵乘法。\n",
        "## Profiler 与 Sanitizer\n",
    ]
    for dirname, label in [("software", "软件版"), ("native", "原生对照")]:
        status = json.loads((DATA / dirname / "profiling.json").read_text())
        checks = [
            v
            for k, v in status.items()
            if k.startswith(("memcheck", "racecheck", "synccheck"))
        ]
        assert all(x.get("returncode") == 0 for x in checks)
        text.append(
            f"- {label}：memcheck、racecheck、synccheck 共 {len(checks)} 次全部通过；[原始日志与退出码](results/4090d/tuning/{dirname}/profiling.json)。"
        )
    text += [
        "\n两种格式、两路实现均采集 Nsight Systems 时间线，见各 `software/`、`native/` 下的 `timeline_*.nsys-rep` 和 `nsys_stats_*.csv`。分析使用无插桩 event 时间；时间线用于确认 kernel 与启动次数。Nsight Compute 仍因宿主机 `ERR_NVGPUCTRPERM` 无法访问计数器，未把静态资源或逻辑带宽冒充硬件 occupancy、DRAM 或 stall 测量。\n",
        "![本轮对比](results/4090d/tuning/comparison.png)\n",
        "## 复现\n",
        "```bash\nmake clean\nmake all native ARCH=89 NVCC=/usr/local/cuda/bin/nvcc HOSTCXX=g++\nmkdir -p results/4090d/tuning\npython3 tests/validate.py --output results/4090d/tuning/correctness.json\n"
        + f"python3 tests/validate.py --binary build/{'hadamard' if HADAMARD else 'quantize'}_native --output results/4090d/tuning/correctness_native.json\n"
        + f"python3 tests/benchmark_tuning.py --before /path/to/first-optimized/{'hadamard' if HADAMARD else 'quantize'} --after build/{'hadamard' if HADAMARD else 'quantize'} --native build/{'hadamard' if HADAMARD else 'quantize'}_native --output results/4090d/tuning/comparison.json\n"
        + "python3 tests/profile_native.py --output results/4090d/tuning/software\n"
        + f"python3 tests/profile_native.py --binary build/{'hadamard' if HADAMARD else 'quantize'}_native --output results/4090d/tuning/native\n"
        + "python3 tests/report_tuning.py\n```\n",
        "微基准用 `nvcc -std=c++17 -O3 --fmad=false -lineinfo -arch=sm_89` 编译 `tests/tune_*.cu` 后直接运行即可输出 CSV；题目 3 另加 `-DLP_CUDA_ARCH=89`。BF16 位模式检查位于题目 3 的 `tests/check_bf16.cu`。测试和采样顺序执行，不与其他 GPU 负载并发。第一轮完整结果保留在 [REPORT_4090D.md](REPORT_4090D.md)，本轮环境与源代码/二进制指纹见 [environment.json](results/4090d/tuning/environment.json)。\n",
    ]
    (ROOT / "REPORT_TUNING.md").write_text("\n".join(text))
    chosen = [
        r for r in records if r["rows"] != 32 and (not HADAMARD or r["cols"] == 1024)
    ]
    keys = (
        ["factorized_tc_ms", "factorized_tc_fused_ms"]
        if HADAMARD
        else ["quant_ms", "dequant_ms"]
    )
    titles = (
        ["Hadamard transform", "Hadamard + quantization"]
        if HADAMARD
        else ["Full quantization", "Dequantization (FP32 output)"]
    )
    fig, axs = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
    labels = [f"{r['rows']}x{r['cols']}\n{r['dtype']} / {r['format']}" for r in chosen]
    for ax, key, title in zip(axs, keys, titles):
        x = np.arange(len(chosen))
        w = 0.25
        for i, (name, color) in enumerate(
            [("before", "#9da6b0"), ("after", "#296fa1"), ("native", "#368567")]
        ):
            ax.bar(
                x + (i - 1) * w,
                [r["median"][name][key] * 1000 for r in chosen],
                w,
                label={
                    "before": "Previous submission",
                    "after": "Software optimized",
                    "native": "Native FP8 comparison",
                }[name],
                color=color,
            )
        ax.set_yscale("log")
        ax.set_ylabel("Latency (us, log scale)")
        ax.set_title(title, loc="left")
        ax.set_xticks(x, labels, fontsize=8)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axs[0].set_ylim(top=axs[0].get_ylim()[1] * 2)
    axs[0].legend(frameon=False, ncol=3, fontsize=9)
    fig.savefig(DATA / "comparison.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
