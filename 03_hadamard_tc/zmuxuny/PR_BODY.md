训练营 ID：**曹泽阳**；GitHub ID：`zmuxuny`；提交目录：`03_hadamard_tc/zmuxuny/`。

实现 FP16/BF16 Hadamard 变换及 MXFP8/NVFP4 量化融合，支持头维度 1–1024 的 2 的幂、归一化及固定随机符号。提供蝶形与原生矩阵指令路径，并比较融合和分步执行的完整成本。本目录随附量化与文件模块，可独立构建和测试。

### 实现

蝶形按 `(a,b) → (a+b,a-b)` 分 log₂D 级处理，使用 FP32 中间量。分解矩阵路径通过原生 FP16/BF16 指令计算 H16，再由蝶形完成其余因子，维持 O(D log D)；另保留 FP16 稠密 WMMA 对照。

融合量化保留中间 FP16/BF16 舍入，与自身非融合路径逐字节一致。MXFP8 可单遍融合；NVFP4 比较“重算变换求全局 amax”“写回时同时求 amax”“独立变换再量化”三种流程，计时均包含全部必要步骤。

提供 NVIDIA CUDA、MACA（C500）、CoreX（MR-V100）、MUSA（S4000）及 Ascend C（910B2）后端。各平台独立处理矩阵 fragment 与线程布局；Ascend 使用原生 Cube Mmad/Fixpipe 和 AIV 后处理。默认文件输出为蝶形结果，矩阵路径用独立参数导出。

### 正确性与性能

当前计算实现的 Ascend 910B2 与 RTX 3060 回归各通过 **92 组测试 + 1 组非法尺寸检查**，其中 76 组覆盖分解矩阵。独立 float64 Sylvester 参考测得 FP16 / BF16 最大绝对误差 **0.0078125 / 0.015625**，分别低于题目 0.01 / 0.05；融合的 packed data、scales、global scale 与同算法非融合路径一致。

以下比较各平台同一归档版本、128 MiB FP16 独立变换，单位 ms。输入为 65536×1024 正态数据，归一化开启、随机符号关闭；预热后的完整设备事件计时，三次独立试验取中位数，排除传输。

| 平台 | 蝶形 | 分解矩阵 | 蝶形 / 矩阵 |
|---|---:|---:|---:|
| C500 | 0.47257 | 0.25575 | 1.85× |
| MR-V100 | 0.89576 | 0.48758 | 1.84× |
| S4000 | 0.99830 | 1.39601 | 0.72× |
| Ascend 910B2 | 120.42426 | 63.55028 | 1.89× |

矩阵与融合收益随平台、尺寸变化。S4000 此配置应选蝶形；MR-V100 的 8192×1024 FP16/NVFP4 矩阵非融合 / 重算融合 / 写回+amax 为 **0.18826 / 0.27475 / 0.15633 ms**。4090 D 的同尺寸 MXFP8 矩阵融合为 **0.01865 ms**。完整 NVIDIA 结果、CPU 含传输对照和其他配置见 [总结报告](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/REPORT.md)。

### 平台证据与复现

| 内容 | 材料 |
|---|---|
| 构建、输入形状、各路径运行和输出参数 | [README](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/README.md) |
| 算法、误差、矩阵/蝶形、融合及 CPU 对照 | [总结报告](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/REPORT.md) |
| 各平台实验 | [NVIDIA](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/REPORT_TUNING.md)、[C500](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/REPORT_C500.md)、[MR-V100](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/REPORT_ILUVATAR.md)、[S4000](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/REPORT_MUSA.md)、[Ascend](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/REPORT_ASCEND.md) |
| 指令证据、Profiler 与 Sanitizer 范围 | [工具分析](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/PROFILING.md) |

NVIDIA 有 Nsight Systems 与矩阵指令反汇编，软件及原生两路共 36 次 Sanitizer 检查通过；MR-V100 有矩阵硬件计数器及 12 次检查；Ascend 有 msprof 与 4 次基础 memcheck。其他平台工具可用范围和失败记录在工具分析中列明。

各平台性能与工具结果对应报告中的实验版本。当前计算实现完整回归设备为 Ascend 910B2 和 RTX 3060，其余设备保留原版本证据。原始 JSON、逐次计时、源码及二进制指纹随提交提供；本 PR 仅包含题目 3 目录，不依赖题目 2。
