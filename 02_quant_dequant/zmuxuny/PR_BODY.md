训练营 ID：**曹泽阳**；提交目录：`02_quant_dequant/zmuxuny/`。

实现题目 2 的 MXFP8 / NVFP4 软件量化、权重持久化与独立反量化，支持 NVIDIA CUDA、沐曦 MACA、天数 CoreX、摩尔线程 MUSA 和昇腾 CANN。输入支持 FP32/FP16，输出支持 FP32/FP16/BF16，提供块级/张量级缩放、最近偶数舍入与按种子可复现的随机舍入。本目录可独立构建和测试，不依赖题目 3。

### 实现与平台

- 软件实现 E4M3/E2M1 编码、E8M0/E4M3 缩放和 FP4 打包；保存的文件包含形状、格式、缩放与 packed data，可在没有原始输入和配置的情况下反量化。
- 默认块长且整块对齐时融合块最大值、缩放与量化，按平台选择向量宽度、线程数和 amax 网格；奇数列、尾块和自定义块长使用通用路径。NVFP4 全局归约包含在完整量化计时中。
- MUSA 使用精确整数乘积处理 FP32 次正规数下溢；NVFP4 快速除法经 FMA 残差及严格舍入区间检验，不满足条件时退回 SDK 精确除法。shuffle 后尾块缩放的编译器问题采用局部兼容处理，并说明触发条件。
- Ascend C 内核在 UB 中处理软件码本，按最多 1024 个输入批量搬运 packed data 和 scales；全局 amax 使用向量归约，对全次正规数 FP32 tile 保留标量处理，保持数值语义。
- BF16 转换保留 RNE 语义。CoreX 的主机/设备函数差异及 lane 哈希优化问题有局部兼容处理和专门回归；平台差异在代码旁说明依据。

| 平台 | 构建与实现 | 完整报告 |
|---|---|---|
| NVIDIA RTX 3060 / 4090 D | `make ARCH=86`（4090 D 用 `ARCH=89`）；默认软件编码兼容 sm_75，另有独立 Ada 原生 FP8 对照 `make native ARCH=89` | [NVIDIA 报告](https://github.com/zmuxuny/Learning-CUDA/blob/projects/quant-dequant/02_quant_dequant/zmuxuny/REPORT_TUNING.md) |
| MetaX C500 64GB | `make PLATFORM=metax`；MACA cu-bridge、64 线程 wave、平台专属量化/归约配置 | [C500 报告](https://github.com/zmuxuny/Learning-CUDA/blob/projects/quant-dequant/02_quant_dequant/zmuxuny/REPORT_C500.md) |
| Iluvatar 智铠 100 / MR-V100 32GB | `make PLATFORM=iluvatar`；CoreX 4.4 clang / ivcore11、逻辑缩放组、按 dtype/格式调优访存与归约 | [MR-V100 报告](https://github.com/zmuxuny/Learning-CUDA/blob/projects/quant-dequant/02_quant_dequant/zmuxuny/REPORT_ILUVATAR.md) |
| Moore Threads MTT S4000 48GB | `make PLATFORM=musa`；MUSA 4.3.6 / mp_22；按规模选择向量、线程及归约网格，保留逐位舍入语义 | [S4000 报告](https://github.com/zmuxuny/Learning-CUDA/blob/projects/quant-dequant/02_quant_dequant/zmuxuny/REPORT_MUSA.md) |
| Ascend 910B2 64GB | `make PLATFORM=ascend`；ARM64 / CANN 9.0；Ascend C 分块 DMA、向量 amax、UB 软件编码及 ACL 事件计时 | [昇腾报告](https://github.com/zmuxuny/Learning-CUDA/blob/projects/quant-dequant/02_quant_dequant/zmuxuny/REPORT_ASCEND.md) |

### 验证

- 独立 NumPy 码本参考覆盖两种格式、输入/输出 dtype、缩放/舍入、码本中点及相邻 ULP、极大/极小值、负零、奇数列、尾块和 65536 元素分派边界；主程序另做 CPU/GPU 逐字节比较。
- Ascend 910B2 与 RTX 3060 本轮完整回归各 **182 组数值测试 + 3 组非法文件检查通过**，各通过 **49,152 项哈希、65,536 项乘法和 1,048,576 项除法的主机/设备逐位比较**。S4000、MR-V100、C500 归档各为 182 + 3 组通过；MR-V100 另有 49,152 项哈希比较。4090 D 软件与原生对照各 158 + 3 组通过，两路 packed 文件一致。
- MR-V100：**12 次 ixsan memcheck/racecheck/initcheck 全部通过**，附 4 份 ixsys 时间线及 10 组 ixkn 硬件计数器。4090 D：软件与原生两路合计 **48 次 Compute Sanitizer 检查通过**，附 Nsight Systems、SASS 和资源报告。C500 附 mcTracer 原始数据；其镜像未提供可用设备端 Sanitizer。NVIDIA NCU 的宿主机计数器权限限制保留原始记录。

- Ascend：提供 4 组 msprof 任务时间与 PipeUtilization 硬件计数器，以及 **4 组基础 memcheck 全部通过**的实际执行日志。当前镜像的完整源码插桩构建/运行未成功；racecheck、initcheck、synccheck 要求该插桩，未计作通过。
- S4000：提供 MUPTI 采样程序、4 组原始活动记录及状态校验。该镜像返回的 kernel 起止时间均为零，明确标记 `UNAVAILABLE_TIMESTAMPS`；性能比较使用未插桩的 MUSA event，未取得硬件计数器或设备端 Sanitizer 结果。

各平台验证记录关联其报告中的环境与二进制哈希。当前公共代码在 Ascend 910B2 与 RTX 3060 完成完整回归；其余平台的归档结果对应各自报告中的原始版本。

### 性能与复现

以下均为同一台设备上、128MiB 输入的完整量化对照。采用预热后的设备事件计时，三次独立交替试验取中位数，排除文件 I/O、CPU 参考和主机传输。

| 平台 / 输入格式 | 报告基线 μs | 当前实现 μs | 加速比 |
|---|---:|---:|---:|
| RTX 4090 D / FP16 NVFP4 | 752.06 | 332.19 | 2.26× |
| C500 / FP16 NVFP4 | 757.22 | 605.95 | 1.25× |
| MR-V100 / FP32 MXFP8 | 474.79 | 384.10 | 1.24× |
| MR-V100 / FP16 NVFP4 | 1087.42 | 817.70 | 1.33× |
| S4000 / FP32 NVFP4 | 18406.49 | 1835.51 | 10.03× |
| S4000 / FP16 NVFP4 | 36038.70 | 2933.58 | 12.28× |
| Ascend 910B2 / FP16 NVFP4 | 307117.59 | 102372.11 | 3.00× |

4090 D 的基线为 `1f259ef`；国产平台的基线为各自正确性已通过的初始移植。S4000 基线使用 SDK 精确除法，当前采用经舍入区间校验的快速路径；这组加速比是对初始正确移植的优化，不代表跨硬件比较。MR-V100 的 FP16/MXFP8 大尺寸基本持平，完整尺寸、分布、缩放配置和未获收益的候选均在报告中。反量化统一输出 FP32，FP16 的 128MiB 输入组对应 256MiB 输出。

昇腾反量化选择直接 GM 读取并批量写回；显式 UB 预取候选在本轮更慢，测量及最终选择在报告中保留。

保留逐次计时、二进制/输出 SHA256、参数扫描、原始分析日志、源码校验清单及 MR-V100 / S4000 / Ascend 基线还原补丁。生成数据默认在 GitHub diff 中折叠，代码与报告可优先 review。

[构建与使用](https://github.com/zmuxuny/Learning-CUDA/blob/projects/quant-dequant/02_quant_dequant/zmuxuny/README.md) · [Profiler / Sanitizer 复现](https://github.com/zmuxuny/Learning-CUDA/blob/projects/quant-dequant/02_quant_dequant/zmuxuny/PROFILING.md)
