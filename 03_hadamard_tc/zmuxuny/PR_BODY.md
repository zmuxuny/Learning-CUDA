训练营 ID：**曹泽阳**；提交目录：`03_hadamard_tc/zmuxuny/`。

实现题目 3 的 FP16/BF16 Hadamard 变换与 MXFP8/NVFP4 量化融合，支持 NVIDIA CUDA、沐曦 MACA、天数 CoreX、摩尔线程 MUSA 和昇腾 CANN。头维度支持 1–1024 的 2 的幂，提供归一化、固定随机符号旋转及可复现随机舍入。本目录随附数值编码、文件读写和独立参考模块，可独立构建测试，不依赖题目 2。

### 算法与平台

- CUDA 及兼容后端的蝶形路径在 FP32 寄存器和 XOR shuffle 中完成 O(D log D) 变换；逻辑线程组和每个 CTA 的行数按平台、维度及归约/输出阶段选择。
- 分解矩阵路径以原生 FP16/BF16 矩阵指令计算 H16，再用 FP32 蝶形完成剩余因子，保持 O(D log D)。CUDA 及兼容后端另保留 FP16 稠密矩阵对照；Ascend 提供 H16 分解矩阵路径。各平台的 fragment 坐标、参与线程数及尾行处理分别实现并注释，避免复用不兼容的布局。
- Ascend 使用原生 Cube Mmad/Fixpipe 和 AIV 向量 Add/Sub，H16 的 FP32 中间结果经 GM 交接。符号准备、Cube 计算、剩余蝶形和最终转换均计入矩阵路径；不依赖 PyTorch 或预装 MatMul 算子包。
- 融合量化保留中间 FP16/BF16 舍入，因此 packed data、scales、global scale 与该后端自己的非融合流程逐字节一致。NVFP4 同时提供重算式全融合与“变换+amax 写回，再量化”的两步方案。
- 默认保存蝶形输出，分解矩阵及两步方案通过独立参数保存。软件 FP8/FP4 编码不依赖原生低比特硬件；Ada 原生 FP8 转换是独立对照程序。

| 平台 | 构建与矩阵实现 | 完整报告 |
|---|---|---|
| NVIDIA RTX 3060 / 4090 D | `make ARCH=86`（4090 D 用 `ARCH=89`）；sm_80+ PTX MMA，FP32 寄存器蝶形；另有 FP16 WMMA 对照 | [NVIDIA 报告](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/REPORT_TUNING.md) |
| MetaX C500 64GB | `make PLATFORM=metax`；MACA 原生 64 线程矩阵 fragment，向量读取及 FP16/BF16 MMA | [C500 报告](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/REPORT_C500.md) |
| Iluvatar 智铠 100 / MR-V100 32GB | `make PLATFORM=iluvatar`；CoreX 4.4 / ivcore11 原生 64 线程 FP16/BF16 矩阵指令，直接构造寄存器 fragment | [MR-V100 报告](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/REPORT_ILUVATAR.md) |
| Moore Threads MTT S4000 48GB | `make PLATFORM=musa`；MUSA 4.3.6 / mp_22；128 线程原生 WMMA H16，FP32 蝶形分解，独立处理 fragment 布局与尾行屏障 | [S4000 报告](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/REPORT_MUSA.md) |
| Ascend 910B2 64GB | `make PLATFORM=ascend`；ARM64 / CANN 9.0；原生 FP16/BF16 Mmad / Fixpipe H16、64×16 Cube tile、AIV FP32 蝶形与融合量化 | [昇腾报告](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/REPORT_ASCEND.md) |

### 验证

- 独立 NumPy float64 Sylvester 矩阵及码本参考覆盖 dtype、维度、尾行、CTA 边界、归一化、随机符号与舍入。Ascend 910B2 和 RTX 3060 本轮各 **92 组测试 + 1 组非法尺寸检查通过**，其中 76 组覆盖分解矩阵路径。FP16/BF16 最大绝对误差分别为 **0.0078125 / 0.015625**，低于 0.01 / 0.05；融合及 NVFP4 两步方案均通过 packed 字节一致性检查。
- Ascend 910B2 与 RTX 3060 本轮各通过 **49,152 项哈希、65,536 项乘法和 1,048,576 项除法的主机/设备逐位比较**。MUSA 保留次正规数乘积，并用舍入区间校验快速除法；CoreX 的 lane 哈希优化问题采用局部兼容处理，CoreX 稠密对照使用补偿求和控制长串累加误差。
- S4000、MR-V100、C500 归档各为 92 组通过、76 组覆盖矩阵路径，另有非法尺寸检查；MR-V100 有 49,152 项哈希比较。4090 D 软件与原生对照各 76 组通过、60 组覆盖矩阵路径，保留 BF16 转换位模式测试及旋转重建误差实验。
- MR-V100：**12 次 ixsan memcheck/racecheck/initcheck 全部通过**，附 4 份 ixsys 时间线和 12 组 ixkn 硬件计数器。8192×1024 的独立/融合变换均记录到 **32,768 条矩阵指令**，与 H16 tile 数一致。4090 D 两路合计 **36 次 Compute Sanitizer 检查通过**，附 Nsight Systems、HMMA/SASS 和资源报告。C500 附 mcTracer 数据；其镜像未提供设备端 Sanitizer。NVIDIA NCU 的宿主机计数器权限限制保留原始记录。

- Ascend：提供 4 组 msprof 任务时间与 PipeUtilization 硬件计数器，以及 **4 组基础 memcheck 全部通过**的实际执行日志。当前镜像的完整源码插桩构建/运行未成功；racecheck、initcheck、synccheck 要求该插桩，未计作通过。
- S4000：提供 MUPTI 采样程序、4 组原始活动记录及状态校验。该镜像返回的 kernel 起止时间均为零，明确标记 `UNAVAILABLE_TIMESTAMPS`；性能比较使用未插桩的 MUSA event，未取得硬件计数器或设备端 Sanitizer 结果。

各平台验证记录关联其报告中的环境与二进制哈希。当前公共代码在 Ascend 910B2 与 RTX 3060 完成完整回归；其余平台的归档结果对应各自报告中的原始版本。

### 性能与路径选择

同机交替运行基线/当前程序，每项三次独立试验，采用预热后的设备事件计时中位数，校验输出 SHA256。计时包含完整 GPU 流程，NVFP4 包含必要初始化和全局 amax；排除文件 I/O、CPU 参考与主机传输。

| 平台 / 128MiB 独立变换 | 初始蝶形 μs | 当前蝶形 μs | 当前矩阵 μs | 矩阵 / 初始蝶形加速比 |
|---|---:|---:|---:|---:|
| C500 / FP16 | 706.06 | 472.57 | 255.75 | 2.76× |
| C500 / BF16 | 719.29 | 510.59 | 278.66 | 2.58× |
| MR-V100 / FP16 | 1749.15 | 895.76 | 487.58 | 3.59× |
| MR-V100 / BF16 | 1836.10 | 924.60 | 488.41 | 3.76× |
| S4000 / FP16 | 998.48 | 998.30 | 1396.01 | 0.72× |
| S4000 / BF16 | 1155.80 | 1155.31 | 1805.00 | 0.64× |
| Ascend 910B2 / FP16 | 235416.77 | 120424.26 | 63550.28 | 3.70× |
| Ascend 910B2 / BF16 | 250455.32 | 133556.43 | 74348.82 | 3.37× |

表中加速比是矩阵算法相对初始蝶形的比较，同一路径调优前后的数据另在原始 JSON 中完整保留。RTX 4090 D 的 8192×1024 FP16/MXFP8 融合为 **18.65μs**，原生 FP8 对照为 **15.87μs**；对应报告中 `e7c2386` 基线为 22.75μs。

S4000 的原生矩阵路径通过正确性验证，但大尺寸下蝶形路径更快；128MiB FP16/NVFP4 的蝶形非融合为 **3976.26μs**，蝶形全融合为 **16182.10μs**。全融合相对初始移植的 80257.97μs 已有改善，实测仍应选择先蝶形变换再量化。矩阵路径的共享内存搬运、同步与融合中的数值处理成本保留在完整流程计时内。

最优融合方式随平台和尺寸变化。例如 MR-V100 的 8192×1024 FP16/NVFP4：矩阵非融合 **188.26μs**、全融合 **274.75μs**、变换+amax 写回后量化 **156.33μs**。因此保留并比较全部路径；报告结合硬件计数器、指令与访存解释选择依据，不把更少的 kernel 数量直接等同于更快。

昇腾 128MiB FP16 的矩阵路径由 317.441ms 降至 63.550ms，同一路径约 5.00×。大尺寸 MXFP8 优选矩阵融合，NVFP4 优选矩阵非融合：FP16 的非融合 / 全融合 / 写回+amax 分别为 165.998 / 191.621 / 172.297ms。该基准开启归一化、关闭随机符号；随机符号另在独立参考和工具采样中覆盖。

保留所有试验、参数扫描、二进制/输出哈希、源码校验清单和 MR-V100 / S4000 / Ascend 基线还原补丁。生成数据默认在 GitHub diff 中折叠，便于优先审阅实现与报告。

[构建与使用](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/README.md) · [Profiler / Sanitizer 复现](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/PROFILING.md)
