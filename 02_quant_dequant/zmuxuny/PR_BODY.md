训练营 ID：**曹泽阳**；GitHub ID：`zmuxuny`；提交目录：`02_quant_dequant/zmuxuny/`。

实现 MXFP8 / NVFP4 软件量化、真实位宽打包、权重持久化及独立反量化。FP32/FP16 输入可恢复为 FP32/FP16/BF16，支持块级/张量级缩放和最近偶数/随机舍入。默认 CUDA 程序不要求原生 FP8/FP4 硬件，目录内可独立构建和测试。

### 实现

MXFP8 使用 E4M3 元素与 E8M0 块缩放；NVFP4 使用 E2M1 元素、E4M3 局部缩放与 FP32 全局缩放，每字节保存两个元素。文件自包含形状、格式、packed data 和 scales，可脱离原输入及配置加载。

对齐路径融合局部 amax、缩放与编码，使用向量读写；尾块及自定义布局保留通用路径。NVFP4 的全局归约计入完整量化耗时。提供 NVIDIA CUDA、MACA（C500）、CoreX（MR-V100）、MUSA（S4000）和 Ascend C（910B2）后端；Ada 原生 E4M3 转换是独立可选对照。

### 正确性与结果

- 当前计算实现在 Ascend 910B2 与 RTX 3060 各通过 **182 组数值测试 + 3 组非法文件检查**。独立 NumPy 码本验证编码、scales、重建值与文件重载，覆盖中点、相邻 ULP、尾块、次正规数和随机舍入。
- 1024×1024 FP32 正态输入、默认块缩放、nearest：MXFP8 / NVFP4 的 MSE 为 **0.000706241 / 0.00904878**，含 scales 的 payload 压缩率为 **3.8788× / 7.1111×**。均匀、正态、异常值的 max error、MAE、MSE 和完整文件压缩率见 [总结报告](https://github.com/zmuxuny/Learning-CUDA/blob/projects/quant-dequant/02_quant_dequant/zmuxuny/REPORT.md)。
- RTX 4090 D 的上述配置中，量化加反量化的含传输时间为 **0.803 / 0.814 ms**，单线程 CPU 参考为 **18.273 / 58.411 ms**；不含文件 I/O、分配与 packed 文件另行下载。
- 4090 D 的 128 MiB FP16/NVFP4 完整量化从基线 `1f259ef` 的 **0.75206 ms** 降至 **0.33219 ms（2.26×）**。三次独立同机交替试验取中位数，设备事件计时包含 amax、排除传输。完整尺寸及未获收益的配置在平台报告中保留。

### 平台证据与复现

| 内容 | 材料 |
|---|---|
| 构建、最小运行示例、配置与文件协议 | [README](https://github.com/zmuxuny/Learning-CUDA/blob/projects/quant-dequant/02_quant_dequant/zmuxuny/README.md) |
| 功能、误差、压缩率、CPU 对照和调优分析 | [总结报告](https://github.com/zmuxuny/Learning-CUDA/blob/projects/quant-dequant/02_quant_dequant/zmuxuny/REPORT.md) |
| 各平台实验 | [NVIDIA](https://github.com/zmuxuny/Learning-CUDA/blob/projects/quant-dequant/02_quant_dequant/zmuxuny/REPORT_TUNING.md)、[C500](https://github.com/zmuxuny/Learning-CUDA/blob/projects/quant-dequant/02_quant_dequant/zmuxuny/REPORT_C500.md)、[MR-V100](https://github.com/zmuxuny/Learning-CUDA/blob/projects/quant-dequant/02_quant_dequant/zmuxuny/REPORT_ILUVATAR.md)、[S4000](https://github.com/zmuxuny/Learning-CUDA/blob/projects/quant-dequant/02_quant_dequant/zmuxuny/REPORT_MUSA.md)、[Ascend](https://github.com/zmuxuny/Learning-CUDA/blob/projects/quant-dequant/02_quant_dequant/zmuxuny/REPORT_ASCEND.md) |
| Profiler 与 Sanitizer 范围、命令及日志 | [工具分析](https://github.com/zmuxuny/Learning-CUDA/blob/projects/quant-dequant/02_quant_dequant/zmuxuny/PROFILING.md) |

NVIDIA 有 Nsight Systems 时间线，软件及原生两路共 48 次 Sanitizer 检查通过；MR-V100 有时间线、硬件计数器与 12 次检查；Ascend 有 msprof 与 4 次基础 memcheck。C500 有时间线，S4000 的 MUPTI 时间戳无效；工具限制和失败记录按平台保留。

各平台性能与工具结果对应报告中的实验版本。当前计算实现完整回归设备为 Ascend 910B2 和 RTX 3060，其余设备保留原版本证据。原始 JSON、逐次计时、源码及二进制指纹随提交提供；本 PR 仅包含题目 2 目录，不依赖题目 3。
