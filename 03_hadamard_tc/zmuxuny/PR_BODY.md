实现题目 3 的 FP16/BF16 Hadamard 变换、MXFP8/NVFP4 融合量化及 FP16 WMMA Tensor Core 对照。支持 1–1024 的 2 的幂头维度和随机符号旋转。融合保留中间 dtype 舍入，scales 与 packed data 和先变换后量化逐字节一致。

本 PR 仅新增 `03_hadamard_tc/zmuxuny/`，随附所需数值、文件与参考模块，可独立构建测试，不依赖题目 2 PR 合并。

验证：RTX 3060 Laptop 上通过 44 组独立 float64 稠密矩阵/量化参考测试及 1 组非法尺寸检查。FP16 最大绝对误差 0.0078125，BF16 为 0.0000610352。8192 行、D=64/128/256 的融合路径约有 1.6–2.1× 加速。稠密 WMMA 在大批量上慢于蝶形；报告保留该对照及 D=1024 的融合性能边界。

附实测报告、原始三次试验数据、旋转重建误差和 HMMA 指令证据。Profiler/Sanitizer 的 WSL/Windows 环境限制及失败日志单独记录，未声称通过。

[运行说明](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/README.md) · [实测报告](https://github.com/zmuxuny/Learning-CUDA/blob/projects/hadamard/03_hadamard_tc/zmuxuny/REPORT.md)
