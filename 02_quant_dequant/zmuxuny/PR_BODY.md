实现题目 2 的 MXFP8/NVFP4 软件量化、真实位宽打包及反量化，普通 CUDA GPU 即可运行。支持 FP32/FP16 文件输入、block/per-tensor scaling、nearest/stochastic 舍入，以及 FP16/BF16/FP32 输出和从 packed 文件单独重载。

本 PR 仅新增 `02_quant_dequant/zmuxuny/`，可独立构建、验证和重建报告。

验证：RTX 3060 Laptop 上通过 151 组独立 NumPy 码本测试和 3 组无效文件检查。1024×1024 FP32 输入的含缩放 payload 压缩率分别约 3.88× / 7.11×。报告附三类分布的误差和三次独立运行的性能数据；默认构建 sm_75 + PTX，不依赖原生 FP8/FP4 指令。

Nsight Compute 旧版与本机 WSL 不兼容，新版 Sanitizer 被 Windows 调试接口设置阻挡，原始日志已保留，未将这些检查计作通过。

[运行说明](https://github.com/zmuxuny/Learning-CUDA/blob/projects/quant-dequant/02_quant_dequant/zmuxuny/README.md) · [实测报告](https://github.com/zmuxuny/Learning-CUDA/blob/projects/quant-dequant/02_quant_dequant/zmuxuny/REPORT.md)
