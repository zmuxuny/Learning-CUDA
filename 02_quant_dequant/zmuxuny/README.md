# MXFP8 / NVFP4 软件量化与反量化

当前优化结果见 [RTX 4090 D 实验报告](REPORT_4090D.md)：包含同一 GPU 上的首版/优化版比较、128 MiB 输入实验、Nsight Systems 时间线和已通过的 Compute Sanitizer 检查。原 RTX 3060 Laptop 数据保留在 [首版报告](REPORT.md)。

题目 2，提交 ID：`zmuxuny`。纯 CUDA 软件实现，默认编译目标为 Turing `sm_75`，不使用硬件 FP8/FP4 转换指令或 Tensor Core。数值编码、缩放、打包、解包均由本项目实现。本目录可独立构建、测试和提交。

## 构建与运行

依赖 CUDA Toolkit >= 11.0、兼容的 C++17 host 编译器、Make；测试需要 Python 3 和 NumPy。CUDA 11.5 与本机 GCC 11 的标准库存在编译兼容问题，Makefile 优先选择已安装的 `g++-10`；其他机器可用 `HOSTCXX=/path/to/g++` 指定。

```bash
cd 02_quant_dequant/zmuxuny
make
mkdir -p results/tmp
python3 tests/generate.py --rows 1024 --cols 1024 --dtype fp32 \
  --distribution outliers --output results/tmp/input.bin
./build/quantize --input results/tmp/input.bin --config configs/nvfp4.toml \
  --packed results/tmp/weights.bin --output results/tmp/restored.bin \
  --log results/tmp/metrics.json --repeats 100
# 仅从保存的权重文件反量化，不需要原始输入或配置文件
./build/quantize --mode dequantize --packed results/tmp/weights.bin \
  --output results/tmp/reloaded.bin --output_type bf16 --log results/tmp/reload.json
python3 tests/validate.py
python3 tests/benchmark.py
```

`make test` 运行较短的参数组合测试；完整 `validate.py` 还覆盖所有 FP8 有限码值、FP4 码值、舍入中点及其两侧、零值、大动态范围、奇数列、尾块和文件错误。测试与正式程序分目录存放。

切换目标架构时先 `make clean`，再例如 `make ARCH=86`。默认 `sm_75` 二进制及 PTX 已在 RTX 3060 上运行；这证明编译路径不依赖 Ampere 指令，T4 的实际运行性能仍需在 T4 上测量。

## 配置语义

| 参数 | 支持值 / 含义 |
|---|---|
| `format` | `mxfp8` / `nvfp4` |
| `block_size` | 默认分别为 32 / 16；支持 16 的倍数，范围 16–1024 |
| `scale_mode` | `block` 为每行独立分块；`tensor` 为全张量共享一个局部缩放码 |
| `output_type` | `fp16` / `bf16` / `fp32` |
| `rounding` | `nearest` 为 ties-to-even；`stochastic` 按相邻可表示值距离随机选择 |
| `seed` | 随机舍入种子，默认 42；与元素线性索引共同决定结果 |
| `target_gpu` | 仅记录目标机器说明；日志的 `gpu` 是实际运行设备 |

默认块尺寸对应标准格式；其他块尺寸与 per-tensor 模式是本题要求的缩放对照扩展，不能当作硬件原生 MXFP8/NVFP4 块布局。输入限定有限 FP32/FP16 数值；全零块、尾块、负零均有定义。输出类型应能容纳反量化数值，例如大幅值 FP32 输入应选择 FP32 输出。

## 数值与布局

MXFP8 采用 E4M3 有限数编码，最大幅值 448；每 32 个连续元素默认保存 1 字节 E8M0 缩放码。设块最大幅值为 `a`，选择 `e = clamp(ceil(log2(a/448)), -127, 127)`，保存 `e+127`，反量化为 `decode_E4M3(q) * 2^e`。全零块使用缩放 1。向上取整策略参照 [NVIDIA MXFP8 说明](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/mxfp8/mxfp8.html)。

NVFP4 的 E2M1 正幅值码本为 `{0, 0.5, 1, 1.5, 2, 3, 4, 6}`，符号位在 bit 3。全局缩放 `g = max(abs(x))/(448*6)` 存为 FP32；局部缩放 `s = E4M3_RNE((block_amax/g)/6)` 存为 1 字节。反量化为 `(decode_E2M1(q) * decode_E4M3(s)) * g`，与 [NVIDIA NVFP4 定义](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html) 一致。全零张量设 `g=1`；极小全局缩放下限为 FP32 最小正规数，避免全局除法溢出。局部缩放舍入为零时该块量化为零。

元素舍入使用配置指定策略；缩放因子始终确定性计算。溢出到格式范围外时饱和到最大有限值；低于最小非零值时按指定策略舍入。FP8 不产生 NaN 编码。

NVFP4 每个字节保存同一行的两个元素：偶数列在低 4 位，奇数列在高 4 位。奇数列数的行最后一个高半字节填零，因此 packed data 大小为 `rows * ceil(cols/2)`。缩放块不会跨行，最后一个不足块长的块仅统计有效元素。

### 二进制文件协议 v1

所有整数/浮点字段为 little-endian，无隐式结构体 padding。题目仅给出逻辑字段，未指定字符串长度和物理头格式，本实现明确以下协议，并提供 Python 读写工具。

张量头长 32 字节，Python struct 格式 `<8sQQII`：

| 字节偏移 | 类型 | 内容 |
|---:|---|---|
| 0 | char[8] | `LPTENS1\0` |
| 8 | uint64 | rows |
| 16 | uint64 | cols |
| 24 | uint32 | dtype：0=FP32，1=FP16，2=BF16 |
| 28 | uint32 | reserved=0 |
| 32 | dtype[] | 行主序数据，无 padding |

packed 头长 56 字节，Python struct 格式 `<8sQQIIIfQQ`：

| 字节偏移 | 类型 | 内容 |
|---:|---|---|
| 0 | char[8] | `LPPACK1\0` |
| 8 / 16 | uint64 / uint64 | rows / cols |
| 24 | uint32 | format：0=MXFP8，1=NVFP4 |
| 28 | uint32 | block_size |
| 32 | uint32 | tensor scaling：0/1 |
| 36 | float32 | global scale；MXFP8 固定 1 |
| 40 / 48 | uint64 / uint64 | scale 字节数 / packed data 字节数 |
| 56 | uint8[] | scale 数组，随后是 packed data |

BF16 以 16 位原始位模式存储，使用整数实现 RNE 转换，在 `sm_75` 上可运行。文件格式为线性软件交换格式，不是 Blackwell GEMM 的 swizzled 硬件布局。

## 验证与计时

主程序运行 scalar C++ CPU 参考并逐字节比较 GPU 输出。独立 Python oracle 用显式码本搜索实现编码，避免与 CUDA 编码算法同源。

`quant_ms` 包含所需的全局归约、局部 scale kernel 和打包 kernel；`dequant_ms` 为解包反量化 kernel。每组预热 3 次，用 CUDA event 统计多次执行的均值。`gpu_with_transfers_ms` 包含 H2D 输入、量化、反量化、D2H 输出，不含文件 I/O、内存分配和正确性检查；对应 CPU 时间是单线程 scalar 量化加反量化。基准脚本每项启动三个独立进程并报告中位数，同时保留全部原始日志。显存数据热缓存，未锁 GPU 时钟。

`quant_effective_gbps` 与 `dequant_effective_gbps` 按逻辑输入、输出和 scale 字节数计算，不代表 profiler 测得的 DRAM 流量。压缩率分别报告 payload（含全部 scales 和 global FP32）以及完整文件（含 header）；不能仅用 32/4 推算实际压缩率。

实测结果及分析见 [REPORT.md](REPORT.md)，原始数据见 [results/](results/)。

工具检查脚本为 `tests/profile.py`，可用 `--ncu`、`--sanitizer` 指定可用工具路径；仅检查本题。本机 WSL 的 profiler / 调试接口限制及失败日志单独记录，不计作检查通过。报告和图表通过 `python3 tests/report.py` 从本题 JSON 重建，额外依赖 Matplotlib。

原生 Linux 分析结果及本机 WSL 设置说明见 [PROFILING.md](PROFILING.md)。

## 优化实现与原生 Linux 分析

默认块长度的 scale 计算和量化写出已融合，整块对齐张量使用连续 tile，奇数列和行尾保持独立打包。E4M3 编码/解码使用整数位域，E8M0 使用精确二次幂缩放；NVFP4 最近偶数舍入使用缩放后的精确阈值，随机舍入保留原有商与概率计算。

```bash
make clean && make ARCH=89 NVCC=/usr/local/cuda/bin/nvcc HOSTCXX=g++
python3 tests/validate.py
python3 tests/profile_native.py
python3 tests/report_4090.py
```

大张量前后交替比较可用 `tests/benchmark_extended.py --before /path/to/old/quantize --after build/quantize`。`tests/report.py` 对应首版数据；`tests/report_4090.py` 使用独立归档的 4090 D 结果。
