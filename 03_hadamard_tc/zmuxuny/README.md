# Hadamard 变换与量化融合

题目 3，提交 ID：`zmuxuny`。实现 FP16/BF16 快速 Walsh-Hadamard 变换、MXFP8/NVFP4 融合量化，以及 FP16 WMMA Tensor Core 对照路径。本目录随附量化、文件读写和独立参考模块，可单独构建、测试和提交。

## 使用

```bash
cd 03_hadamard_tc/zmuxuny
make
mkdir -p results/tmp
python3 tests/generate.py \
  --rows 8192 --cols 128 --dtype fp16 --output results/tmp/input.bin
./build/hadamard --input results/tmp/input.bin --batch 1 --seq 1024 --heads 8 \
  --output results/tmp/hadamard.bin --packed results/tmp/fused.bin \
  --unfused_packed results/tmp/unfused.bin --log results/tmp/metrics.json \
  --format nvfp4 --random_sign 1 --normalize 1 --repeats 100
python3 tests/validate.py
python3 tests/benchmark.py
```

依赖 CUDA Toolkit >=11.0、C++17 编译器及 Make，Python 脚本依赖见 `requirements.txt`。默认目标 `ARCH=75`；CUDA 11.5 优先使用 GCC 10，可通过 `HOSTCXX` 指定。切换架构前执行 `make clean`。文件格式见 [FILE_FORMAT.md](FILE_FORMAT.md)。输入张量文件使用二维行主序格式，逻辑形状为 `[batch, seq, heads, head_dim]`，前三维展平为 rows。`--batch * --seq * --heads` 必须等于 rows；默认是 `[1, rows, 1, cols]`。头维度支持 1–1024 的 2 的幂，包括验收所需的 64、128、256。

## 变换定义

采用自然顺序 Sylvester 矩阵 `H_1=[1]`，`H_2D=[[H_D,H_D],[H_D,-H_D]]`。默认输出 `y=x H_D / sqrt(D)`；`--normalize 0` 使用未归一化变换。`--random_sign 1` 在变换前沿最后一维应用固定 Rademacher 符号，所有行共用；`--sign_seed` 默认 7。

计算在 FP32 寄存器中进行，最终转换回输入 dtype。每个 warp 处理一行、每个 CUDA block 有 4 个 warp；前 5 级蝶形通过 warp XOR shuffle 完成，其余级在同一线程的寄存器间完成。

## 融合语义

`--format mxfp8` 使用 32 元素块，`--format nvfp4` 使用 16 元素块。融合路径在寄存器中完成 Hadamard、转换回 FP16/BF16、块最大值计算、缩放与打包。**中间 dtype 舍入保留**，因此结果和先写出 Hadamard 结果、再调用本目录量化代码逐字节一致。支持 `--rounding nearest/stochastic` 和 `--seed`（默认 42）。

NVFP4 的全局 scale 依赖整个变换后的张量，不能仅凭单个 warp 的数据确定。实现先执行只计算变换后 amax 的 kernel，再执行融合输出 kernel。报告的 `fused_ms` 包含这次归约、清零和第二次变换；不会把先算 scale 的时间排除。MXFP8 不需要全局归约，仅执行一次融合 kernel。

主程序同时运行非融合和融合路径，比较 scales、global scale、packed data；任何不一致都以非零状态退出。融合文件可由 `tests/reference.py` 的 `read_packed` / `dequantize` 读取，协议也与题目 2 兼容。

## Tensor Core 对照

FP16 且 `head_dim >=16` 时默认运行 WMMA `16x16x16`，输入为 FP16、累加为 FP32，并输出 `tensor_core_ms` 和误差。`--tc_output path` 可保存结果，`--tensor_core 0` 可关闭。它实现稠密 `X*H`，复杂度为 `O(D²)`；蝶形为 `O(D log D)`，性能比较明确区分算法。

BF16 使用可移植蝶形路径。Tensor Core 对照当前只实现 FP16，不把 BF16 转成 FP16 来冒充全范围 BF16 支持。主程序输出仍为蝶形结果，量化融合也在蝶形路径上完成。

## 测试与实验

- 独立 NumPy float64 稠密 Hadamard 矩阵验证，FP16 最大绝对误差 `<1e-2`、BF16 `<5e-2`。
- 覆盖所有支持头维度、两种 dtype、两种融合格式、非整 CTA 行数、零行、单位脉冲、归一化/未归一化及随机符号。
- 融合量化与非融合结果比较，并与 NumPy 独立量化参考比较。
- 基准覆盖小批量及 8192 行、多个头维度，保留三次独立运行数据，报告中位数。
- `rotation_quality.json` 在原坐标系比较直接量化与“随机旋转→量化→反变换”的 MSE，避免把不同坐标系的逐元素误差直接比较。

使用 CUDA event，预热 3 次；重复次数由 `--repeats` 指定。有效带宽按逻辑输入输出字节数计算。`unfused_ms` 包含变换和完整量化，`fused_ms` 包含融合所需的所有 GPU 步骤。小尺寸主要受启动开销影响，数据传输计时单独报告。实测分析见 [REPORT.md](REPORT.md)。

运行 `python3 tests/report.py` 可重建本题报告和图；`tests/profile.py` 可独立运行 profiler/sanitizer 检查。`include/` 和 `tests/reference.py` 是本作者题目 2 数值模块的本地副本，初始数值版本为 b7480df；未改动运算规则，保留副本是为使两份 PR 不依赖彼此的合并顺序。

本机 WSL 工具限制的处理步骤见 [PROFILING.md](PROFILING.md)。
