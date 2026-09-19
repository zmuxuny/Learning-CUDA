# RTX 4090 D：已完成的工具检查

2026-09-19 在原生 Linux 容器、CUDA 12.8、驱动 570.124.06 上完成。Profiler 为题目加分项；Compute Sanitizer 未被题目列为硬性提交条件，本版已补齐真实检查。

## Compute Sanitizer

`python3 tests/profile_native.py` 依次运行 memcheck、racecheck 和 synccheck，并设置 `--error-exitcode 99`。题目 3 覆盖 19×128 FP16、35×1024 FP16/BF16，两种格式，共 18 次检查；同一次程序执行包括蝶形、稠密 WMMA（FP16）、分解 MMA 及融合路径。检查全部通过，原始命令、工具摘要和退出码位于 `results/4090d/after/`，汇总见 [profiling.json](results/4090d/after/profiling.json)。数值正确性另由独立 NumPy oracle 检查。

## Nsight Systems

成功使用 `nsys profile --sample=none --cpuctxsw=none --trace=cuda,nvtx` 采集两种格式的 CUDA 时间线；关闭 CPU 采样是因为容器不开放 Linux perf_event，CUDA tracing 正常。

- [MXFP8 时间线](results/4090d/after/timeline_mxfp8.nsys-rep)、[kernel/API 汇总](results/4090d/after/nsys_stats_mxfp8.csv)。
- [NVFP4 时间线](results/4090d/after/timeline_nvfp4.nsys-rep)、[kernel/API 汇总](results/4090d/after/nsys_stats_nvfp4.csv)。

程序包含多个实现、验证和传输实验，因此汇总百分比属于整个验证程序，各实现的调用次数也不同；比较路径使用无插桩 benchmark 的 CUDA event 时间。初始化分配、CPU 参考和文件 I/O 不计入 kernel benchmark。

采样输入为 8192×128 FP16，开启随机符号。以下为 Nsight Systems kernel 平均耗时（带插桩）：

| 阶段 | 蝶形 μs | 分解 Tensor Core μs |
| --- | --- | --- |
| 单独变换，MXFP8 实验 | 3.281 | 2.989 |
| MXFP8 融合输出 | 5.838 | 4.502 |
| NVFP4 变换后 amax 预遍历 | 4.064 | 2.641 |
| NVFP4 融合输出 | 8.100 | 5.174 |

每个融合/预遍历 kernel 调用 23 次（3 次预热 + 20 次重复）。原稠密 WMMA 平均约 13.3–13.5 μs；分解实现将完整 D×D 矩阵乘法替换为 H16 小块乘法与寄存器蝶形，时间线显示单独变换和融合输出均有收益。MXFP8 融合省去中间张量及后续独立量化 kernel；NVFP4 仍保留两阶段，Tensor Core 同时加速预遍历和最终输出。其总流程还包含清零和启动间隔，不能仅将两个 kernel 的平均值当作总延迟。

[反汇编证据](results/4090d/after/tensor_core_instructions.txt) 按函数保留 `HMMA` 指令，可确认 FP16/BF16 的分解变换及融合函数实际使用 Tensor Core。[静态资源报告](results/4090d/after/resources.txt) 显示分解 MMA 各实例无 local-memory 分配和栈帧；D=128 的融合使用32–37 个寄存器，D=1024 使用 56–64 个寄存器。融合输出不使用共享内存；全局 amax 预遍历使用每 CTA 16 字节共享内存汇总 4 个 warp 的最大值。这些是编译器资源信息，不等同于实测 occupancy。


## Nsight Compute 权限

NCU 2025.1.0 已能识别设备，但返回 `ERR_NVGPUCTRPERM`。宿主机 `RmProfilingAdminOnly=1`，容器缺少 CAP_SYS_ADMIN，容器内 root 不能自行开放计数器。这次保留 [原始错误](results/4090d/after/ncu.txt)，没有硬件 DRAM 利用率、实际 occupancy 或 stall 指标。Nsight Systems 时间线、CUDA event 对比和反汇编证据均已完成。

若后续算力提供方开放性能计数器，可继续用 `ncu --set full --launch-count 10 ...` 补充硬件分析，不必重写实现。

---

# 本机 WSL 工具设置（首版环境）

## 是否为验收必需

题目文档对题目 2、3 都写明“包含 ncu 和/或 nsys 使用和分析的加分”。Profiler 属于加分项；文档未将 Compute Sanitizer 列为必需项。程序正确性、测试、误差与实测性能报告仍须完成。

## Nsight Compute：升级工具并开放计数器

当前实际报错来自 Nsight Compute 2021.3.1，该版本不支持本机 WSL profiling，不能据此判断新版本也不能用。

1. 在 WSL 内安装与显卡驱动兼容的 Linux x86_64 版本。当前 Windows 驱动为 576.80；Nsight Compute 2025.2 系列的官方推荐 Windows 驱动起点为 576.57，可作为匹配选择。先确认下载版本要求，不要直接安装需要更新驱动的最新版本。
2. 在 Windows NVIDIA 控制面板中开启开发者设置，并允许 GPU 性能计数器访问。WSL 中的 root 身份不能替代 Windows 侧设置。
3. 在本题目录运行下面的检查脚本，指定新工具的实际路径：

```bash
python3 tests/profile.py --ncu /path/to/new/ncu --sanitizer /path/to/new/compute-sanitizer
```

当前脚本是少量 kernel 的检查入口；正式性能分析可对较大输入使用 `ncu --set full --launch-count 10 ...`，按 kernel 筛选后采集带宽、占用率、寄存器和 stall 指标。Profiler 下运行时间受插桩/重放影响，不能直接替换无插桩 benchmark。

依据：[Nsight Compute 2025.2 下载与驱动要求](https://developer.nvidia.com/tools-overview/nsight-compute/get-started-2025_2)、[WSL 支持与计数器要求](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/system-requirements.html)。

## Compute Sanitizer：启用 Windows 调试接口

已尝试 CUDA 12.9 配套的 Compute Sanitizer 2025.2.1，报错为 `Failed to initialize WDDM debugger interface`，要求管理员运行 `EnableDebuggerInterface.bat`。目前是工具初始化失败，未完成 kernel 内存/竞争/同步检查。

可在 Windows 管理员终端运行 NVIDIA 提供的 `EnableDebuggerInterface.bat`。官方文档明确规定的等效设置是以下注册表 DWORD（在 Windows 管理员 PowerShell 或 CMD 执行，不是在 Linux shell）：

```powershell
reg.exe add "HKLM\SOFTWARE\NVIDIA Corporation\GPUDebugger" /v EnableInterface /t REG_DWORD /d 1 /f
```

设置后使用新版工具重跑本题的 `tests/profile.py`，确认 `memcheck`、`racecheck`、`synccheck` 实际检查完成且无错误。初始化失败时不能以程序数值输出正常推断 sanitizer 已通过。

依据：[Compute Sanitizer 官方 Windows 调试接口要求](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html#windows-specific-behavior)。

## 另一条可行路径

将本题分支拉到原生 Linux NVIDIA GPU 服务器上，用配套 Toolkit 工具编译运行，避免 WSL 的 Windows 调试接口限制。`ncu` 仍可能需要管理员开放 GPU 性能计数器权限。这也是后续在训练营算力上补充跨设备性能数据的方式。

本次首版保留现有失败日志及原因，没有将这些工具检查计作通过。完成环境设置后再提交真实检查结果和性能分析。
