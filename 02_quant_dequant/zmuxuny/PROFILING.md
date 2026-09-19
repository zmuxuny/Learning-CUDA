# 本机 Profiler / Sanitizer 处理步骤

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
