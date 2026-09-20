# 题目 3 提交说明

本题独立提交；文档未强制要求每题单独 PR，这样拆分便于逐题评审。基于上游 `2026-summer-project`，仅新增 `03_hadamard_tc/zmuxuny/`，不依赖另一题 PR 的合并顺序。

- fork：`zmuxuny/Learning-CUDA`
- head：`projects/hadamard`
- base：`InfiniTensor/Learning-CUDA:2026-summer-project`
- 标题：`【训练营】【题目3】Hadamard 矩阵加速与量化融合（多平台）`

本题目录内运行：

```bash
make
python3 tests/validate.py
python3 tests/benchmark.py
python3 tests/report.py
```

提交后官网若要求 fork、commit 链接，使用本题分支及对应 commit。PR 正文见 `PR_BODY.md`。旧的合并开发分支 `projects/quant-hadamard` 仅用于本地集成，不作为本题 PR 的 head。
