# M4 轨迹与可见性重放分析

最终核验版本：**[run_20260910_verified/REPORT.md](run_20260910_verified/REPORT.md)**。

三个任务共同限制：M3、冻结协议及既有结果只读；不运行规划器；不修改、重建或覆盖既有结果；新增文件全部位于本目录。PPT 未修改。

| 内容 | 文件 |
|---|---|
| 546 条有效正式计划 + 26 条 pilot 的逐计划诊断量 | [trajectory_diagnostics.csv](run_20260910_verified/trajectory_diagnostics.csv) |
| 按 dataset / γ / family 的均值和中位数 | [diagnostic_summary.csv](run_20260910_verified/diagnostic_summary.csv) |
| 按 JS 进一步分组的诊断量 | [diagnostic_summary_by_js.csv](run_20260910_verified/diagnostic_summary_by_js.csv) |
| 与同场景、同 γ oracle 配对的差值汇总 | [diagnostic_oracle_deltas.csv](run_20260910_verified/diagnostic_oracle_deltas.csv) |
| 6 条无效正式计划及原因 | [excluded_plans.csv](run_20260910_verified/excluded_plans.csv) |
| γ 语义及源代码行号引用 | [gamma_semantics.md](run_20260910_verified/gamma_semantics.md) |
| γ=0.05 可见性得失图 | [coverage_maps_gamma_0.05.png](run_20260910_verified/pilot_figures/coverage_maps_gamma_0.05.png) |
| γ=0.05 先验与轨迹叠加图 | [trajectories_gamma_0.05.png](run_20260910_verified/pilot_figures/trajectories_gamma_0.05.png) |
| pilot 两个 γ 并排的 gained / lost / net | [pilot_gains_losses_comparison.csv](run_20260910_verified/pilot_gains_losses_comparison.csv) |
| 正式套件逐方向可见质量及得失 | [suite_visibility_by_direction.csv](run_20260910_verified/suite_visibility_by_direction.csv) |
| 正式套件逐场景数值 | [suite_visibility_by_scene.csv](run_20260910_verified/suite_visibility_by_scene.csv) |
| 场景均值、中位数、净增场景数 | [suite_visibility_summary.csv](run_20260910_verified/suite_visibility_summary.csv) |
| 三条原始计划的独立抽查数值与格子计数 | [three_plan_spot_checks.json](run_20260910_verified/three_plan_spot_checks.json) |
| 7,020 个原文件的完整性核验 | [source_integrity_audit.json](run_20260910_verified/source_integrity_audit.json) |
| 四张 PNG 与原图逐像素比较 | [plot_pixel_comparison.json](run_20260910_verified/plot_pixel_comparison.json) |
| 10 项单元测试记录 | [unit_tests_verified.log](unit_tests_verified.log) |

图片目录同时提供 γ=0.10 和 PDF。原项目其实已有 γ=0.05 的两类 pilot 图；本次沿用原绘图函数或原绘图语句，在新目录生成，四张 PNG 均与原图逐像素一致。

诊断量主表分别列出三种熵：50 个等权规划节点在 40×40 先验网格上的占用熵、连续轨迹到 tf 的格子停留时间熵、执行到 15 秒的格子停留时间熵。规划器自身计算的是 Fourier ergodic metric，不是网格熵；三种熵不可混用。单位与聚合口径详见报告。

任务 1 的汇总对有效计划等权；正式套件和 pilot 分开。任务 3 先在场景内平均 x/y 方向，再对场景等权；不完整的方向对在逐场景表保留并标记，不进入跨场景汇总。不存在 pilot 的 uniform 规划结果，本次未补跑。

从仓库根目录复现（自动创建全新运行目录，拒绝覆盖已有运行目录）：

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 /opt/miniconda3/envs/erg/bin/python -B milestone_4/results/mechanism/replay_analysis.py
PYTHONDONTWRITEBYTECODE=1 /opt/miniconda3/envs/erg/bin/python -B -m unittest discover -s milestone_4/results/mechanism -p test_replay.py -v
```

`run_20260910/` 是首次运行的保留输出；`run_20260910_verified/` 是增加三条正式可见性重放、逐像素检查并消除数值库警告后的核验输出。既有输出没有覆盖。`source_inventory_before.json` 保存工作开始时的原文件 SHA-256；缓存也限定在本目录。
