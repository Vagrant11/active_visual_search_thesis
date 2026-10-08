# M11：传统搜索基线

已完成20个M8 val φ的评测。阅读 [REPORT.md](REPORT.md)、[summary.csv](summary.csv) 和 [results.csv](results.csv)。没有训练、求解ETO或读取test结果；旧文件的大小和修改时间全部保持不变，见 verification.json。

新增代码：experiment.py（策略、离散动力学、复用harness评测）和 report.py（等权汇总和图）。策略及数据全部仅放在此目录；运行禁止生成旧目录字节码，绘图缓存也限定在本目录。

初次运行命令（本目录已完成，保护锁会拒绝重复运行）：

```sh
/opt/miniconda3/envs/erg/bin/python -B milestone_11_classical_baselines/experiment.py
```

若生成完42条轨迹后仅评测中断，可读取已落盘轨迹继续，绝不重新生成：

```sh
/opt/miniconda3/envs/erg/bin/python -B milestone_11_classical_baselines/experiment.py --evaluate-existing
```

本次曾修复汇总字典中重复的tf字段，之后使用上述评测恢复入口；原失败日志保留于run.log，42条轨迹没有重跑。

参数直接读取M9保存的M8环境和planner实际约束。算法新增固定参数见PROTOCOL.json：路由64顶点、径向余量1e-4、观测停靠点偏移1e-6、搜索上限120s。偏移仅避免相机与格子中心共点时方位角未定义，不将到达视为发现。没有按结果调参。

数据与统计约定：

- 主表共同时间Tφ为同φ五条ETO的平均tf；目标概率为五条ETO在Tφ的平均Pdet。
- first_seen_grid和条件检测时间原样复用M4函数；原φ加权，不用贪心更新后的后验加权评测。
- reached=false表示120s内未达到目标；reach_time为空。restricted_reach_time对此按120计入，不能解释为成功时间。
- 全样本和物理可行子集同时保留。子集按φ等权，phi_count列显示真正参与汇总的φ数。
- 四种传统策略终点自由；既有ETO/NN/diffusion终点固定，报告明确注明这一限制差异。
- 新策略计算时间包含整条搜索策略的构建/在线模拟，旧方法引用历史计时，不能据此作严格硬件性能排名。
