# PocketWorld 现有结果与失败案例摘要

本页只汇总仓库已有结果和已验证状态；v1 正式三 seed adaptive-horizon 报告已经完成，完整报告通过申请材料中的 evidence release 公开，提交摘要同时保留来源 SHA256。

## 已经证明了什么

| 证据 | 现有结论 | 不能推出的结论 |
|---|---|---|
| 64×64 RGB simulator + 自动交互数据 | 可以从零实现可复现的 world-model 训练与规划闭环 | 不能推出真实机器人可用 |
| 一步/多步 imagination-gap 评测 | 视界增加时模型误差会影响真实执行 | 不能推出某个 horizon 普遍最优 |
| temporal velocity + held-out calibration | 速度表征、不确定性 coverage、NLL 和 collision Brier 可被独立测量 | 不能把 calibration 当成安全保证 |
| route-aware/remaining-budget 研究 | 路线进度是 local collision score 看不到的重要变量 | 不能证明纯学习 planner 已学会通用障碍绕行 |
| v25 adaptive solver gate | ordinary/robust MPC 切换可节省部分计算 | 不能称为 adaptive horizon，也未建立稳定安全收益 |
| RGB/A* hybrid fallback | 显式几何可作为可靠对照或安全 fallback | 其成功不能归因于纯世界模型 |

## 关键历史数字

README 和 `docs/evaluation-2026-08.md` 中已经提交的结果显示，v3-final 的 24-step imagined/real success 为 `100.0% / 95.3%`，32-step 为 `100.0% / 93.3%`；这说明存在可测量的 imagination-real gap。障碍实验中，纯 learned route/planner 曾出现低成功率或高碰撞，而含 RGB geometry/A* fallback 的 hybrid 方法达到更高完成率。上述数字属于各自历史协议，不应与新的 adaptive-horizon 协议混用。

## 当前新方向的已验证状态

- 新模块 `pocketworld/adaptive_horizon.py` 已实现风险预算、最长可行视界选择和 hysteresis。
- 新评测器比较固定 8/16/24/32、旧 solver gate、新 adaptive horizon 和 robust ablation。
- calibration/final split 已冻结为 calibration `53,67`、final `11,23,41`。
- smoke 已真实运行：一条 final smoke case 中出现 32→8 horizon 切换，纯学习 real success 为 `0.0`；这只是管线检查，不是正式结论。
- 全量测试 `145 passed`，覆盖率基线 `70.40%`。
- 完整三 seed 正式评测已完成：4 个条件、每条件 60 个 paired episodes（每个 final seed 20 个）。完整本地报告为 `artifacts/evaluation-adaptive-horizon-v1.json`，可提交的精简结果为 [`docs/results/evaluation-adaptive-horizon-v1-summary.json`](../results/evaluation-adaptive-horizon-v1-summary.json)。

## 正式 adaptive-horizon 结论

在纯学习 ID 轨道，adaptive horizon 的 real success 为 **11.7%**、碰撞为
**15.53/episode**；fixed-16 为 **8.3%**、**13.65/episode**。地图平移、
快速速度和联合 OOD 下，adaptive horizon 都降低了 model-query 预算，但
没有相对 fixed-16 降低碰撞。因此不能宣称安全性提升，只能宣称“计算预算
适应 + 暴露了短视界不足以解决障碍穿越的负面结果”。

注意：adaptive evaluator 的 `imagined_success` 只表示第一次选中计划的模型预测终点是否到达目标；它与闭环 `real_success` 的差值是 first-plan diagnostic，不应写成严格的 imagined-vs-real policy gap。

含 A* 的 hybrid 轨道成功率更高，但碰撞与纯学习不同，必须单独归因；
robust-MPC ablation 的低碰撞伴随显著更高延迟，也不能归因于 horizon 本身。

## 失败案例的研究价值

### 1. 想象成功不等于真实成功

模型 rollout 可能认为某条候选序列最终接近目标，但真实惯性、碰撞或长时状态漂移使执行偏离。这个 gap 正是自适应 horizon 要处理的对象。

### 2. local collision risk 看不到路线承诺

智能体已经绕过一个障碍后，若 planner 只看当前 endpoint 或重新随机采样，可能重新选择另一侧路线，造成不必要的回退和碰撞。因此项目加入 route progress、remaining budget 和 route lock；这些改动解决的是评测中暴露出的控制问题，不等于已得到通用导航能力。

### 3. OOD 下风险信号可能失准

速度变化、地图平移和观测创新会让 ID calibration 失效。若自适应策略在 OOD 下频繁缩到最短 horizon 但仍失败，说明“有不确定性数值”不等于“有可用的风险边界”。这类负面结果将直接决定下一阶段是否需要更强的传感器模型、ensemble 或机器人数据。

## 评审应关注的判断

申请人不应说“已经解决安全规划”。更准确的说法是：已经构建了能把 world-model error、horizon、collision risk 和真实失败放在同一配对协议中观察的研究原型；下一步需要算力、机器人仿真和专业指导，验证这种可解释的自适应视界是否真的改善可靠性。
