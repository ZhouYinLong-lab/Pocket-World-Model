# PocketWorld 项目申请书（学生科研项目草案）

## 1. 项目摘要

本项目研究一个模型式强化学习中的基础问题：当智能体依赖学习到的世界模型进行规划时，模型预测误差会随着 imagination horizon 增长而累积，并可能把“模型中可行”的路线变成真实环境中的碰撞或失败。PocketWorld 将问题压缩到一个 64×64 RGB 二维世界中，使预测、规划、碰撞和真实执行都可以被可视化、配对和复现。

项目的下一阶段问题是：**在固定交互和计算预算下，基于校准不确定性的自适应想象视界，能否减少世界模型误差造成的真实规划失败？**

项目不是要声称已经解决机器人安全规划，也不把二维结果直接外推到 Unitree 机器人。它的价值在于先建立一个低成本、可审计的研究显微镜，再用宇树的仿真、硬件和工程指导验证哪些结论可以迁移、哪些不能迁移。

## 2. 前期基础与独立完成工作

申请人已完成从环境、数据、模型、规划、评测到展示的完整原型：

1. 环境：确定性二维 RGB simulator，含惯性、墙壁、圆形智能体、目标和碰撞反馈。
2. 世界模型：CNN encoder、GRU/latent dynamics、decoder，以及显式位置/速度表征。
3. 不确定性：状态标准差、held-out scale calibration、概率碰撞预测、时序速度估计和无标签 shift score。
4. 规划：random shooting、普通/鲁棒 MPC、route-aware progress、剩余预算、路线锁定和在线重规划。
5. 评测：固定视界、想象—真实 gap、OOD 速度/地图变化、碰撞、成功率、延迟和三 seed 报告。
6. 工程：Python package、CLI、pytest、coverage、GitHub Actions、可视化页面和 JSON 报告。

仓库的历史提交记录、实验计划、失败结果和测试构成前期投入证据。当前本地可核验 HEAD 为 `90689c0`；`pytest -q` 实际输出为 `140 passed`，`pytest --cov=pocketworld --cov-report=term-missing` 实际总覆盖率为 `70%`。新 adaptive-horizon smoke 已成功生成严格 JSON，但完整三 seed 正式评测在本材料编写时仍需以最终退出码和结果文件确认。

## 3. 已知结果与问题边界

已有三 seed 结果支持三个谨慎结论：

- 短视界的想象规划与真实执行之间的差距较小；视界增长后 gap 变大，说明“下一帧预测正确”不等于“长视界规划可靠”。
- learned temporal velocity 和 calibrated uncertainty 可以被实现并在 held-out rollout 上测量 coverage、NLL 和 shift detection。
- 障碍穿越是更难的失效模式。显式 RGB geometry/A* fallback 可以提高完成率，但它是混合方法，不是纯世界模型理解的证据。

同时必须保留负面结论：历史 v25 adaptive solver gate 在固定 horizon 下只是在 ordinary MPC 和 robust MPC 之间切换；它更像计算预算适应，没有证明稳定的安全性提升。新 adaptive horizon 也可能只降低 model queries，甚至因缺少远期路线信息而降低成功率。

## 4. 研究假设与方法

可证伪假设 H1：在相同交互步数和配对任务下，在线校准 uncertainty、collision risk、route alignment 和 shift score 的 adaptive horizon，相比 fixed horizon 16 能在成功率基本不下降的情况下减少碰撞或真实失败。

反例也明确：若 adaptive horizon 只减少计算量、碰撞不降，结论为计算预算适应；若成功率下降或 OOD calibration 不稳定，结论为当前信号不足以支持安全视界控制。

候选视界为 `(8,16,24,32)`。透明风险分数为：

```text
R_h = 0.40 U_h + 0.30 C_h + 0.15 A + 0.10 O + 0.05 P
```

其中 `U_h` 是校准累计状态不确定性，`C_h` 是累计碰撞概率，`A` 是路线对齐误差，`O` 是无标签 shift score，`P` 是近期风险压力。选择满足预算的最长 horizon；全部超预算时回退到 8；用 entry/exit hysteresis 避免频繁振荡。第一轮固定求解器，robust MPC 只作为独立消融，避免把 solver 作用与 horizon 作用混在一起。

## 5. 数据划分与实验协议

- 训练：沿用现有训练 seeds `101,103,107`。
- 校准：只使用 `53,67`，用于 held-out uncertainty scale 和阈值设定。
- 最终 holdout：只使用 `11,23,41`，不参与阈值调整。

正式协议每个 final seed 至少 20 个 paired episodes，包含 ID、地图平移、速度变化和联合 OOD；固定 48-step interaction budget。固定 horizon、adaptive horizon 和 solver reference 共享初始案例和 action bank；纯学习轨道与 A* fallback 轨道分开报告。指标包括 imagined/real success、gap、collision、final distance、route completion、replanning、latency、horizon distribution/switching、coverage、Brier/NLL、跨 seed 均值和标准差。

## 6. 需要帮扶的原因

个人条件足以完成二维原型，却不足以有质量地完成下一阶段的多条件统计验证与机器人迁移。主要约束是：

- GPU/云算力不足，使完整多方法、多 seed、多 OOD paired matrix 运行成本高，无法快速进行重复校验。
- 没有经过工程验证的移动机器人仿真与传感器模型，无法评估相机噪声、延迟、速度估计和控制频率变化。
- 缺少低风险硬件测试平台，无法把“模型误差—规划失败”链条从像素模拟扩展到受控真实观测。
- 缺少机器人导航、模型式 RL 和安全实验设计方面的专业反馈，尤其是碰撞代价、停止策略和 sim-to-real 边界。

宇树的资源能够把下一阶段从“个人电脑上的可信原型”推进到“受控仿真和低风险硬件验证”，而不是简单替项目增加功能。

## 7. 预期交付与负面结果处理

12 周内交付一份可复现结果包、固定/自适应 horizon 对比图、OOD calibration 报告、失败案例集、移动机器人仿真 baseline、噪声/延迟实验记录、技术评审纪要和公开代码。若 H1 不成立，仍交付：

- 一个能复现 imagination-real gap 的评测工具；
- 一个明确区分 solver gate 与 adaptive horizon 的实现；
- 一份说明 calibration、route alignment 或 shift detection 哪个环节失效的负面报告；
- 一套为后续机器人实验降低风险的基线和安全检查清单。

这使项目即使没有漂亮的成功率，也能形成可检查的方法学成果。
