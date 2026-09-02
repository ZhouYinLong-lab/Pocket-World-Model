# PocketWorld 帮扶申请材料包

这组材料用于支持宇树科技“天才少年”项目的经费、设备、技术指导和实验资源申请。材料基于仓库当前可核验状态编写，不把 PocketWorld 表述成已经解决真实机器人安全规划问题的成果。

## 建议阅读顺序

1. [一页项目摘要](one-page-summary.md)
2. [正式项目申请书](project-proposal.md)
3. [现有结果与失败案例](results-and-failures.md)
4. [十二周执行计划](twelve-week-plan.md)
5. [里程碑与经费预算](budget-and-milestones.md)
6. [为什么需要宇树帮扶](why-unitree-support.md)
7. [安全与合法合规说明](safety-and-compliance.md)
8. [最小复现说明](minimal-reproduction.md)
9. [证据清单](evidence-manifest.json)

## 当前证据状态

- 申请材料中的正式结果只引用仓库中已提交且可复现的结果。
- adaptive-horizon v1 三 seed 正式评测已完成；smoke 只用于入口和日志检查，不能替代正式结果。
- v1 的统计摘要已经包含每 seed 结果和 paired bootstrap 区间；v2 的风险条件分析、噪声/延迟和连续控制迁移仍是申请支持的下一阶段。
- A*、RGB wall geometry 和 route-field hybrid 结果必须标记为混合方法，不能归因于纯世界模型。
- 当前项目是二维小型模拟器上的研究原型，不是已经通过真实机器人安全验证的系统。
- adaptive evaluator 的 `imagined_success` 是第一次选中计划的预测终点诊断，不与闭环 `real_success` 构成严格的 policy-level gap。
