# 最小复现实验

## 环境

```bash
python -m pip install -e ".[dev]"
```

## 最小检查

```bash
pytest -q
pytest --cov=pocketworld --cov-report=term-missing
```

当前验证结果：`141 passed`，总覆盖率基线为 `70.40%`。

## adaptive horizon smoke

```bash
python -m pocketworld.evaluate_adaptive_horizon \
  --protocol configs/adaptive-horizon-v1.json \
  --smoke \
  --output artifacts/evaluation-adaptive-horizon-smoke.json
```

smoke 使用 calibration seed 53 和 final seed 11 的最小子集，仅用于验证入口、日志和 JSON 合法性；它不是三 seed 正式结论。

## 正式协议

```bash
python -m pocketworld.evaluate_adaptive_horizon \
  --protocol configs/adaptive-horizon-v1.json \
  --output artifacts/evaluation-adaptive-horizon-v1.json
```

正式配置固定 calibration `53,67`、final holdout `11,23,41`，每个 final seed 每个条件 20 个 paired episodes，包含 ID、地图平移、速度变化和联合 OOD。只有当命令正常退出且 JSON 存在时，才能引用正式结果。

## 从完整报告生成统计摘要

完整报告生成后，使用仓库内的汇总器计算每 seed 指标和 paired bootstrap 区间：

```bash
pocketworld-summarize-adaptive-horizon \
  artifacts/evaluation-adaptive-horizon-v1.json \
  --output docs/results/evaluation-adaptive-horizon-v1-summary.json \
  --resamples 2000 \
  --bootstrap-seed 20260902
```

正式 v1 的公开资源、文件大小、SHA256 和对应提交见
[adaptive-horizon-v1 evidence release](https://github.com/ZhouYinLong-lab/Pocket-World-Model/releases/tag/adaptive-horizon-v1)。其中 `pocketworld-map-suite-v3-final.pt` 是世界模型 checkpoint，`general-route-sketch-v28-budgeted-locked-distance-field.pt` 仅用于 solver reference；纯学习结论不依赖 A* fallback。

## 复现检查清单

- 记录 Git commit、Python/PyTorch 版本和 checkpoint 路径。
- 检查 `protocol.calibration_and_final_disjoint == true`。
- 检查每个 condition 的 `case_hash` 和 `case_ids` 在所有方法间一致。
- 检查 `json.dumps(report, allow_nan=False)` 成功。
- 区分 `pure_learning`、`astar_fallback` 和 `solver_reference`。
- 失败时保留命令、退出码和 stderr，不用旧结果覆盖新结果。
