# Uncertainty, history, and three-seed validation plan

## Objective

Improve learned single-barrier planning beyond the v0.2.0 focused result by addressing two remaining causes: velocity is not observable from one RGB frame, and collision risk is evaluated only at a point estimate of future state. Validate the resulting planner on three seeds at larger scale.

## Baseline

- Checkpoint: `artifacts/pocketworld-collision-v5.pt`
- Focused protocol: seed 71, 20 episodes, horizon 48, 512 candidates
- Result: 100% imagined success, 75% real success
- Failure concentration: narrow top/bottom passages; failed routes accumulate velocity/position error near boundaries

## Design decisions

### History velocity

- Estimate current pixel velocity from the latest valid agent positions in 2–4 RGB observations.
- Clamp the estimate to the environment speed limit and clear it when recent positions indicate a collision stop.
- Pass the estimate into compact-state rollout and waypoint generation.
- Require at least two frames before overriding velocity; one frame preserves the model encoder fallback. A seed-71 ablation showed that forcing zero from one frame reduced real success from 75% to 55%.

### Uncertainty risk boundary

- Evaluate the learned collision head over a small robust set around each predicted position rather than only the mean state.
- Radius grows with rollout depth to represent accumulated state uncertainty.
- Aggregate with maximum learned collision probability; do not use wall boxes or the explicit pixel collision baseline.
- Keep geometric goal distance, peak collision risk, and planning score as separate metrics.

### Validation

- Tune only on the existing focused seed-71 protocol.
- Final protocol: three fixed seeds, 50 episodes per seed, horizon 48, 1,024 candidates.
- Compare the v0.2 planner against history-only, uncertainty-only, and combined variants on identical episodes.
- Report per-seed rates plus mean/std; do not replace the headline if gains do not survive all seeds.

## Progress log

- 2026-08-03: plan created.
- 2026-08-03: robust collision-state set and RGB-history velocity estimator implemented. One-frame zero-velocity override rejected by ablation (75% → 55%); estimator now activates only with temporal evidence.
- 2026-08-03: focused uncertainty sweep selected open-loop radius `0.5 + 0.05*sqrt(step)`: success stayed at 75%, mean distance improved 10.17px → 9.03px, and collisions fell 1.85 → 1.55.
- 2026-08-03: history-aware route-preserving closed loop reached 85% real success and 6.93px mean distance. A smaller combined boundary (`0.25 + 0.025*sqrt(step)`) retained 85%, reduced mean collisions to 0.30, and was frozen before multi-seed validation.
- 2026-08-03: robust planning changed to point-estimate shortlist followed by 64-candidate robust rescoring, reducing focused combined evaluation from about 49s to 18s without changing its 85% result.

## Acceptance criteria

- History velocity estimator has deterministic unit tests for motion, reset, and clipping.
- Uncertainty aggregation has tests showing risk is at least the point-estimate risk.
- Existing test/coverage gates remain green.
- Three-seed report and exact CLI/config are committed.
