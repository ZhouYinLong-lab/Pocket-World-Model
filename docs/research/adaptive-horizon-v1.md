# Adaptive imagination horizon v1

## Research question

Under a fixed interaction and compute budget, can a calibrated uncertainty
policy reduce real planning failures caused by world-model error by changing
the imagination horizon online?

This is a falsifiable extension of the repository's original imagination-gap
question. The expected result is not assumed to be positive: a shorter horizon
may reduce model exposure while also removing the look-ahead needed to route
around obstacles.

## Important terminology

The repository contains four different mechanisms:

1. **Fixed-horizon ordinary MPC** uses one horizon and the ordinary local
   solver.
2. **Fixed-horizon robust MPC** uses one horizon and robust local scoring.
3. **Adaptive solver gate** is the existing `adaptive_mpc_decision`: it keeps
   the planning horizon fixed and switches between ordinary and robust MPC.
   The v25 result showed computation adaptation, not a reliable safety gain;
   that negative result remains historical evidence.
4. **Adaptive horizon** is the new `AdaptiveHorizonPolicy`: it changes the
   number of imagined steps before a plan is selected. The first comparison
   keeps the learned random-shooting solver fixed; robust MPC is reported only
   as a separate ablation.

The old adaptive solver gate must not be described as adaptive horizon.

## Method

At each replanning point, the evaluator supplies only online-observable
signals: a calibrated cumulative state-uncertainty curve, learned cumulative
collision risk, route alignment error, label-free predictive shift score, and
recent risk pressure. For each candidate horizon `h`, the auditable score is:

```text
R_h = 0.40 U_h + 0.30 C_h + 0.15 A + 0.10 O + 0.05 P
```

`U_h` and `C_h` are clipped to `[0, 1]`; `A` is alignment error divided by the
6 px alignment budget; `O` is the predictive shift score divided by its 2.0
budget; and `P` is recent normalized risk pressure. The policy chooses the
longest candidate with `R_h <= 0.45`, falls back to the shortest candidate if
none is feasible, and uses separate entry (`0.55`) and exit (`0.35`)
thresholds to avoid horizon oscillation. Every decision logs all candidate
risks and its reason.

The default candidates are `(8, 16, 24, 32)` and can be changed in the locked
JSON protocol or with `--horizons 8,16,24,32`. Candidate action banks are
deterministically generated once per paired case and reused across fixed and
adaptive methods, with the selected horizon taking a prefix of the same bank.

The pure-learning track never calls A*, reads wall geometry, or uses future
collision labels. The separate `astar_fallback` track explicitly adds the
existing RGB geometry/A* proposals and must not be attributed to the learned
world model.

## Data split and locked protocol

- Training seeds: the existing training seeds `(101, 103, 107)`.
- Calibration seeds: `(53, 67)`, used to fit the existing held-out Gaussian
  uncertainty scale and inspect calibration metrics.
- Final holdout seeds: `(11, 23, 41)`, used only for the final report.

The evaluator rejects overlapping split sets and writes the complete split to
the JSON report. Thresholds are not fitted from final holdout outcomes. The
formal protocol is stored in
[`configs/adaptive-horizon-v1.json`](../../configs/adaptive-horizon-v1.json).

## Baselines and metrics

The evaluator compares fixed horizons 8/16/24/32, the existing fixed-horizon
adaptive solver gate, calibrated adaptive horizon, and adaptive horizon with
robust MPC as an independent ablation. Each final condition reuses the same
case IDs and action-bank seed for every method. It reports pure learning and
explicit A* fallback separately.

The machine-readable report includes imagined and real success, their gap,
collisions, final distance, route completion, replans, latency, model-query
budget, selected-horizon distribution, switch count, decision reasons,
calibration coverage, Gaussian NLL, collision Brier score, ID/map-shift/fast-
speed OOD conditions, per-seed summaries, and the fixed-horizon position-error
curve.

## Results

The locked three-seed protocol completed with 60 paired episodes per
condition (20 per final seed). The fixed-horizon position error curve on ID
cases increased from **2.34 px at 8 steps** to **5.23 px at 16**, **8.30 px at
24**, and **11.19 px at 32**. Calibration used 192 samples and measured 90%
interval coverage of **98.4% for position** and **91.4% for velocity**; the
collision Brier score was **0.0972**.

The formal result is negative for the strong safety hypothesis. In the pure
learning ID track, adaptive horizon reached **11.7% real success** and
**15.53 collisions/episode**, versus fixed-16's **8.3%** and
**13.65 collisions/episode**. On map-shift, fast-speed, and joint OOD, the
adaptive method reduced model-query cost but did not reduce collisions versus
fixed-16. It selected horizon 8 for 450–524 replanning decisions per
condition, while retaining a small number of long-horizon decisions.

The hybrid A* track had higher success in several conditions, but adaptive
horizon also had more collisions than hybrid fixed-16 and those results must
not be attributed to pure learning. The robust-MPC ablation reduced collisions
in the solver-reference track at roughly 3.2–3.4 seconds planning latency,
compared with roughly 0.46–0.54 seconds for the existing solver gate; this is
an independent solver trade-off, not evidence that adaptive horizon itself is
safe.

The formal JSON is summarized in
[`docs/results/evaluation-adaptive-horizon-v1-summary.json`](../results/evaluation-adaptive-horizon-v1-summary.json).
The appropriate conclusion is: **adaptive horizon provides computation-budget
adaptation and exposes a useful long-horizon trade-off, but this protocol does
not support a safety-improvement claim for obstacle traversal.**

## Limitations and interpretation

The environment is still a deterministic 64×64 two-dimensional simulator,
not a robot or a physics benchmark. The model is a compact deterministic
latent model with calibrated diagonal uncertainty, not a Bayesian posterior.
The route field and A* fallback use observed RGB geometry and are hybrid
references, not pure learned planning. A positive result must preserve success
within a pre-declared tolerance while reducing collision or real-failure rate
against fixed horizon 16 across final seeds. A result that only reduces model
queries is a **computation-budget adaptation**, not a safety improvement.

The principal negative result to preserve is the existing v25 adaptive solver
gate: changing ordinary/robust solver selection at a fixed horizon did not
establish a safety gain. If adaptive horizon also fails to improve real
planning reliability, that is a valid result about this model and protocol.

Reproduce the smoke protocol with:

```bash
python -m pocketworld.evaluate_adaptive_horizon \
  --protocol configs/adaptive-horizon-v1.json \
  --smoke \
  --output artifacts/evaluation-adaptive-horizon-smoke.json
```

The locked formal command is:

```bash
python -m pocketworld.evaluate_adaptive_horizon \
  --protocol configs/adaptive-horizon-v1.json \
  --output artifacts/evaluation-adaptive-horizon-v1.json
```
