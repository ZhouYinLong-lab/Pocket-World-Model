# Learnable temporal velocity and calibrated probabilistic uncertainty

Status: implementation complete; large-scale three-seed validation remains the next experiment.

## Objective

Replace the previous hand-designed history velocity estimate and fixed robust uncertainty radius with learned, inspectable components:

1. infer velocity from a short RGB history using a trainable temporal representation;
2. predict transition uncertainty as a probability distribution;
3. calibrate that distribution on held-out rollouts;
4. use the calibrated distribution in collision-aware planning without making every candidate expensive.

## Design

### Temporal velocity representation

- The last four observations are encoded by the existing image encoder and a dedicated RGB motion encoder.
- The temporal sequence contains both latent frames and adjacent latent differences.
- A GRU predicts normalized velocity mean and an auxiliary scale.
- A small position head is trained from RGB positions so the motion pathway cannot ignore the small mint agent.
- The old pixel-position finite-difference estimator remains a privileged diagnostic baseline.
- `--temporal-only` freezes the world model and fine-tunes the temporal pathway independently.

### Calibrated probability

- The state transition head predicts diagonal standard deviations for normalized `(x, y, vx, vy)`.
- Training uses a Gaussian residual objective in addition to the existing state and image losses.
- A held-out validation rollout set estimates one scale per coordinate from the requested residual quantile.
- The planner accumulates diagonal transition variance, samples future landing states, and averages learned collision-event probabilities over those samples.
- Candidate ranking is point-based first; only the robust shortlist receives probabilistic rescoring.

The uncertainty is intentionally described as a marginal split-calibrated Gaussian approximation. It is not a Bayesian posterior, deep ensemble, or distribution-free joint confidence set.

## Evaluation contract

`pocketworld.evaluate` reports:

- learned temporal velocity MAE/RMSE in pixels;
- the finite-difference baseline and relative gap;
- empirical position and velocity coverage at 50%, 80%, 90%, and 95%;
- 90% interval width and state Gaussian NLL.

`pocketworld.evaluate_collision` reports the new `learned_velocity_probabilistic_closed` planner beside the previous four variants. The intended final protocol is 50 episodes per seed, seeds `71,83,97`, horizon 48, and 1,024 candidates, followed by a separate OOD calibration check.

## Current evidence

The v8 development checkpoint reaches 0.407–0.450px learned velocity MAE across three 50-episode seeds, compared with 0.215–0.283px for the privileged finite-difference baseline. Fresh-seed 90% marginal coverage is 86.4–88.4% for position and 89.5–91.0% for velocity; the held-out calibration split itself is calibrated by construction. The machine-readable slice is [evaluation-temporal-probability-v8.json](../results/evaluation-temporal-probability-v8.json). These numbers are diagnostic rather than headline results until the three-seed barrier-planning protocol is complete.

The remaining questions are whether the learned velocity path improves real barrier success over the hand-designed history path, and whether calibration remains reliable under changed walls, speed, and planning horizon.
