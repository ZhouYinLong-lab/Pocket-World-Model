# Learnable temporal velocity and calibrated probabilistic uncertainty

Status: route-aware scoring and online shift monitoring implemented; the main remaining gap is real barrier success and reliable map-OOD detection.

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

### Route alignment and shift monitoring

- Route-aware scoring combines endpoint distance, mean along-route distance, progress regressions, and terminal collision risk.
- The planner can execute a short prefix of a full-route candidate before replanning, avoiding a false choice between long-horizon scoring and expensive open-loop execution.
- Every executed action records the distance between the imagined route position and the RGB-extracted real position.
- An online standardized innovation score compares the observed next position/velocity against the calibrated transition distribution. A threshold is fit from an in-distribution 95th percentile and can trigger immediate replanning.

The uncertainty is intentionally described as a marginal split-calibrated Gaussian approximation. It is not a Bayesian posterior, deep ensemble, or distribution-free joint confidence set.

## Evaluation contract

`pocketworld.evaluate` reports:

- learned temporal velocity MAE/RMSE in pixels;
- the finite-difference baseline and relative gap;
- empirical position and velocity coverage at 50%, 80%, 90%, and 95%;
- 90% interval width and state Gaussian NLL.

It also supports an OOD calibration matrix covering nominal speed, 0.8x and 1.2x speed, a changed wall map, and the joint changed-map/1.2x-speed condition. Each condition uses fresh rollouts and reports per-seed 90% position/velocity coverage.

`pocketworld.evaluate_collision` reports the new `learned_velocity_probabilistic_closed` planner beside the previous four variants. The intended final protocol is 50 episodes per seed, seeds `71,83,97`, horizon 48, and 1,024 candidates, followed by a separate OOD calibration check.

## Current evidence

The v8 development checkpoint reaches 0.407–0.450px learned velocity MAE across three 50-episode seeds, compared with 0.215–0.283px for the privileged finite-difference baseline. Fresh-seed 90% marginal coverage is 86.4–88.4% for position and 89.5–91.0% for velocity; the held-out calibration split itself is calibrated by construction. The machine-readable slice is [evaluation-temporal-probability-v8.json](../results/evaluation-temporal-probability-v8.json).

The completed full barrier run uses 50 episodes per seed, seeds `71,83,97`, horizon 48, and 1,024 candidates. The learned temporal/probabilistic planner reaches 92.7% ± 1.9pp imagined success, 0% real success, 31.33 ± 0.06px final distance, and 0.547 ± 0.025 collisions per episode. Its low collision count is stable, but the imagined detours do not cross the real barrier reliably. See [evaluation-temporal-probability-probabilistic-v8-full.json](../results/evaluation-temporal-probability-probabilistic-v8-full.json).

The OOD matrix shows nominal position/velocity coverage of 87.7%/90.8%, improving to 90.1%/96.2% under 0.8x speed, but falling to 83.8%/82.8% at 1.2x speed, 80.9%/87.2% on the changed map, and 79.0%/81.5% under the joint shift. This establishes a calibration stability boundary: the current diagonal Gaussian is useful near the training regime, but should not be treated as OOD-safe without a shift detector or recalibration. See [evaluation-temporal-probability-ood-v8.json](../results/evaluation-temporal-probability-ood-v8.json).

The route alignment diagnostic reports about 2.48px mean imagined-vs-real position error and 7.12px maximum error, but still 0% real barrier success while collisions rise to 6.45/episode in the low-cost route-aware run. See [evaluation-route-alignment-sanity-v8.json](../results/evaluation-route-alignment-sanity-v8.json). The shift detector fits a 0.9813 threshold with 5.25% ID alarms. After adding the visible wall-context gate, it catches 88% changed-map and 86% joint map/fast transitions (AUROC 0.98/0.97), while fast-speed-only detection remains 9%. See [evaluation-temporal-probability-shift-detection-v8.json](../results/evaluation-temporal-probability-shift-detection-v8.json).

The next research question is how to align route-level imagined progress with real barrier crossing while improving map-OOD detection. Candidate follow-ups are route-conditioned collision supervision, latent feature density or conformal scores, and uncertainty-aware replanning triggers that distinguish ordinary model noise from a genuine map shift.
