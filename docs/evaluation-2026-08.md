# PocketWorld evaluation — 2026-08-03

The current main checkpoint is trained with 1,000 trajectories, 200 validation trajectories, 16-step unrolls, sticky actions (`p=0.75`), and full-map start/goal sampling. The report was generated with 50 episodes per seed, 3 seeds (`11, 23, 41`), and 1,024 random-shooting candidates.

The collision-supervised barrier-mix checkpoint is an ablation, trained with the same scale plus `--barrier-probability 0.25`. Its quick report uses 10 episodes per seed and 512 candidates, so it is not used to replace the larger main-checkpoint headline numbers.

## Current result

| Metric | In-distribution | OOD |
| --- | ---: | ---: |
| 1-step latent position error | 1.22 px | 0.99 px |
| 5-step latent position error | 1.86 px | 1.59 px |
| 10-step latent position error | 2.50 px | 2.49 px |
| 20-step latent position error | 3.94 px | 4.19 px |
| 20-step image MAE | 0.0468 | 0.0704 |

Planning now uses a 16-step default horizon, with a sweep to expose imagination drift:

| Horizon | Imagined success | Real success | Gap |
| ---: | ---: | ---: | ---: |
| 8 | 0% | 0% | 0pp |
| 16 | 98% | 96% | 2pp |
| 24 | 100% | 93% | 7pp |
| 32 | 100% | 93% | 7pp |

The repeated-action diagnostic averages 4.04 px position error across up/down/left/right. This is the clearest evidence that the compact model is now action-sensitive; the previous sticky-only model had errors of roughly 9–35 px and could not produce meaningful imagined planning.

## Renderer and collision follow-up

The renderer checkpoint adds a separate agent-mask head and a state-conditioned RGB composition path. Raw RGB decoding still loses the small agent, but the composited output has 100% position coverage and 1.69/2.32/3.15/3.94px position error at 1/5/10/20 steps in a 20-episode check. The raw decoder remains available as the honest pixel baseline; the composited output is the deployable visualization path.

The learned mask head is useful as a diagnostic but is not yet a good shape decoder: its 20-episode mask IoU is about 0.03–0.04 ID/OOD, despite 100% thresholded coverage after focused fine-tuning. This is why the final RGB path uses the structured position plus the known circular agent geometry rather than claiming that the mask head has solved pixel rendering.

The original pure learned collision planner reached roughly 93% imagined / 0% real success on the 20-episode barrier check. A first `hybrid` planner added a local wall-patch guard and post-collision freeze; it reached about 5% real success and reduced mean final distance to about 25px, while the explicit pixel-wall baseline remained the stronger obstacle reference. That negative result motivated the focused follow-up below.

Next work should focus on wall-relative state transitions or a closed-loop planner that learns from collision-free waypoint progress. The current RGB deployment path, ONNX position contract, and negative collision result are already reproducible.

The barrier evaluation compares the unconstrained planner with a collision-aware planner that extracts wall pixels from the current observation and penalizes trajectories after the first wall intersection. This is an explicit planning baseline; it is not counted as learned wall dynamics until the compact state model predicts collision events itself.

The first barrier result was intentionally a negative result: the unconstrained planner reached 100% imagined success but 0% real success. Structured top/bottom detour proposals now raise collision-aware open-loop success to 10% on the same 10-episode, 40-step check and reduce mean final distance from 28.18px to 21.34px. At horizon 64, the mean distance falls further to 17.54px, but success remains only 10%.

The implementation now includes `receding_horizon_plan`, which replans from the latest real observation after every action. Its barrier result is tracked separately from the open-loop baselines so closed-loop correction can be measured without changing the headline planning numbers.

The quick closed-loop check still reached 0% real success: 1-step replanning averaged 28.19px, 4-step commitment averaged 27.91px, and route-preserving replanning averaged 27.72px. All remain worse than executing the best structured detour open-loop (21.27px in the same check), confirming that the missing piece is collision-state modeling rather than route-cache lifetime. The route-preserving mode caches the imagined path and only replans on collision or >6px route deviation; its result is reported separately as `collision_aware_route`.

The collision-supervised barrier-mix ablation reached 81.9% ID collision accuracy / 69.4% recall and 76.9% OOD accuracy / 56.1% recall. Its learned-collision planner still achieved 100% imagined but 0% real success on the barrier challenge, while the pixel wall baseline reached about 5% real success. This is a useful negative result: event labels alone do not make the compact state wall-relative.

A follow-up added a 7x7 wall-relative patch around the predicted landing point to the collision head. It improved the quick OOD collision accuracy to 82.9% / 65.9% recall, but learned-collision planning remained 100% imagined / 0% real. The subsequent post-collision response therefore changed the imagined state, rather than only classifying the event.

The post-collision response is now implemented: predicted events freeze position and clear velocity in learned imagined rollouts. The pure learned barrier result remains approximately 93–95% imagined / 0% real, so the missing issue is event localization on detour candidates, not the response rule itself. The hybrid visual guard is reported as a separate engineering baseline.

## Engineering completion checks

- Repository metadata: MIT license committed; GitHub topics set for world models, model-based RL, PyTorch, Gymnasium, and machine learning.
- Python verification: `25 passed`, with a 70% CI coverage gate (current coverage: 74%).
- ONNX export: verified against `pocketworld-renderer-v5.pt`; the compatible composited RGB model is written successfully.
- ONNX contract: the exported graph exposes `next_observation[batch,3,64,64]` and `next_position[batch,2]`; the browser prefers the supervised position channel for its model marker.
- Web verification: `npm install --no-audit --no-fund` followed by `npm run build` succeeds. Vite reports only a bundle-size warning for the ONNX Runtime WASM asset.

## Learned-collision follow-up

The focused follow-up separates three previously entangled errors:

1. A collision-seeking curriculum raises useful barrier-event density from about 10.8% to 47–60% while retaining safe approach negatives and corner/boundary examples.
2. The collision-probability rollout now freezes state after a predicted impact, and the planner ranks map-agnostic two-bend routes by continuous peak risk rather than a brittle binary threshold.
3. A system-identification stage learns only acceleration, friction, and speed limit from collision-free transitions. The learned values moved from approximately `0.188/0.807/0.647` to `0.242/0.813/0.762`, close to the simulator's normalized `0.250/0.840/0.767` values.

On the same 20-episode single-barrier distribution (`seed=71`, horizon 48, 512 candidates), `pocketworld-collision-v5.pt` reaches **100% imagined success / 75% real success**. Mean collision count is 1.85 per episode, concentrated in the five failed routes; successful routes generally complete without wall collisions. This result was used only for focused tuning before the frozen validation below.

## History and uncertainty validation

The next planner adds two scoped mechanisms without changing the checkpoint:

- A velocity estimator uses the latest 2–4 RGB agent positions to initialize the compact velocity state. It activates only when at least two frames exist; forcing zero velocity from one reset frame reduced the focused result from 75% to 55% and was rejected.
- A robust collision boundary evaluates the learned collision head at the mean landing position and four nearby positions. Radius grows with `sqrt(step)`. To control cost, all candidates receive point scoring and the best 64 candidates, including every structured waypoint proposal, receive robust rescoring.

Focused tuning on seed 71 selected `0.5 + 0.05*sqrt(step)` pixels for open uncertainty and `0.25 + 0.025*sqrt(step)` for history-aware closed loop. Parameters were then frozen.

The final protocol uses 50 episodes for each of seeds 71, 83, and 97, horizon 48, and 1,024 candidates. Results are means across seed-level rates:

| Planner | Imagined success | Real success | Final distance | Collisions |
| --- | ---: | ---: | ---: | ---: |
| Point open | 98.7% ± 0.9pp | 64.0% ± 0.0pp | 10.77 ± 1.21px | 1.89 ± 0.17 |
| Uncertainty open | 87.3% ± 2.5pp | 61.3% ± 5.0pp | 11.48 ± 0.65px | 1.48 ± 0.15 |
| History closed | 98.7% ± 0.9pp | 73.3% ± 0.9pp | 9.77 ± 0.32px | 0.63 ± 0.03 |
| History + uncertainty | 96.0% ± 1.6pp | **76.7% ± 2.5pp** | **9.31 ± 0.83px** | **0.56 ± 0.06** |

Combined per-seed real success is 80% / 76% / 74%, compared with 64% / 64% / 64% for point-open planning. The combined planner improves real success by 12.7 percentage points and cuts mean collisions by about 70%. Uncertainty alone is not beneficial to success, so the evidence supports its use as a boundary inside history-aware closed-loop recovery rather than as a standalone conservative planner.

The full machine-readable report is committed at [`docs/results/evaluation-collision-v3.json`](results/evaluation-collision-v3.json). The remaining gap is a 19.3pp combined imagined-versus-real success difference, now concentrated in narrow-passage recovery and model calibration rather than missing temporal state.
