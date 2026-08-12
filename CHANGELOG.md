# Changelog

All notable changes to PocketWorld are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses semantic versioning.

## [0.24.0] - 2026-08-12

### Added

- Route-progress and remaining-budget metrics for learned-field planning.
- A locked `distance_field_budgeted_hybrid_mpc` ablation with explicit RGB/A*
  fallback, cooldown, and trigger diagnostics.
- OOD speed/map-shift evaluation for the route-budgeted hybrid method.

### Results

- On the medium three-seed obstacle holdout, locked hybrid planning reached
  100.0% real success and 0.067 collisions/episode versus 66.7% and 0.600 for
  learned-field MPC, with 1.33 A* calls/episode.
- The improvement has a substantial latency cost (about 5.01 s versus 0.74 s
  per episode in the same run), so this is reported as a reliability/latency
  trade-off rather than a pure learned-planner win.
- In the focused OOD matrix covering nominal/±2 px wall shifts and speed
  0.75/1.25, the hybrid reached 100% success in all six conditions; the
  baseline ranged from 33.3% to 77.8%.

See [the v28 OOD report](artifacts/evaluation-general-ood-v28-budgeted.json).

## [0.26.0] - 2026-08-12

### Added

- A disjoint calibration protocol for the gated robust route hybrid.
- Final and OOD comparisons between fast hybrid and gated robust hybrid.

### Results

- Calibration seeds `53,67` produced identical outcomes for thresholds
  `0.25`–`0.65`; the selected `0.25` threshold is therefore not evidence of
  calibrated risk.
- On the final holdout, gated robust planning remains at 100% success but is
  slightly worse than fast hybrid in collisions (0.433 vs 0.400) and latency
  (0.858 s vs 0.679 s).
- Across six OOD speed/map conditions, both methods remain at 100% success;
  gated robust improves only one condition and is otherwise tied or worse.
  The negative result is retained as a boundary of the current observable risk
  score.

## [0.27.0] - 2026-08-12

### Results

- Scaled fixed-checkpoint holdout: 3 seeds × 20 episodes. Learned-field MPC
  reaches 81.67% success and 0.883 collisions/episode; fast budgeted hybrid
  reaches 100.0% and 0.500.
- Scaled OOD replication uses six speed/map conditions and 57–60 paired
  episodes per condition. Fast hybrid reaches 100.0% in every condition;
  baseline ranges from 63.16% to 93.33%.
- The scaled result strengthens the route-completion conclusion while retaining
  the explicit A* intervention and collision/latency trade-off.

## [0.25.0] - 2026-08-12

### Added

- A matched fast-vs-robust route-budget ablation. Both variants share the same
  route progress, remaining-budget, wall confirmation, cooldown, and route-lock
  logic; only the post-fallback local MPC robustness differs.

### Results

- Ordinary-MPC hybrid reaches 100.0% success with 0.400 collisions/episode and
  0.61 s planning, while robust-MPC hybrid reaches 100.0% with 0.133
  collisions/episode and 9.36 s planning on the same holdout.
- The fast hybrid keeps 100% success across the six-condition OOD smoke matrix,
  with 0.22–0.67 collisions/episode and 0.13–0.19 s planning time.
- The project now exposes a reliability/latency Pareto comparison instead of
  presenting the expensive robust fallback as the only route-level solution.

## [0.20.0] - 2026-08-12

### Added

- Balanced route-family coverage controls and a three-condition density study:
  two families/600 total samples, four families/600, and four families/1200.
- A paired planner-comparison runner with one shared holdout, RGB/A* geometric
  reference, learned field projection, MPC, robust MPC, and guarded MPC.
- Explicit negative-result reporting for clearance-field and guarded-MPC
  variants instead of selecting only the best method.

### Results

- Four-family/600 training is the strongest learned setting in the paired
  coverage study: distance-field MPC reaches 95.0% real success with 0.367
  collisions per episode.
- Robust MPC keeps the same 95.0% success and reduces collisions to 0.150,
  but raises mean planning time to 2.80 seconds versus 255 ms for MPC.
- Increasing the same four-family training budget to 1200 total samples does
  not improve success (93.3%); coverage and per-family density interact, so
  this is not evidence for monotonic scaling.
- Under speed 1.25, ordinary MPC keeps 95.0% nominal success and 78.9% under
  walls-x-plus-one, while robust MPC reduces nominal/OOD collisions to 0.433 /
  1.053 but reaches 93.3% / 73.7% success. Robustness is therefore a risk
  trade-off, not a uniformly stronger planner.
- The guarded-MPC ablation falls to 66.7% success, while clearance training is
  indistinguishable from the base field on this holdout.
- v25 adds a calibration-split adaptive MPC gate. It selects entry 0.25 and
  exit 0.1667 on disjoint seeds `53,67`; on the final holdout it reaches 95.0%
  success with 0.450 collisions and 304 ms planning, versus ordinary MPC's
  0.367 collisions / 178 ms and fixed robust MPC's 0.150 / 1.98 s.
- The adaptive gate is therefore an efficiency result, not a safety win: it
  reduces robust calls by over 90% but does not reliably lower collisions under
  map/speed shifts. This negative result is retained as the current research
  boundary for risk calibration.

See the [v23 coverage report](docs/results/evaluation-coverage-study-v23.json)
and [v23 planner comparison](docs/results/evaluation-planner-comparison-v23.json).
The paired OOD matrix is in
[evaluation-coverage-study-v23-ood.json](docs/results/evaluation-coverage-study-v23-ood.json).

The adaptive-gate calibration and v25 paired reports are in
[evaluation-adaptive-calibration-v25.json](docs/results/evaluation-adaptive-calibration-v25.json),
[evaluation-planner-comparison-v25.json](docs/results/evaluation-planner-comparison-v25.json),
and [evaluation-planner-comparison-v25-ood.json](docs/results/evaluation-planner-comparison-v25-ood.json).

## [0.23.0] - 2026-08-12

### Added

- Per-horizon temperature calibration for the learned collision head.
- Independent calibration-map holdout with reliability bins, Brier score, ECE,
  AUROC/AUPRC, and threshold operating-point metrics.
- Raw-versus-calibrated planner comparison on the unchanged final holdout.

- Simulator-labelled, route-conditioned short-horizon collision probability
  head for 1/2/4-step risk prediction.
- Disjoint train/calibration/final-holdout protocol and calibrated robust-MPC
  gate; the learned head never receives future state or evaluation labels.
- Complete nominal and map/speed OOD reports with per-family risk, switch,
  robust-call, collision, success, and planning-time metrics.

### Results

- On the shared 60-task holdout, corrected v26 reaches 95.0% success, 0.317
  collisions/episode, and 544 ms planning, versus ordinary MPC's 0.367/178 ms
  and fixed robust MPC's 0.150/1.98 s.
- Under walls+1, it reaches 0.404 collisions at speed 1.0 and 0.754 at speed
  1.25; this improves collision rate over the heuristic adaptive gate and
  ordinary MPC, but success remains lower than ordinary MPC at speed 1.0.
- The first v26 run selected threshold 0.0 due to an ascending-threshold bug
  and is intentionally not reported; the corrected run selects 0.2051 using
  the highest threshold satisfying 80% calibration recall.

### Results

- On the independent calibration holdout (`29,31,37`), horizon-2 ECE changes
  from 0.02476 to 0.02241 and Brier score from 0.062274 to 0.062266;
  horizon-2 AUROC remains 0.927.
- On the untouched final `11,23,41` holdout, raw and calibrated gates both
  reach 95.0% success and 0.183 collisions/episode. This is a small
  calibration improvement, not a claim that temperature scaling solves OOD
  obstacle traversal.

See the [v27 calibration report](docs/results/evaluation-collision-head-v27-final.json),
the [v26 nominal report](docs/results/evaluation-collision-head-v26-corrected.json)
and [v26 OOD report](docs/results/evaluation-collision-head-v26-ood.json).

## [0.17.0] - 2026-08-12

### Added

- RGB-only short-horizon inertial MPC for the learned route-field executor.
- Exact continuous collision geometry shared with `PocketWorldEnv`, avoiding
  an over-conservative square-dilation safety model.
- Focused method selection via `--methods` and per-episode MPC call/override
  metrics for reproducible ablations.

### Results

- On the fixed 600-task train / 60-task holdout protocol, distance-field beam
  + RGB projection reaches 88.33% success with 2.20 collisions/episode.
- The new distance-field beam + RGB inertial MPC reaches 91.67% success with
  0.383 collisions/episode, without A* at evaluation time.
- The guarded-MPC trigger is retained as a transparent negative ablation:
  75.00% success and 4.90 collisions/episode. Sparse safety overrides are not
  enough when the trigger and the replacement controller use different local
  objectives.

See the [v19 machine-readable report](docs/results/evaluation-general-routes-v19-mpc.json).

## [0.18.0] - 2026-08-12

### Added

- Fixed-checkpoint MPC ablation runner covering horizon, beam width, RGB vs
  action-fused velocity, and robust velocity-envelope conditions.
- Family-level summaries, planning-time metrics, and 95% episode-level
  uncertainty intervals.

### Results

- H=6/B=12 RGB MPC reaches 93.3% success on the fixed 60-task holdout.
- Action-fused H=4/B=12 reaches 90.0% success with 0.250 collisions/episode.
- Robust H=4/B=12 reaches 0.117 collisions/episode but costs about 3.4 seconds
  of CPU planning per episode and lowers multi-channel success to 71.4%.

See the [v20 machine-readable ablation](docs/results/evaluation-mpc-ablation-v20.json).

## [0.19.0] - 2026-08-12

### Added

- General-route OOD evaluator for paired speed and deterministic wall shifts.
- Train-only coarse wall-signature calibration and explicit shift-aware MPC/A*
  fallback comparison.
- Reachability exclusions, shift detection rate, fallback calls, and family
  breakdowns in the machine-readable report.

### Results

- RGB MPC remains effective under speed 1.25: 90.0% success and 0.717
  collisions/episode versus 81.7% and 13.883 for the v18 RGB projection
  baseline.
- On `walls_x_plus1 @ speed1.0`, pure RGB MPC reaches 75.4% success and 0.596
  collisions; shift-aware MPC/A* fallback reaches 78.9% and 0.404.
- The fallback is explicitly hybrid and is not counted as pure learned
  planning; detector triggers also occur on nominal holdout family shifts.

See the [v21 OOD report](docs/results/evaluation-general-ood-v21.json).

## [0.16.0] - 2026-08-12

### Added

- General procedural obstacle families: staggered blocks, multi-channel walls,
  staircases, and offset L-shaped obstacles.
- A learned coarse 16×16 route-distance field and fixed-width beam executor.
- Same-protocol comparisons for route sketch, RGB projection, distance field,
  RGB guard, learned-to-A* fallback, and pure RGB/A* reference methods.
- Edge-level RGB collision checks for coarse-grid transitions and separate
  route-sketch/distance-field checkpoints.

### Results

- On 600 training tasks and 60 held-out tasks across three seeds, continuous
  route sketch reaches 6.7% real success; the learned distance field reaches
  50.0%; and distance field + beam + RGB guard reaches 88.3%.
- The learned-to-A* fallback reaches 100.0% with 0.217 collisions/episode;
  pure RGB/A* reaches 100.0% with 0.067 collisions/episode.
- The multi-channel holdout remains difficult for the best pure learned method:
  78.6% success and 5.93 collisions/episode.
- A conservative one-step shield is published as a negative ablation: 63.3%
  success and 40.62 collisions/episode due to no-action-safe corners under the
  discrete inertial dynamics.

These numbers establish a broader representation study, not a claim that pure
learned general obstacle navigation is solved.

## [0.15.0] - 2026-08-12

### Added

- Deterministic procedural route tasks with two-to-four vertical barriers,
  varied gap widths, and distinct train/holdout endpoint distributions.
- Continuous route-sketch policy (v16) and structured per-barrier gap route
  policy (v17) for a direct representation-level comparison.
- RGB visible-gap feasibility projection and a paired raw-vs-projected
  evaluation, with explicit no-target-leakage protocol fields.
- 105-test regression suite covering procedural sampling, feature contracts,
  projection bounds, policy round-trips, and the v17 evaluation contract.

### Results

- On 600 training tasks and 60 held-out tasks across three evaluation seeds,
  the honest raw v17 student reaches 83.3% real success and 1.417 collisions
  per episode.
- Adding the observable RGB feasibility projection reaches 100.0% real success
  and 0.367 collisions per episode; four-barrier raw success is 65.5%, while
  projected success is 100.0%.
- The projection result is explicitly reported as a structured learned
  predictor plus RGB safety layer, not as pure end-to-end learned navigation.

## [0.14.0] - 2026-08-12

### Added

- Action-level RGB route policy with DAgger aggregation and a strict no-A* evaluation path.
- Route-level RGB policy that predicts `direct`, `top`, `bottom`, or `gap`, followed by an observable waypoint controller.
- Same-protocol comparison reports for action imitation, route-mode distillation, and the geometry-locked hybrid reference.

### Results

- Under the three-seed, four-layout, 240-task protocol, action-level distillation reaches 59.2% real success with 17.63 collisions per episode.
- Route-level distillation reaches 100.0% real success with 0.042 collisions per episode, while using no A* during student evaluation.
- The result isolates route commitment as a stronger distillation target than next-action imitation for this benchmark; it does not claim general learned obstacle navigation because the mode vocabulary and waypoint executor are benchmark-specific.

## [0.13.0] - 2026-08-12

### Added

- Backward-compatible 9D baseline and 17D RGB map-aware route feature contracts.
- A resumable `pocketworld-evaluate-route-variants` experiment runner.
- Three-seed comparison across shifted barriers, narrow/wide gaps, and speed shifts.

### Results

- Map context raises mean imagined route success by 7.5pp and lowers collisions by 0.36 per episode in the published eight-condition matrix.
- Mean real route success changes by 0.0pp, so map-aware route classification is a diagnostic/safety improvement, not a solved obstacle-crossing planner.

### Hybrid execution follow-up

- Added geometry-locked adaptive route control: short visible detours can use diagonal A* with longer lookahead, while long barrier detours keep conservative cardinal control.
- Separated geometry-planning calls from learned model-query counts in closed-loop reports.
- On the four-map, three-seed, 64-step execution matrix, the adaptive hybrid reaches 100.0% / 100.0% / 98.3% / 98.3% real success on single, shifted, narrow-gap, and wide-gap barriers, with 0.650 / 0.233 / 0.067 / 0.017 collisions per episode.
- This is explicitly a visible-geometry hybrid result; learned model queries are zero in this route override and it is not presented as pure world-model obstacle navigation.

## [0.12.1] - 2026-08-12

### Fixed

- Synchronized the published OOD evaluator sources with the v0.12.0 reports.
- Aligned the package version with the v0.12.1 reproducibility patch release.

## [0.12.0] - 2026-08-12

### Added

- Resumable soft-RGB penalty and OOD safety sweep runner.
- Explicit shifted-barrier, narrow-gap, wide-gap, and speed-shift evaluation scenarios.
- Separate in-distribution penalty-selection and OOD holdout reports.
- Fixed planner-name random streams so subset sweeps preserve the full-tournament candidate randomness.

### Results

- Penalty 16 is selected on the original barrier by a predeclared success-first rule: 71.7% real success, 2.02 collisions/episode, +10.0pp paired improvement.
- On six held-out map/velocity conditions, the frozen penalty never beats learned collision; paired deltas range from 0.0pp to −25.0pp.
- The gap-task sampler was corrected so starts and goals lie outside the wall footprint on opposite sides; earlier gap numbers from the invalid protocol are not reported.

## [0.11.0] - 2026-08-12

### Added

- Comparable route safety mechanisms: joint learned-risk/RGB gating, RGB-only hard gating, and soft RGB wall penalties.
- A fixed-predictor `pocketworld-evaluate-safety-methods` entry point for separating predictor training from planner ablations.
- Formal three-seed safety-method report with paired success deltas, collision counts, RGB imagined-wall diagnostics, and query budgets.

### Results

- Soft RGB penalties reach 71.7% real success and 2.02 collisions/episode, versus 61.7% and 2.47 for learned collision.
- RGB-only hard gating reaches 65.0% and 2.02 collisions; joint gating reaches 65.0% and 2.20 collisions.
- The soft method improves by +10.0pp paired success, with per-seed deltas of +20, +5, and +5pp; this is promising but not yet a statistically stable generalization claim.

## [0.10.0] - 2026-08-12

### Added

- RGB wall-mask safety gate as an isolated route-completion planner ablation.
- Selected-route RGB collision diagnostics in planner tournament reports.
- Formal three-seed barrier comparison covering learned collision, route completion, RGB safety gating, route-completion MPC, and route-aware hybrid.
- Machine-readable v7 result and route predictor checkpoint.

### Changed

- Route feasibility handling now combines learned-risk and visible-wall gates when their intersection is supported, with a bounded fallback when calibration makes the intersection empty.

### Results

- The RGB safety gate reaches 65.0% real success and 2.20 collisions/episode, versus 61.7% and 2.47 for learned collision; the paired gain is +3.33pp and remains seed-sensitive.
- Route completion without the gate reaches 53.3%, while route-completion MPC reaches 6.7%; the high-query route-aware hybrid remains 100.0%.

## [0.9.0] - 2026-08-12

### Added

- Closed-loop route-completion MPC comparison with short-prefix replanning and alignment fallback.
- Query-cost accounting for route-completion MPC.
- Formal negative result showing that repeated online replanning amplifies a misspecified route probability.

### Changed

- Route-completion documentation now separates one-shot ranking, risk-gated ranking, hard-negative mining, and closed-loop MPC.

## [0.8.0] - 2026-08-12

### Added

- Explicit route-conditioned completion predictor trained from real simulator route outcomes.
- Three-seed comparison of route-completion weights, risk gating, and hard-negative mining.
- Held-out route-candidate AUROC/Brier metrics and paired planner win/loss/tie deltas.
- Documented negative result showing that strong candidate classification does not guarantee real obstacle traversal.

### Changed

- Route completion can be injected into one-shot planning with an explicit score weight and collision-risk gate.
- Planner reports now include predicted route-completion probability for selected plans.

## [0.7.0] - 2026-08-12

### Added

- Three-checkpoint ensemble collision-risk aggregation using mean plus member disagreement.
- Split-conformal upper collision-risk calibration with held-out coverage reporting.
- A paired three-seed barrier study comparing single-model, ensemble, and conformal risk methods.
- Explicit negative-result documentation showing that route-level conformal coverage can be safe but unusable when calibration routes are all failures.

### Changed

- Planners can inject a collision-risk model independently from the primary dynamics model, keeping method comparisons controlled.
- Full verification now covers 69 tests.

## [0.6.0] - 2026-08-11

### Added

- Discrete Beam Search and collision-aware CEM planner variants.
- Six-method, three-seed planner tournament covering open-space and single-barrier tasks.
- Explicit negative result showing that search improvements do not repair a misspecified obstacle model.
- Calibrated collision-risk budget ablation for CEM, with a machine-readable negative result.

### Changed

- Planner reports now expose the beam expansion budget and separate one-shot model queries from repeated closed-loop route-control queries.

## [0.5.0] - 2026-08-11

### Added

- Discrete categorical CEM planner with explicit candidate-budget accounting.
- Paired open-space and single-barrier planner tournament covering Random Shooting, CEM, learned collision planning, and route-aware hybrid control.
- Machine-readable per-seed planner comparisons with estimated planning calls and model queries.

### Changed

- The planner comparison documents a controlled negative result: CEM improves imagined search quality but does not repair obstacle-model error; route-aware closed-loop geometry remains the strongest barrier controller.

## [0.4.0] - 2026-08-11

### Added

- v3-final maturity-gate checkpoint with learned kinematics identification, formal imagined-versus-real planning-gap evaluation, and browser-ready ONNX export.
- Mature-window RGB/action speed-response shift detector with three-seed OOD stability measurements.

### Changed

- Formal open-space imagination gap is 4.7pp at 24 steps and 6.7pp at 32 steps on the v3-final checkpoint.
- Map-suite v3 reduces 20-step position error to 3.02px train / 3.72px unseen and reaches 98.9% / 93.3% two-waypoint task success across three seeds.
- The maturity report now covers 90% uncertainty coverage and speed/map shift AUROC rather than reporting map-only calibration.

## [Unreleased]

### Added

- Fixed-budget planner tournament with categorical CEM, paired task generation, open-space and single-barrier protocols, and machine-readable comparisons.
- Learnable RGB/latent temporal velocity representation with motion encoder, latent deltas, auxiliary position supervision, and `--temporal-only` fine-tuning.
- Diagonal probabilistic transition uncertainty with held-out residual-quantile calibration and shortlist Monte Carlo collision risk.
- Temporal velocity and empirical uncertainty-coverage metrics in the evaluation report.
- RGB-history velocity estimation for compact-state rollout initialization.
- Horizon-growing learned uncertainty boundaries with robust shortlist rescoring.
- Reproducible three-seed collision planner evaluation and committed result data.
- OOD calibration matrix for nominal, slow, fast, changed-map, and joint speed/map rollouts.
- Full 50-episode × three-seed revalidation for the learned temporal/probabilistic barrier planner.
- Route-aware candidate scoring, executed-prefix alignment metrics, and online predictive-shift alarms.
- Machine-readable route-alignment and shift-detection diagnostics for the v8 checkpoint.
- Explicit route-completion probability and 6px alignment-triggered wall-aware hybrid fallback.
- RGB-observed footprint-inflated A* global routes with bend-only inertial tracking for obstacle-crossing fallback.
- Route-side commitment, remaining-budget penalties, geometric progress tracking, and emergency side-switch diagnostics.
- Early 4px alignment-triggered fallback configuration to keep route distance compatible with the 48-step execution budget.
- Named `train`/`holdout`/`all` map suites with double-barrier, cross, zig-zag, and open layouts.
- Sequential waypoint task generation and `pocketworld-evaluate-generalization` for unseen-layout and multi-goal evaluation.
- Formal map-suite checkpoint trained from v8 on 1,000 episodes across three named training layouts.
- `pocketworld-evaluate-maturity` for three-seed uncertainty calibration and OOD shift gates.
- RGB-only four-action route follower with visible landing safety guard and a route-budget-preserving generalization evaluator.

### Changed

- The planner can compare learned temporal velocity plus calibrated probabilistic uncertainty against the existing finite-difference and robust-radius baselines.
- History-aware uncertainty planning reaches 76.7% ± 2.5pp real barrier success across 150 episodes, versus 64% for point-open planning.
- Mean collisions fall from 1.89 to 0.56 per episode in the three-seed comparison.
- Python verification now covers 54 tests at 79% line coverage.
- The A* fallback raises the strict 150-episode barrier result to 30.67% ± 2.49pp real success, with 3.19 ± 0.52 collisions per episode.
- Locked-budget A* fallback raises the same result to 34.67% ± 6.80pp real success, with 2.03 ± 0.35 collisions and zero emergency side switches per episode.
- Early alignment fallback raises the same result to 50.67% ± 2.49pp real success, with 6.88 ± 0.14px final distance, 1.14 ± 0.23 collisions, and 7.37px remaining route distance.
- Formal map-suite evaluation reaches 4.85px train / 4.84px unseen 20-step position error; two-waypoint task success remains 6.67% / 5.00%.
- v2 collision-aware three-seed map-suite evaluation reaches 3.94 ± 0.27px train / 4.13 ± 0.30px unseen 20-step position error and 97.8% / 93.3% two-waypoint task success.
- Three-waypoint stress evaluation reaches 96.7% train / 93.3% unseen task success; the remaining gap is collision frequency on a small set of hard routes and calibration under joint speed/map shift.

### Planned

- Improve route-level model alignment so low learned collision risk transfers to real barrier success without relying on explicit geometry fallback.
- Strengthen shift detection with representation-level and route-level features; the current innovation alarm is not sufficient for map OOD.
- Online demo deployment and browser controls for history/uncertainty planning.

## [0.3.0] - 2026-08-04

### Added

- Route-aware v8 probabilistic planner checkpoint and complete three-seed barrier evaluation.
- Release asset metadata and per-seed route alignment results.

### Changed

- Route-aware planning reaches 0.667% ± 0.943pp real success in the strict 150-episode protocol, improving the prior 0% boundary while documenting that the core problem remains open.
- Alignment-triggered hybrid fallback raises the same protocol to 7.333% ± 0.943pp real success and reduces mean collisions from 5.92 to 5.10 per episode.

## [0.2.0] - 2026-08-03

### Added

- Collision-seeking barrier curriculum and collision-only fine-tuning mode.
- Kinematics-only system identification for acceleration, friction, and speed limit.
- Map-agnostic learned-risk waypoint proposals and post-impact rollout response.
- Separate geometric imagined distance, collision risk, and planning score metrics.

### Changed

- Focused single-barrier learned planning improves from 0% to 75% real success.
- Python verification now covers 30 tests at approximately 76% line coverage.

## [0.1.0] - 2026-08-03

### Added

- Deterministic 64×64 Gymnasium environment with inertia, walls, collisions, and goals.
- CNN/GRU world model with structured position and velocity dynamics.
- Multi-step, OOD, planning-gap, collision, and renderer evaluation suites.
- Random-shooting, detour, receding-horizon, learned-collision, and hybrid planners.
- State-conditioned RGB agent composition and ONNX position output.
- React/TypeScript side-by-side simulator and model-imagination demo.
- Reproducible README GIF/SVG assets, MIT license, tests, coverage, and GitHub Actions.

### Known limitations

- Pure learned collision planning still fails the real single-barrier benchmark.
- The learned agent-mask head has low shape IoU and remains an ablation.

[Unreleased]: https://github.com/ZhouYinLong-lab/Pocket-World-Model/compare/v0.6.0...HEAD
[0.6.0]: https://github.com/ZhouYinLong-lab/Pocket-World-Model/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/ZhouYinLong-lab/Pocket-World-Model/compare/v0.4.0...v0.5.0
[0.4.0]: https://github.com/ZhouYinLong-lab/Pocket-World-Model/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/ZhouYinLong-lab/Pocket-World-Model/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ZhouYinLong-lab/Pocket-World-Model/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ZhouYinLong-lab/Pocket-World-Model/releases/tag/v0.1.0

