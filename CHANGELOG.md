# Changelog

All notable changes to PocketWorld are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses semantic versioning.

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
