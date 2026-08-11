# Changelog

All notable changes to PocketWorld are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses semantic versioning.

## [Unreleased]

### Added

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

### Changed

- The planner can compare learned temporal velocity plus calibrated probabilistic uncertainty against the existing finite-difference and robust-radius baselines.
- History-aware uncertainty planning reaches 76.7% ± 2.5pp real barrier success across 150 episodes, versus 64% for point-open planning.
- Mean collisions fall from 1.89 to 0.56 per episode in the three-seed comparison.
- Python verification now covers 54 tests at 79% line coverage.
- The A* fallback raises the strict 150-episode barrier result to 30.67% ± 2.49pp real success, with 3.19 ± 0.52 collisions per episode.
- Locked-budget A* fallback raises the same result to 34.67% ± 6.80pp real success, with 2.03 ± 0.35 collisions and zero emergency side switches per episode.
- Early alignment fallback raises the same result to 50.67% ± 2.49pp real success, with 6.88 ± 0.14px final distance, 1.14 ± 0.23 collisions, and 7.37px remaining route distance.

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

[Unreleased]: https://github.com/ZhouYinLong-lab/Pocket-World-Model/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/ZhouYinLong-lab/Pocket-World-Model/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/ZhouYinLong-lab/Pocket-World-Model/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ZhouYinLong-lab/Pocket-World-Model/releases/tag/v0.1.0
