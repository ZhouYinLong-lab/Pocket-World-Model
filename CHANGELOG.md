# Changelog

All notable changes to PocketWorld are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses semantic versioning.

## [Unreleased]

### Planned

- Multi-seed validation of learned barrier planning and uncertainty-aware narrow-passage risk.
- Published raw multi-seed evaluation data and an online demo deployment.

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

[Unreleased]: https://github.com/ZhouYinLong-lab/Pocket-World-Model/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/ZhouYinLong-lab/Pocket-World-Model/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/ZhouYinLong-lab/Pocket-World-Model/releases/tag/v0.1.0
