# Contributing to PocketWorld

PocketWorld keeps the model, simulator, planner, and evaluation loop deliberately small. Contributions are welcome when they preserve that observability.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,export]"
npm ci --no-audit --no-fund
```

## Before opening a pull request

Run the same checks as CI:

```bash
python -m pytest -q --cov=pocketworld --cov-report=term-missing --cov-fail-under=70
npm run build
```

For changes to training, planning, or evaluation, include:

- the exact command and random seed;
- the checkpoint configuration and evaluation episode count;
- both imagined and real execution metrics;
- negative or regressed results, not only the best seed;
- an update to `docs/evaluation-2026-08.md` when headline numbers change.

## Scope

Good first contributions include tests, reproducibility improvements, evaluation plots, browser visualization, and compact collision-aware dynamics. Large architecture changes should start with an issue explaining what research question they answer.

Please keep generated checkpoints and local experiment outputs in `artifacts/`; release-worthy files are published through GitHub Releases instead of Git history.
