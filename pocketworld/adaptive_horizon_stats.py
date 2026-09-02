"""Auditable summaries for the adaptive-horizon paired evaluation.

The formal evaluator stores every episode for every method.  This module
turns that machine-readable report into a compact application-facing summary
without editing metrics by hand.  Treatment deltas are paired by the
``(seed, map_id)`` case key and use a deterministic paired bootstrap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_BOOTSTRAP_RESAMPLES = 2_000
DEFAULT_BOOTSTRAP_SEED = 20_260_902
PAIR_METRICS = (
    "real_success",
    "collision_count_per_episode",
    "final_distance_px",
    "planning_latency_ms",
    "model_queries",
)


def _finite_array(values: Iterable[float]) -> np.ndarray:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.isfinite(array).all():
        raise ValueError("bootstrap samples must be a non-empty finite vector")
    return array


def summarize_samples(
    values: Iterable[float],
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Return mean/std/median and a deterministic percentile bootstrap CI."""

    if resamples < 1:
        raise ValueError("resamples must be positive")
    array = _finite_array(values)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(resamples, len(array)))
    bootstrap_means = array[indices].mean(axis=1)
    result = {
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "median": float(np.median(array)),
        "samples": int(len(array)),
        "bootstrap": {
            "method": "percentile_mean",
            "confidence": 0.95,
            "resamples": int(resamples),
            "seed": int(seed),
            "ci95": {
                "lower": float(np.quantile(bootstrap_means, 0.025)),
                "upper": float(np.quantile(bootstrap_means, 0.975)),
            },
        },
    }
    return result


def _pair_rows(
    baseline_rows: list[dict[str, Any]],
    treatment_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def index(rows: list[dict[str, Any]]) -> dict[tuple[int, str], dict[str, Any]]:
        result: dict[tuple[int, str], dict[str, Any]] = {}
        for row in rows:
            key = (int(row["seed"]), str(row["map_id"]))
            if key in result:
                raise ValueError(f"duplicate paired case: {key}")
            result[key] = row
        return result

    baseline = index(baseline_rows)
    treatment = index(treatment_rows)
    if set(baseline) != set(treatment):
        raise ValueError("paired methods do not share the same case keys")
    keys = sorted(baseline)
    return [baseline[key] for key in keys], [treatment[key] for key in keys]


def paired_bootstrap_delta(
    baseline_rows: list[dict[str, Any]],
    treatment_rows: list[dict[str, Any]],
    metric: str,
    *,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Summarize treatment minus baseline on exactly paired cases."""

    baseline, treatment = _pair_rows(baseline_rows, treatment_rows)
    deltas = [float(treatment_row[metric]) - float(baseline_row[metric]) for baseline_row, treatment_row in zip(baseline, treatment)]
    summary = summarize_samples(deltas, resamples=resamples, seed=seed)
    summary["comparison"] = "treatment_minus_baseline"
    summary["metric"] = metric
    return summary


def _source_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _method_summary(method_payload: dict[str, Any]) -> dict[str, Any]:
    summary = dict(method_payload["summary"])
    summary["by_seed"] = method_payload["by_seed"]
    return summary


def build_summary(
    report: dict[str, Any],
    *,
    source_report: str,
    source_path: Path,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Build a compact, JSON-safe summary from a completed full report."""

    conditions = report["conditions"]
    pure_learning: dict[str, Any] = {}
    astar_fallback: dict[str, Any] = {}
    paired_deltas: dict[str, Any] = {}
    for condition_name, condition in conditions.items():
        pure = condition["pure_learning"]
        fallback = condition["astar_fallback"]
        pure_learning[condition_name] = {
            method: _method_summary(payload) for method, payload in pure.items()
        }
        astar_fallback[condition_name] = {
            method: _method_summary(payload) for method, payload in fallback.items()
        }
        paired_deltas[condition_name] = {
            "pure_learning_adaptive_minus_fixed_horizon_16": {
                metric: paired_bootstrap_delta(
                    pure["fixed_horizon_16"]["rows"],
                    pure["adaptive_horizon"]["rows"],
                    metric,
                    resamples=resamples,
                    seed=bootstrap_seed + index,
                )
                for index, metric in enumerate(PAIR_METRICS)
            },
            "astar_fallback_adaptive_minus_fixed_horizon_16": {
                metric: paired_bootstrap_delta(
                    fallback["fixed_horizon_16"]["rows"],
                    fallback["adaptive_horizon"]["rows"],
                    metric,
                    resamples=resamples,
                    seed=bootstrap_seed + 100 + index,
                )
                for index, metric in enumerate(PAIR_METRICS)
            },
        }

    protocol = dict(report["protocol"])
    protocol.update(
        {
            "source_report": source_report,
            "source_report_sha256": _source_sha256(source_path),
            "summary_generator_revision": _git_revision(),
            "bootstrap": {
                "method": "paired_percentile_mean",
                "confidence": 0.95,
                "resamples": int(resamples),
                "seed": int(bootstrap_seed),
                "pair_key": ["seed", "map_id"],
            },
            "imagined_success_definition": (
                "success of the first selected plan's predicted endpoint; "
                "not closed-loop episode success"
            ),
            "imagined_real_gap_comparable_at_policy_level": False,
        }
    )
    id_curve = conditions["id"]["protocol"]["fixed_horizon_error_curve"]
    fixed_curve = {
        str(horizon): float(values["mean_position_error_px"])
        for horizon, values in id_curve.items()
    }
    return {
        "protocol": protocol,
        "calibration": report["calibration"],
        "pure_learning": pure_learning,
        "astar_fallback": astar_fallback,
        "paired_bootstrap_deltas": paired_deltas,
        "fixed_horizon_error_curve_id_px": fixed_curve,
        "conclusion": (
            "The formal result does not support claiming adaptive-horizon safety "
            "improvement: pure-learning collision rates are not lower than "
            "fixed-16, while model queries and latency are lower. It supports a "
            "negative/qualified result: the policy adapts computation and often "
            "selects short horizons, but shorter imagination alone does not solve "
            "obstacle traversal. Hybrid gains must remain attributed to the "
            "explicit A* fallback. The imagined-success field is a first-plan "
            "diagnostic and must not be presented as a closed-loop policy gap."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", help="completed full adaptive-horizon JSON report")
    parser.add_argument("--output", required=True, help="compact summary JSON path")
    parser.add_argument("--resamples", type=int, default=DEFAULT_BOOTSTRAP_RESAMPLES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()
    source_path = Path(args.report)
    report = json.loads(source_path.read_text(encoding="utf-8"))
    summary = build_summary(
        report,
        source_report=str(args.report),
        source_path=source_path,
        resamples=args.resamples,
        bootstrap_seed=args.bootstrap_seed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {output} from {source_path}")


if __name__ == "__main__":
    main()
