import json

import numpy as np
import pytest

from pocketworld.adaptive_horizon_stats import (
    build_summary,
    paired_bootstrap_delta,
    summarize_samples,
)


def _row(seed, map_id, **metrics):
    return {"seed": seed, "map_id": map_id, **metrics}


def test_paired_bootstrap_uses_treatment_minus_baseline_and_is_reproducible():
    baseline = [_row(11, "a", real_success=0.0), _row(11, "b", real_success=1.0)]
    treatment = [_row(11, "a", real_success=1.0), _row(11, "b", real_success=1.0)]

    first = paired_bootstrap_delta(baseline, treatment, "real_success", resamples=100, seed=7)
    second = paired_bootstrap_delta(baseline, treatment, "real_success", resamples=100, seed=7)

    assert first == second
    assert first["mean"] == 0.5
    assert first["comparison"] == "treatment_minus_baseline"


def test_paired_bootstrap_rejects_unpaired_or_duplicate_cases():
    baseline = [_row(11, "a", real_success=0.0)]
    with pytest.raises(ValueError, match="same case keys"):
        paired_bootstrap_delta(baseline, [_row(11, "b", real_success=1.0)], "real_success")
    with pytest.raises(ValueError, match="duplicate"):
        paired_bootstrap_delta(
            baseline + [_row(11, "a", real_success=0.0)],
            [_row(11, "a", real_success=1.0), _row(11, "b", real_success=1.0)],
            "real_success",
        )


def test_summary_contains_ci_median_and_finite_json():
    summary = summarize_samples([1.0, 2.0, 3.0], resamples=100, seed=3)
    assert summary["median"] == 2.0
    assert summary["bootstrap"]["ci95"]["lower"] <= 2.0
    json.dumps(summary, allow_nan=False)


def test_build_summary_marks_first_plan_imagination_semantics(tmp_path):
    rows = []
    for method, real, collision in (("fixed_horizon_16", 0.0, 2.0), ("adaptive_horizon", 1.0, 1.0)):
        rows.append(_row(11, "case", method=method, real_success=real, collision_count_per_episode=collision, final_distance_px=1.0, planning_latency_ms=1.0, model_queries=2.0))
    def payload(row):
        return {"summary": {"episodes": 1}, "by_seed": {"11": {"episodes": 1}}, "rows": [row]}

    report = {
        "protocol": {},
        "calibration": {"samples": 1},
        "conditions": {
            "id": {
                "protocol": {"fixed_horizon_error_curve": {"16": {"mean_position_error_px": 1.0}}},
                    "pure_learning": {"fixed_horizon_16": payload(rows[0]), "adaptive_horizon": payload(rows[1])},
                    "astar_fallback": {"fixed_horizon_16": payload(rows[0]), "adaptive_horizon": payload(rows[1])},
                }
            },
        }
    source = tmp_path / "report.json"
    source.write_text(json.dumps(report), encoding="utf-8")

    summary = build_summary(report, source_report=str(source), source_path=source, resamples=20, bootstrap_seed=1)

    assert summary["protocol"]["imagined_real_gap_comparable_at_policy_level"] is False
    assert summary["paired_bootstrap_deltas"]["id"]["pure_learning_adaptive_minus_fixed_horizon_16"]["real_success"]["mean"] == 1.0
    json.dumps(summary, allow_nan=False)
