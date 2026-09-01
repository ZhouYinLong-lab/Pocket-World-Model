import inspect
import json

import numpy as np
import pytest
import torch

from pocketworld.adaptive_horizon import AdaptiveHorizonPolicy, HorizonDecision
from pocketworld.evaluate_adaptive_horizon import (
    _candidate_bank,
    _case_hash,
    _build_cases,
    _condition_cases,
    _decision_fixed,
    _decision_log,
    _fixed_horizon_error_curve,
    _horizon_diagnostics,
    _calibration_rollout_tensors,
    _parse_floats,
    _parse_ints,
    _report_method,
    _safe_distance,
    _strict_float,
    _summarize_by_seed,
    _summarize_rows,
    load_protocol,
    validate_seed_splits,
)


def _low_risk_curves():
    return (
        {8: 0.05, 16: 0.08, 24: 0.12, 32: 0.15},
        {8: 0.02, 16: 0.03, 24: 0.04, 32: 0.05},
    )


def test_low_risk_conditions_allow_longest_horizon():
    policy = AdaptiveHorizonPolicy()
    uncertainty, collision = _low_risk_curves()

    decision = policy.select_horizon(uncertainty, collision, alignment_error=0.0)

    assert decision.horizon == 32
    assert decision.reason == "longest_risk_feasible_initial"
    assert set(decision.candidate_risks) == {8, 16, 24, 32}


def test_risk_increase_never_selects_a_longer_horizon():
    policy = AdaptiveHorizonPolicy()
    low_uncertainty, low_collision = _low_risk_curves()
    low = policy.select_horizon(low_uncertainty, low_collision, alignment_error=0.0)
    high = policy.select_horizon(
        {8: 0.9, 16: 0.95, 24: 1.0, 32: 1.0},
        {8: 0.8, 16: 0.9, 24: 1.0, 32: 1.0},
        alignment_error=0.0,
        previous_horizon=low.horizon,
    )

    assert high.horizon <= low.horizon
    assert high.horizon == 8


def test_extreme_risk_falls_back_to_shortest_candidate():
    policy = AdaptiveHorizonPolicy()
    decision = policy.select_horizon(1.0, 1.0, alignment_error=100.0, ood_score=100.0)

    assert decision.horizon == 8
    assert decision.reason == "all_candidates_exceed_risk_budget"


def test_hysteresis_holds_between_entry_and_exit_thresholds():
    policy = AdaptiveHorizonPolicy()
    middle = policy.select_horizon(0.95, 0.0, alignment_error=0.0, previous_horizon=16)
    high = policy.select_horizon(1.0, 1.0, alignment_error=0.0, previous_horizon=16)
    low = policy.select_horizon(0.0, 0.0, alignment_error=0.0, previous_horizon=8)

    assert middle.horizon == 16
    assert middle.reason == "hysteresis_hold"
    assert high.horizon < 16
    assert low.horizon > 8


def test_decision_is_json_safe_and_contains_complete_audit_fields():
    decision = AdaptiveHorizonPolicy().select_horizon(*_low_risk_curves(), alignment_error=1.0)
    assert isinstance(decision, HorizonDecision)
    payload = decision.as_dict()
    for field in (
        "horizon",
        "uncertainty_score",
        "collision_risk",
        "alignment_error",
        "reason",
        "risk_score",
        "risk_budget",
        "candidate_risks",
    ):
        assert field in payload
    assert set(payload["candidate_risks"]) == {"8", "16", "24", "32"}
    json.dumps(payload, allow_nan=False)


def test_policy_has_no_environment_or_future_state_input():
    signature = inspect.signature(AdaptiveHorizonPolicy.select_horizon)
    names = {name.lower() for name in signature.parameters}
    assert not names.intersection({"env", "true_state", "future", "label", "collision_label"})
    assert "env" not in AdaptiveHorizonPolicy.select_horizon.__code__.co_names


def test_calibration_and_final_seed_splits_are_disjoint():
    validate_seed_splits((101, 103, 107), (53, 67), (11, 23, 41))
    with pytest.raises(ValueError, match="overlap"):
        validate_seed_splits((101,), (101,), (11,))


def test_paired_cases_and_action_banks_are_deterministic():
    cases = _build_cases((11,), 2)
    assert _case_hash(cases) == _case_hash(_build_cases((11,), 2))
    first = _candidate_bank(11, 0, 0, cases[11][0].start, cases[11][0].goal, 8, 32)
    second = _candidate_bank(11, 0, 0, cases[11][0].start, cases[11][0].goal, 8, 32)
    assert first.shape == (8, 32)
    assert (first == second).all()


def test_evaluator_audit_helpers_are_finite_and_json_safe():
    assert _strict_float(1.25, "value") == 1.25
    with pytest.raises(ValueError, match="finite"):
        _strict_float(float("nan"), "value")
    assert _parse_ints("8, 16,32") == (8, 16, 32)
    assert _parse_floats("0.1, 1.5") == (0.1, 1.5)
    assert _safe_distance({"distance_to_goal": float("nan")}) == 64.0
    assert _safe_distance({"distance_to_goal": 3.0}) == 3.0

    fixed = _decision_fixed(16, 8)
    logged = _decision_log(fixed, 4, "fixed_horizon_16")
    assert logged["step"] == 4
    assert logged["method"] == "fixed_horizon_16"
    json.dumps(logged, allow_nan=False)


def test_evaluator_summaries_include_seed_and_horizon_audits():
    rows = [
        {
            "seed": 11,
            "imagined_success": 1.0,
            "real_success": 0.5,
            "imagination_real_success_gap": 0.5,
            "collision_count_per_episode": 1.0,
            "final_distance_px": 4.0,
            "route_completion": 0.5,
            "replanning_count": 2.0,
            "planning_latency_ms": 10.0,
            "model_queries": 32.0,
            "horizon_switches": 1.0,
            "alignment_error_px": 2.0,
            "max_alignment_error_px": 3.0,
            "astar_fallback_calls": 0.0,
            "selected_horizons": [32, 16],
            "horizon_decisions": [
                {"reason": "hysteresis_hold"},
                {"reason": "entry_threshold_shorten"},
            ],
        }
    ]
    summary = _summarize_rows(rows)
    assert summary["episodes"] == 1
    assert summary["selected_horizon_distribution"] == {"16": 1, "32": 1}
    assert summary["horizon_decision_reasons"]["hysteresis_hold"] == 1
    assert _summarize_by_seed(rows)["11"]["episodes"] == 1
    assert _report_method(rows)["rows"] == rows


def test_fixed_error_curve_is_a_paired_evaluation_diagnostic():
    class ZeroModel:
        def imagine_positions(self, observation, actions, **kwargs):
            return torch.zeros((observation.shape[0], actions.shape[1], 2))

    cases = _build_cases((11,), 1)
    curve = _fixed_horizon_error_curve(ZeroModel(), cases, 1.0, (8, 16))
    assert set(curve) == {"8", "16"}
    assert curve["8"]["samples"] == 1
    assert np.isfinite(curve["16"]["mean_position_error_px"])


def test_condition_case_hash_is_preserved_for_nominal_pairs():
    cases = _build_cases((11,), 1)
    conditioned, excluded = _condition_cases(cases, "nominal")
    assert excluded == {}
    assert _case_hash(conditioned) == _case_hash(cases)


def test_online_horizon_diagnostics_are_finite_and_horizon_indexed():
    from pocketworld.env import PocketWorldEnv
    from pocketworld.model import PocketWorldModel

    env = PocketWorldEnv(agent_start=(7.0, 7.0), goal=(57.0, 57.0))
    observation, _ = env.reset()
    action_bank = _candidate_bank(11, 0, 0, (7.0, 7.0), (57.0, 57.0), 4, 16)
    uncertainty, collision, metadata = _horizon_diagnostics(
        PocketWorldModel(),
        observation,
        [observation],
        action_bank,
        (8, 16),
        alignment_error=0.0,
        ood_score=0.0,
        recent_risk=0.0,
        uncertainty_budget_px=6.0,
        uncertainty_samples=4,
    )
    assert set(uncertainty) == {8, 16}
    assert set(collision) == {8, 16}
    assert np.isfinite(list(uncertainty.values())).all()
    assert np.isfinite(list(collision.values())).all()
    assert metadata["uncertainty_budget_px"] == 6.0


def test_calibration_rollout_tensor_shapes_are_split_local():
    tensors = _calibration_rollout_tensors(_build_cases((53,), 1), 1, 2)
    observations, actions, positions, velocities = tensors
    assert observations.shape == (1, 3, 3, 64, 64)
    assert actions.shape == (1, 2)
    assert positions.shape == (1, 3, 2)
    assert velocities.shape == (1, 3, 2)


def test_protocol_loader_rejects_non_object(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="object"):
        load_protocol(path)
