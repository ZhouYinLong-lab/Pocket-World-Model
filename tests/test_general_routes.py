from collections import Counter

import numpy as np
import pytest

from pocketworld.general_routes import (
    GENERAL_FAMILIES,
    general_wall_layout,
    sample_general_route_cases,
)
from pocketworld.evaluate_general_routes import GENERAL_DEFAULT_METHODS, train_and_evaluate_general_routes
from pocketworld.evaluate_mpc_ablation import run_mpc_ablation
from pocketworld.evaluate_general_ood import run_general_ood
from pocketworld.evaluate_coverage_study import run_coverage_study
from pocketworld.evaluate_planner_comparison import run_planner_comparison
from pocketworld.route_field import (
    RouteFieldPolicy,
    _coarse_transition_is_safe,
    conservative_field_action,
    local_mpc_action,
    rgb_action_is_safe,
    estimate_action_velocity,
    guarded_mpc_action,
    field_waypoints,
    route_field_targets,
)
from pocketworld.env import PocketWorldEnv
from pocketworld.planner import _astar_path, _dilate, extract_wall_mask


def test_general_families_are_valid_and_reachable():
    for family in GENERAL_FAMILIES:
        walls, channels = general_wall_layout(17, family)
        observation, _ = PocketWorldEnv(
            walls=walls, agent_start=(7.0, 7.0), goal=(57.0, 57.0)
        ).reset()
        path = _astar_path(
            _dilate(extract_wall_mask(observation), 4),
            (7.0, 7.0),
            (57.0, 57.0),
            allow_diagonal=False,
        )
        assert path
        assert channels >= 1


def test_balanced_family_sampling_is_reproducible_and_even():
    cases = sample_general_route_cases(
        101,
        8,
        split="train",
        families=GENERAL_FAMILIES,
        balanced=True,
    )
    counts = Counter(case.family for case in cases)
    assert counts == Counter({family: 2 for family in GENERAL_FAMILIES})
    assert cases == sample_general_route_cases(
        101,
        8,
        split="train",
        families=GENERAL_FAMILIES,
        balanced=True,
    )


def test_general_sampling_is_deterministic_and_covers_unseen_shapes():
    first = sample_general_route_cases(11, 20, split="holdout")
    second = sample_general_route_cases(11, 20, split="holdout")
    assert first == second
    assert set(case.family for case in first) >= {"staircase", "l_shapes"}
    assert all(abs(case.start[0] - case.goal[0]) == 50.0 for case in first)
    counts = Counter(case.family for case in first)
    assert sum(counts.values()) == 20


def test_general_sampling_validates_inputs():
    with pytest.raises(ValueError, match="episodes"):
        sample_general_route_cases(3, 0)
    with pytest.raises(ValueError, match="split"):
        sample_general_route_cases(3, 1, split="test")
    with pytest.raises(ValueError, match="family"):
        general_wall_layout(3, "unknown")


def test_general_evaluation_reports_method_astar_contract(tmp_path):
    report = train_and_evaluate_general_routes(
        train_seeds=(101,),
        evaluation_seeds=(11,),
        train_episodes=8,
        evaluation_episodes=1,
        max_steps=40,
        points=3,
        epochs=1,
        predictor_output=tmp_path / "general.pt",
    )
    contract = report["protocol"]["method_astar_contract"]
    assert contract["learned"] is False
    assert contract["rgb_projection"] is False
    assert contract["hybrid_astar"] is True
    assert contract["rgb_astar"] is True
    assert contract["distance_field_beam_rgb_projection"] is False
    assert contract["distance_field_beam_conservative"] is False
    assert contract["distance_field_beam_mpc"] is False
    assert contract["distance_field_beam_robust_mpc"] is False
    assert contract["distance_field_clearance_beam_rgb_projection"] is False
    assert contract["distance_field_beam_guarded_mpc"] is False
    assert set(report["evaluation"]) == {
        "learned",
        "rgb_projection",
        "hybrid_astar",
        "rgb_astar",
        "distance_field",
        "distance_field_rgb_projection",
        "distance_field_beam_rgb_projection",
        "distance_field_beam_conservative",
        "distance_field_beam_mpc",
        "distance_field_beam_robust_mpc",
        "distance_field_clearance_beam_rgb_projection",
        "distance_field_beam_guarded_mpc",
    }
    assert report["distance_field_checkpoint"].endswith("-distance-field.pt")
    assert report["clearance_field_checkpoint"].endswith("-clearance-field.pt")
    assert report["protocol"]["mpc_horizon"] == 6
    assert report["protocol"]["mpc_beam_width"] == 24
    assert report["protocol"]["methods"] == list(GENERAL_DEFAULT_METHODS)
    assert report["evaluation"]["distance_field_beam_mpc"]["summary"]["mpc_calls"]["mean"] >= 0


def test_route_field_policy_roundtrip_and_waypoints(tmp_path):
    cases = sample_general_route_cases(101, 8, split="train")
    from pocketworld.env import PocketWorldEnv
    import numpy as np

    frames = []
    goals = []
    for case in cases:
        frame, _ = PocketWorldEnv(walls=case.walls, agent_start=case.start, goal=case.goal).reset()
        frames.append(frame)
        goals.append(case.goal)
    frames = np.asarray(frames, dtype=np.uint8)
    goals = np.asarray(goals, dtype=np.float32)
    targets, valid = route_field_targets(frames, goals)
    assert targets.shape == valid.shape == (8, 16, 16)
    risk_targets, risk_valid = route_field_targets(frames, goals, clearance_weight=8.0)
    assert risk_targets.shape == risk_valid.shape == (8, 16, 16)
    with pytest.raises(ValueError, match="clearance_weight"):
        route_field_targets(frames, goals, clearance_weight=-1.0)
    policy = RouteFieldPolicy()
    assert policy.fit(frames, goals, epochs=1)["samples"] == 8
    predicted = policy.predict_field(frames[0], tuple(goals[0]))
    assert predicted.shape == (16, 16)
    assert field_waypoints(frames[0], tuple(goals[0]), predicted)
    assert field_waypoints(frames[0], tuple(goals[0]), predicted, beam_width=4)
    assert local_mpc_action(frames[0], tuple(goals[0]), [frames[0]], horizon=2, beam_width=4) in range(4)
    assert local_mpc_action(frames[0], tuple(goals[0]), [frames[0]], horizon=1, beam_width=1) in range(4)
    assert local_mpc_action(frames[0], tuple(goals[0]), [frames[0]], horizon=2, beam_width=4, robust=True) in range(4)
    assert estimate_action_velocity([frames[0]], [0, 1, 3]).shape == (2,)
    assert isinstance(rgb_action_is_safe(frames[0], 0, [frames[0]], [0]), bool)
    assert guarded_mpc_action(frames[0], tuple(goals[0]), 0, [frames[0]], [0], horizon=2, beam_width=4) in range(4)
    assert conservative_field_action(frames[0], tuple(goals[0]), [frames[0]]) in range(4)
    assert RouteFieldPolicy.load(policy.save(tmp_path / "field.pt")).grid_size == 16


def test_mpc_ablation_has_fixed_protocol_and_family_metrics(tmp_path):
    cases = sample_general_route_cases(101, 8, split="train")
    frames = []
    goals = []
    for case in cases:
        frame, _ = PocketWorldEnv(
            walls=case.walls, agent_start=case.start, goal=case.goal
        ).reset()
        frames.append(frame)
        goals.append(case.goal)
    policy = RouteFieldPolicy()
    policy.fit(np.asarray(frames), np.asarray(goals, dtype=np.float32), epochs=1)
    checkpoint = policy.save(tmp_path / "field.pt")
    report = run_mpc_ablation(
        checkpoint,
        evaluation_seeds=(11,),
        evaluation_episodes=1,
        max_steps=8,
        configs=(
            {"name": "smoke", "horizon": 2, "beam_width": 4, "velocity_source": "rgb", "robust": False},
        ),
    )
    assert report["protocol"]["student_evaluation_uses_astar"] is False
    evaluation = report["results"]["smoke"]["evaluation"]
    assert evaluation["by_family"]
    assert "planning_time_ms" in evaluation["summary"]


def test_general_ood_protocol_keeps_shift_hidden_and_checks_reachability(tmp_path):
    cases = sample_general_route_cases(101, 8, split="train")
    frames = []
    goals = []
    for case in cases:
        frame, _ = PocketWorldEnv(
            walls=case.walls, agent_start=case.start, goal=case.goal
        ).reset()
        frames.append(frame)
        goals.append(case.goal)
    policy = RouteFieldPolicy()
    policy.fit(np.asarray(frames), np.asarray(goals, dtype=np.float32), epochs=1)
    checkpoint = policy.save(tmp_path / "field.pt")
    report = run_general_ood(
        checkpoint,
        evaluation_seeds=(11,),
        evaluation_episodes=1,
        max_steps=8,
        map_shifts=("nominal", "walls_x_plus2"),
        speed_scales=(0.75,),
        methods=("distance_field_mpc_shift_fallback",),
        mpc_horizon=2,
        mpc_beam_width=4,
    )
    assert report["protocol"]["student_evaluation_uses_astar"] is False
    assert report["protocol"]["fallback_method_uses_astar"] is True
    assert report["protocol"]["shift_labels_visible_to_planner"] is False
    assert set(report["results"]) == {"nominal@speed0.75", "walls_x_plus2@speed0.75"}
    assert report["results"]["walls_x_plus2@speed0.75"]["paired_episode_count"] >= 0


def test_coverage_study_keeps_sample_count_and_holdout_shared(tmp_path):
    report = run_coverage_study(
        train_seeds=(101,),
        evaluation_seeds=(11,),
        train_episodes=8,
        evaluation_episodes=1,
        max_steps=8,
        points=3,
        epochs=1,
        mpc_horizon=2,
        mpc_beam_width=4,
        predictor_root=tmp_path / "coverage",
    )
    assert report["protocol"]["holdout_is_shared"] is True
    assert report["protocol"]["balanced_training_families"] is True
    assert set(report["conditions"]) == {
        "two_family_600",
        "four_family_600",
        "four_family_1200",
    }
    for condition in report["conditions"].values():
        assert condition["report"]["protocol"]["methods"] == [
            "distance_field_beam_rgb_projection",
            "distance_field_beam_mpc",
        ]


def test_planner_comparison_uses_one_shared_holdout_and_explicit_reference(tmp_path):
    checkpoint = tmp_path / "field.pt"
    RouteFieldPolicy().save(checkpoint)
    report = run_planner_comparison(
        checkpoint=checkpoint,
        evaluation_seeds=(11,),
        evaluation_episodes=1,
        max_steps=1,
        points=3,
        methods=("rgb_astar",),
    )
    assert report["protocol"]["shared_holdout_cases"] is True
    assert report["protocol"]["rgb_astar_is_geometric_reference"] is True
    assert report["protocol"]["selection_on_holdout"] is False
    assert set(report["evaluation"]) == {"rgb_astar"}


def test_route_field_rgb_guard_checks_the_edge_not_only_centers():
    import numpy as np

    occupied = np.zeros((64, 64), dtype=bool)
    occupied[28:36, 20:44] = True
    assert not _coarse_transition_is_safe(occupied, (4, 7), (5, 7), 4)
    assert _coarse_transition_is_safe(occupied, (4, 4), (4, 5), 4)
