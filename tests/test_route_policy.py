import numpy as np
import pytest
import torch

from pocketworld.env import PocketWorldEnv
from pocketworld.evaluate_learned_route_policy import (
    collect_dagger_data,
    collect_teacher_data,
    evaluate_action_imitation,
    evaluate_policy,
    train_and_evaluate_route_mode,
)
from pocketworld.route_policy import (
    ROUTE_MODES,
    LearnedRoutePolicy,
    RouteModePolicy,
    observable_route_waypoints,
    observable_waypoint_action,
    route_mode_label,
)


def test_learned_route_policy_predicts_valid_actions_and_roundtrips(tmp_path):
    policy = LearnedRoutePolicy()
    observations = np.random.default_rng(4).integers(
        0, 256, size=(12, 3, 64, 64), dtype=np.uint8
    )
    goals = np.full((12, 2), 32.0, dtype=np.float32)
    velocities = np.zeros((12, 2), dtype=np.float32)
    positions = np.full((12, 2), 20.0, dtype=np.float32)
    actions = np.arange(12, dtype=np.int64) % 4
    metrics = policy.fit(
        observations, goals, velocities, actions, positions=positions, epochs=2, batch_size=4
    )
    assert metrics["samples"] == 12
    action = policy.predict_action(observations[0], (32.0, 32.0), [observations[0]])
    assert action in {0, 1, 2, 3}
    checkpoint = policy.save(tmp_path / "route-policy.pt")
    loaded = LearnedRoutePolicy.load(checkpoint)
    assert loaded.predict_action(observations[0], (32.0, 32.0), [observations[0]]) in {0, 1, 2, 3}


def test_teacher_collection_uses_observable_route_inputs():
    observations, goals, velocities, positions, actions = collect_teacher_data(
        seeds=(11,), scenarios=("single_barrier",), episodes=1, max_steps=4
    )
    assert observations.shape[1:] == (3, 64, 64)
    assert goals.shape == velocities.shape == positions.shape == (len(actions), 2)
    assert np.all((actions >= 0) & (actions < 4))


def test_policy_evaluation_does_not_require_astar():
    policy = LearnedRoutePolicy()
    result = evaluate_policy(
        policy,
        seeds=(11,),
        scenarios=("open",),
        episodes=1,
        max_steps=3,
    )
    assert len(result["rows"]) == 1
    assert 0.0 <= result["summary"]["real_success"]["mean"] <= 1.0


def test_action_imitation_reports_accuracy_without_environment_access():
    policy = LearnedRoutePolicy()
    observations = np.zeros((8, 3, 64, 64), dtype=np.uint8)
    goals = np.full((8, 2), 32.0, dtype=np.float32)
    velocities = np.zeros((8, 2), dtype=np.float32)
    positions = np.full((8, 2), 20.0, dtype=np.float32)
    actions = np.arange(8, dtype=np.int64) % 4
    policy.fit(
        observations,
        goals,
        velocities,
        actions,
        positions=positions,
        epochs=1,
        batch_size=8,
    )
    metrics = evaluate_action_imitation(
        policy, observations, goals, velocities, actions, positions
    )
    assert metrics["samples"] == 8
    assert 0.0 <= metrics["action_accuracy"] <= 1.0


def test_dagger_collection_labels_student_visited_states():
    policy = LearnedRoutePolicy()
    data = collect_dagger_data(
        policy,
        seeds=(11,),
        scenarios=("open",),
        episodes=1,
        max_steps=3,
    )
    assert data[0].shape[0] == 3
    assert data[1].shape == data[2].shape == data[3].shape == (3, 2)
    assert np.all((data[4] >= 0) & (data[4] < 4))
    assert np.all((data[5] >= 0) & (data[5] < 4))


def test_route_policy_fit_rejects_bad_shapes():
    policy = LearnedRoutePolicy()
    with pytest.raises(ValueError, match="observations"):
        policy.fit(
            np.zeros((4, 3, 32, 32), dtype=np.uint8),
            np.zeros((4, 2), dtype=np.float32),
            np.zeros((4, 2), dtype=np.float32),
            np.zeros(4, dtype=np.int64),
        )


def test_route_mode_labels_and_waypoints_are_observable():
    env = PocketWorldEnv(
        walls=(),
        agent_start=(8.0, 32.0),
        goal=(55.0, 32.0),
    )
    observation, _ = env.reset()
    assert route_mode_label(observation, (55.0, 32.0)) == 0
    waypoints = observable_route_waypoints(observation, (55.0, 32.0), 0)
    assert waypoints == ((55.0, 32.0),)
    assert observable_waypoint_action(observation, waypoints[0], [observation]) == 3
    assert ROUTE_MODES == ("direct", "top", "bottom", "gap")


def test_route_mode_policy_roundtrip(tmp_path):
    env = PocketWorldEnv(walls=(), agent_start=(8.0, 8.0), goal=(55.0, 55.0))
    observation, _ = env.reset()
    observations = np.stack([observation] * 8)
    goals = np.full((8, 2), 55.0, dtype=np.float32)
    labels = np.zeros(8, dtype=np.int64)
    policy = RouteModePolicy()
    metrics = policy.fit(observations, goals, labels, epochs=2)
    assert metrics["samples"] == 8
    assert policy.predict_mode(observation, (55.0, 55.0)) in range(4)
    loaded = RouteModePolicy.load(policy.save(tmp_path / "route-mode.pt"))
    assert loaded.predict_mode(observation, (55.0, 55.0)) in range(4)


def test_route_mode_end_to_end_report(tmp_path):
    report = train_and_evaluate_route_mode(
        train_seeds=(101,),
        train_scenarios=("single_barrier", "barrier_narrow_gap"),
        evaluation_seeds=(11,),
        evaluation_scenarios=("single_barrier", "barrier_narrow_gap"),
        validation_seeds=(107,),
        validation_scenarios=("single_barrier",),
        train_episodes=4,
        evaluation_episodes=1,
        max_steps=8,
        epochs=2,
        predictor_output=tmp_path / "route-mode-report.pt",
    )
    assert report["protocol"]["student_evaluation_uses_astar"] is False
    assert report["training"]["samples"] == 8
    assert report["validation"]["samples"] == 2
    assert len(report["evaluation"]["rows"]) == 2
