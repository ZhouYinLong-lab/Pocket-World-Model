import numpy as np

from pocketworld.env import PocketWorldEnv, Rect
from pocketworld.maps import get_map
from pocketworld.planner import (
    _collision_prefix,
    _astar_path,
    _dilate,
    _learned_waypoint_templates,
    _path_waypoints,
    _wall_aware_route_templates,
    estimate_agent_velocity,
    estimate_speed_response,
    cem_shooting,
    beam_search,
    extract_wall_boxes,
    extract_wall_mask,
    predictive_shift_score,
    wall_context_shift_score,
    route_following_action,
)


def test_wall_mask_detects_wall_but_not_grid_or_agent():
    env = PocketWorldEnv(walls=(Rect(24, 8, 5, 29),), agent_start=(8, 8))
    frame, _ = env.reset()
    mask = extract_wall_mask(frame)
    assert bool(mask[12, 26])
    assert not bool(mask[8, 8])
    assert not bool(mask[0, 0])


def test_collision_prefix_marks_wall_intersection_and_future_steps():
    wall = np.zeros((64, 64), dtype=bool)
    wall[20:40, 30:34] = True
    positions = np.asarray([[[10, 30], [26, 30], [31, 30], [40, 30]]], dtype=np.float32)
    prefix = _collision_prefix(positions, wall)
    assert prefix.tolist() == [[False, False, True, True]]


def test_wall_boxes_find_single_barrier():
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:54, 29:34] = True
    assert extract_wall_boxes(mask) == ((29.0, 10.0, 33.0, 53.0),)


def test_astar_route_crosses_barrier_without_entering_inflated_wall():
    wall = np.zeros((64, 64), dtype=bool)
    wall[10:54, 29:34] = True
    occupied = _dilate(wall, radius=4)
    path = _astar_path(occupied, (10, 32), (54, 32), vertical_preference="top")

    assert path
    assert all(not occupied[y, x] for x, y in path)
    assert min(y for _, y in path) <= 6


def test_route_following_action_uses_four_action_visible_route():
    env = PocketWorldEnv(walls=(Rect(29, 10, 5, 44),), agent_start=(10, 32), goal=(54, 32))
    observation, _ = env.reset()
    action, remaining = route_following_action(observation, (54, 32), [observation])
    assert action in {0, 1, 2, 3}
    assert remaining > 0.0


def test_route_following_action_does_not_choose_visible_wall_landing():
    env = PocketWorldEnv(walls=(Rect(29, 10, 5, 44),), agent_start=(25, 32), goal=(54, 32))
    observation, _ = env.reset()
    action, _ = route_following_action(observation, (54, 32), [observation])
    _, _, _, _, info = env.step(action)
    assert not info["collision"]


def test_route_following_action_matches_continuous_wall_boundary_at_cross_corner():
    env = PocketWorldEnv(
        walls=get_map("cross").walls,
        agent_start=(44.59, 58.28),
        goal=(7.3, 11.0),
    )
    observation, _ = env.reset()
    action, _ = route_following_action(observation, (7.3, 11.0), [observation])
    _, _, _, _, info = env.step(action)
    assert not info["collision"]


def test_path_waypoints_keep_bends_when_downsampling_dense_grid_route():
    path = [(10, 32), (11, 31), (12, 30), (12, 29), (12, 28), (13, 27)]

    assert _path_waypoints(path, spacing=len(path) + 1) == (
        (10.0, 32.0),
        (12.0, 30.0),
        (12.0, 28.0),
        (13.0, 27.0),
    )


def test_wall_aware_templates_are_generated_from_observed_geometry():
    import torch

    from pocketworld.model import PocketWorldModel

    wall = np.zeros((64, 64), dtype=bool)
    wall[10:54, 29:34] = True
    observation = torch.zeros(1, 3, 64, 64)
    templates = _wall_aware_route_templates(
        PocketWorldModel(),
        observation,
        np.asarray((10.0, 32.0)),
        (54.0, 32.0),
        wall,
        horizon=20,
    )

    assert templates
    assert all(len(template) == 20 for template in templates)
    assert all(set(template).issubset({0, 1, 2, 3}) for template in templates)


def test_wall_aware_route_preference_locks_to_one_side_and_reports_budget_distance():
    import torch

    from pocketworld.model import PocketWorldModel
    from pocketworld.planner import random_shooting

    env = PocketWorldEnv(walls=(Rect(29, 10, 5, 44),), agent_start=(10, 32), goal=(54, 32))
    observation, _ = env.reset()
    wall_mask = extract_wall_mask(observation)
    model = PocketWorldModel()

    top_templates = _wall_aware_route_templates(
        model,
        torch.from_numpy(observation[None]).float() / 255.0,
        np.asarray((10.0, 32.0)),
        (54.0, 32.0),
        wall_mask,
        horizon=12,
        route_preference="top",
    )
    assert len(top_templates) == 1

    result = random_shooting(
        model,
        observation,
        (54, 32),
        horizon=12,
        candidates=4,
        collision_aware=True,
        hybrid_collision=True,
        route_objective=True,
        wall_aware_route=True,
    )
    assert result.wall_route_preference in {"top", "bottom"}
    assert result.wall_route_remaining_px > 0.0


def test_wall_context_shift_score_is_zero_on_training_map_and_high_on_shifted_map():
    from pocketworld.data import _variant_walls

    default_env = PocketWorldEnv(agent_start=(8, 8), goal=(55, 55))
    default_frame, _ = default_env.reset()
    shifted_env = PocketWorldEnv(walls=_variant_walls(np.random.default_rng(3)), agent_start=(8, 8), goal=(55, 55))
    shifted_frame, _ = shifted_env.reset()

    assert wall_context_shift_score(default_frame) == 0.0
    assert wall_context_shift_score(shifted_frame) > 1.0


def test_receding_horizon_result_exposes_executed_trace():
    from pocketworld.model import PocketWorldModel
    from pocketworld.planner import receding_horizon_plan

    env = PocketWorldEnv(walls=(), agent_start=(8, 8), goal=(16, 16))
    observation, _ = env.reset()
    result = receding_horizon_plan(PocketWorldModel(), observation, (16, 16), env.step, max_steps=2, rollout_horizon=2, candidates=4)
    assert result.actions.ndim == 1
    assert result.final_observation.shape == observation.shape


def test_learned_waypoint_templates_include_both_sides_of_direct_route():
    import torch

    from pocketworld.model import PocketWorldModel

    model = PocketWorldModel()
    observation = torch.zeros(1, 3, 64, 64)
    templates = _learned_waypoint_templates(model, observation, np.asarray((10.0, 32.0)), (54.0, 32.0), horizon=20)

    assert len(templates) == 8
    assert all(len(template) == 20 for template in templates)
    assert any(0 in template for template in templates)
    assert any(1 in template for template in templates)


def test_learned_plan_reports_goal_distance_separately_from_risk_score():
    import torch

    from pocketworld.model import PocketWorldModel
    from pocketworld.planner import random_shooting

    env = PocketWorldEnv(walls=(Rect(29, 10, 5, 44),), agent_start=(10, 32), goal=(54, 32))
    observation, _ = env.reset()
    torch.manual_seed(3)
    result = random_shooting(
        PocketWorldModel(),
        observation,
        (54, 32),
        horizon=4,
        candidates=8,
        collision_aware=True,
        learned_collision=True,
    )

    assert 0.0 <= result.imagined_collision_risk <= 1.0
    assert result.planning_score >= result.imagined_distance


def test_history_velocity_estimator_tracks_recent_motion_and_reset():
    env = PocketWorldEnv(walls=(), agent_start=(20, 20), goal=(55, 55))
    first, _ = env.reset()
    second, _, _, _, _ = env.step(3)
    third, _, _, _, _ = env.step(3)

    assert np.allclose(estimate_agent_velocity([first]), 0.0)
    velocity = estimate_agent_velocity([first, second, third])
    assert velocity[0] > 0.0
    assert abs(velocity[1]) < 0.1
    assert np.linalg.norm(velocity) <= 2.3


def test_speed_response_estimator_uses_rgb_action_history():
    env = PocketWorldEnv(walls=(), agent_start=(20, 20), goal=(55, 55))
    frames = [env.reset()[0]]
    actions = [3, 3, 3, 3]
    for action in actions:
        frames.append(env.step(action)[0])
    response = estimate_speed_response(frames, actions)
    assert np.isfinite(response)
    assert response > 0.0


def test_random_shooting_accepts_learned_velocity_and_probability_flags():
    import torch

    from pocketworld.model import PocketWorldModel
    from pocketworld.planner import random_shooting

    env = PocketWorldEnv(walls=(Rect(29, 10, 5, 44),), agent_start=(10, 32), goal=(54, 32))
    first, _ = env.reset()
    second, _, _, _, _ = env.step(3)
    torch.manual_seed(4)
    result = random_shooting(
        PocketWorldModel(),
        second,
        (54, 32),
        horizon=3,
        candidates=4,
        collision_aware=True,
        learned_collision=True,
        observation_history=[first, second],
        use_learned_velocity=True,
        probabilistic_uncertainty=True,
        uncertainty_samples=8,
    )

    assert result.actions.ndim == 1
    assert np.isfinite(result.planning_score)


def test_cem_shooting_returns_valid_discrete_plan_under_budget():
    import torch

    from pocketworld.model import PocketWorldModel

    env = PocketWorldEnv(walls=(), agent_start=(8, 8), goal=(18, 28))
    observation, _ = env.reset()
    torch.manual_seed(9)
    result = cem_shooting(PocketWorldModel(), observation, (18, 28), horizon=6, candidates=32)

    assert 1 <= len(result.actions) <= 6
    assert result.actions.dtype == np.int64
    assert np.all((result.actions >= 0) & (result.actions < 4))
    assert result.imagined_positions.shape == (len(result.actions) + 1, 2)
    assert np.isfinite(result.imagined_distance)


def test_beam_search_returns_valid_discrete_plan_under_budget():
    import torch

    from pocketworld.model import PocketWorldModel

    env = PocketWorldEnv(walls=(), agent_start=(8, 8), goal=(18, 28))
    observation, _ = env.reset()
    torch.manual_seed(10)
    result = beam_search(PocketWorldModel(), observation, (18, 28), horizon=6, candidates=32)

    assert 1 <= len(result.actions) <= 6
    assert result.actions.dtype == np.int64
    assert np.all((result.actions >= 0) & (result.actions < 4))
    assert result.imagined_positions.shape == (len(result.actions) + 1, 2)
    assert np.isfinite(result.imagined_distance)


def test_collision_risk_budget_prefers_feasible_steps_and_has_soft_fallback():
    from pocketworld.planner import _risk_budget_scores

    distances = np.asarray([[10.0, 5.0, 2.0], [10.0, 5.0, 2.0]], dtype=np.float32)
    risks = np.asarray([[0.0, 0.10, 0.30], [0.0, 0.40, 0.50]], dtype=np.float32)
    scores = _risk_budget_scores(distances, risks, 0.20)

    assert np.isclose(scores[0, 1], 5.0)
    assert scores[0, 2] > 100.0
    assert np.isfinite(scores[1]).all()


def test_route_objective_reports_progress_and_alignment_monitor_is_finite():
    import torch

    from pocketworld.model import PocketWorldModel
    from pocketworld.planner import random_shooting

    env = PocketWorldEnv(walls=(), agent_start=(10, 10), goal=(54, 54))
    first, _ = env.reset()
    second, _, _, _, _ = env.step(3)
    result = random_shooting(
        PocketWorldModel(),
        second,
        (54, 54),
        horizon=4,
        candidates=8,
        route_objective=True,
        observation_history=[first, second],
        use_learned_velocity=True,
    )

    assert np.isfinite(result.route_score)
    assert np.isfinite(result.route_progress)
    assert 0.0 <= result.predicted_route_completion_probability <= 1.0
    assert np.isfinite(predictive_shift_score(PocketWorldModel(), first, 3, second, [first]))


def test_receding_horizon_reports_collision_and_replan_counts():
    from pocketworld.model import PocketWorldModel
    from pocketworld.planner import receding_horizon_plan

    env = PocketWorldEnv(walls=(), agent_start=(8, 8), goal=(16, 16))
    observation, _ = env.reset()
    result = receding_horizon_plan(
        PocketWorldModel(),
        observation,
        (16, 16),
        env.step,
        max_steps=2,
        rollout_horizon=2,
        candidates=4,
        use_history_velocity=True,
        route_objective=True,
        shift_threshold=0.0,
        alignment_fallback_threshold=0.0,
    )

    assert result.collision_count >= 0
    assert result.replans >= 1
    assert np.isfinite(result.final_info["distance_to_goal"])
    assert result.route_alignment_error_px >= 0.0
    assert result.shift_detected_count >= 0
    assert result.alignment_fallback_trigger_count >= 1
    assert result.fallback_steps >= 1
