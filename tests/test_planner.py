import numpy as np

from pocketworld.env import PocketWorldEnv, Rect
from pocketworld.planner import _collision_prefix, _learned_waypoint_templates, estimate_agent_velocity, extract_wall_boxes, extract_wall_mask


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
    )

    assert result.collision_count >= 0
    assert result.replans >= 1
    assert np.isfinite(result.final_info["distance_to_goal"])
