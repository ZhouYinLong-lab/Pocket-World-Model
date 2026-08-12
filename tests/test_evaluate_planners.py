import numpy as np

from pocketworld.evaluate_planners import (
    NARROW_GAP_WALLS,
    SHIFTED_BARRIER_WALLS,
    WIDE_GAP_WALLS,
    _scenario_walls,
    evaluate_planner_tournament,
    PLANNER_SEED_OFFSETS,
)
from pocketworld.model import PocketWorldModel


def test_planner_tournament_uses_paired_tasks_and_recursive_summary():
    report = evaluate_planner_tournament(
        PocketWorldModel(),
        seeds=(5,),
        episodes=2,
        horizon=4,
        candidates=8,
        scenario="open",
        planners=("random_shooting", "cem", "beam_search", "cem_collision"),
    )

    assert report["config"]["paired_tasks"] is True
    assert report["config"]["candidates"] == 8
    assert set(report["summary"]) == {"random_shooting", "cem", "beam_search", "cem_collision"}
    for planner in report["summary"].values():
        assert 0.0 <= planner["real_success"]["mean"] <= 1.0
        assert np.isfinite(planner["real_final_distance_px"]["mean"])


def test_closed_loop_route_planner_does_not_fake_imagination_metrics():
    report = evaluate_planner_tournament(
        PocketWorldModel(),
        seeds=(5,),
        episodes=1,
        horizon=8,
        candidates=8,
        scenario="single_barrier",
        planners=("route_aware_hybrid",),
    )

    row = report["summary"]["route_aware_hybrid"]
    assert row["imagined_success"] is None
    assert row["imagined_final_distance_px"] is None
    assert 0.0 <= row["real_success"]["mean"] <= 1.0


def test_fixed_predictor_safety_entrypoint_reports_all_methods(tmp_path):
    import torch

    from pocketworld.evaluate_safety_methods import evaluate_safety_methods
    from pocketworld.model import PocketWorldModel
    from pocketworld.route_completion import RouteCompletionPredictor

    checkpoint_path = tmp_path / "world-model.pt"
    torch.save({"model": PocketWorldModel().state_dict()}, checkpoint_path)
    predictor_path = tmp_path / "route-predictor.pt"
    predictor = RouteCompletionPredictor()
    features = np.asarray(
        [
            [0.8, 0.2, 0.2, 0.6, 0.0, 1.0, 0.4, 0.1, 0.5],
            [0.8, 0.8, 0.7, 0.0, 1.0, 0.3, 0.8, 0.8, 0.5],
            [0.7, 0.3, 0.3, 0.4, 0.1, 0.9, 0.5, 0.2, 0.5],
            [0.7, 0.9, 0.8, -0.2, 0.9, 0.2, 0.9, 0.9, 0.5],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    predictor.fit(features, labels, epochs=1)
    predictor.save(predictor_path)

    report = evaluate_safety_methods(
        checkpoint_path,
        predictor_path,
        evaluation_seeds=(5,),
        evaluation_episodes=1,
        candidates=8,
        horizon=4,
    )

    assert "route_completion_soft" in report["summary"]
    assert "paired_soft_comparison" in report


def test_safety_sweep_scenarios_have_distinct_wall_protocols():
    assert _scenario_walls("single_barrier") != _scenario_walls("barrier_shifted")
    assert _scenario_walls("barrier_narrow_gap") == NARROW_GAP_WALLS
    assert _scenario_walls("barrier_wide_gap") == WIDE_GAP_WALLS
    assert _scenario_walls("barrier_shifted") == SHIFTED_BARRIER_WALLS


def test_planner_seed_offsets_are_name_stable():
    assert PLANNER_SEED_OFFSETS["learned_collision"] == 0
    assert PLANNER_SEED_OFFSETS["route_completion_soft"] == 4
    assert len(set(PLANNER_SEED_OFFSETS.values())) == len(PLANNER_SEED_OFFSETS)


def test_subset_tournament_preserves_planner_random_stream():
    from pocketworld.evaluate_planners import evaluate_planner_tournament

    model = PocketWorldModel()
    single = evaluate_planner_tournament(
        model,
        seeds=(5,),
        episodes=1,
        horizon=4,
        candidates=8,
        scenario="open",
        planners=("learned_collision",),
        include_rows=True,
    )
    subset = evaluate_planner_tournament(
        model,
        seeds=(5,),
        episodes=1,
        horizon=4,
        candidates=8,
        scenario="open",
        planners=("random_shooting", "learned_collision"),
        include_rows=True,
    )
    assert single["rows"]["5"]["learned_collision"] == subset["rows"]["5"]["learned_collision"]


def test_gap_cases_start_on_opposite_sides_of_the_wall():
    for scenario in ("barrier_narrow_gap", "barrier_wide_gap"):
        walls = _scenario_walls(scenario)
        for start, goal in __import__("pocketworld.evaluate_planners", fromlist=["_episode_cases"])._episode_cases(8, 11, scenario):
            assert start[0] < 20.0 and goal[0] > 44.0
            assert not any(
                wall.x - 3 <= start[0] <= wall.x + wall.width + 3
                and wall.y - 3 <= start[1] <= wall.y + wall.height + 3
                for wall in walls
            )
            assert not any(
                wall.x - 3 <= goal[0] <= wall.x + wall.width + 3
                and wall.y - 3 <= goal[1] <= wall.y + wall.height + 3
                for wall in walls
            )
