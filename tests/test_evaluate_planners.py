import numpy as np

from pocketworld.evaluate_planners import evaluate_planner_tournament
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
