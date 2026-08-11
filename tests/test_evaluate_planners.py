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
