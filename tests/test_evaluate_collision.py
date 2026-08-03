from pocketworld.evaluate_collision import _episode_cases, evaluate_seed
from pocketworld.model import PocketWorldModel


def test_collision_evaluation_cases_are_seeded_and_in_expected_regions():
    first = _episode_cases(3, seed=71)
    second = _episode_cases(3, seed=71)

    assert first == second
    assert all(7 <= start[0] < 13 for start, _ in first)
    assert all(51 <= goal[0] < 57 for _, goal in first)


def test_collision_evaluation_compares_all_planner_variants():
    report = evaluate_seed(PocketWorldModel(), episodes=1, horizon=2, candidates=4, seed=5)

    assert set(report) == {
        "point_open",
        "uncertainty_open",
        "history_closed",
        "history_uncertainty_closed",
    }
    assert all("real_success" in metrics for metrics in report.values())
    assert all("collision_count" in metrics for metrics in report.values())
