import numpy as np

from pocketworld.data import collect_random_rollouts
from pocketworld.env import PocketWorldEnv
from pocketworld.evaluate_generalization import evaluate_generalization, evaluate_generalization_seeds
from pocketworld.maps import MAPS, map_names
from pocketworld.model import PocketWorldModel
from pocketworld.tasks import sample_navigation_task, sample_task_suite


def test_named_map_suites_include_disjoint_unseen_layouts():
    assert set(map_names("train")).isdisjoint(map_names("holdout"))
    assert set(map_names("all")) == set(MAPS)
    assert len(MAPS["cross"].walls) >= 2
    assert len(MAPS["zigzag"].walls) >= 2


def test_rollout_collection_supports_named_training_suite():
    batch = collect_random_rollouts(episodes=3, horizon=4, seed=21, map_suite="train")
    assert batch.observations.shape == (3, 5, 3, 64, 64)
    assert batch.positions.shape == (3, 5, 2)


def test_waypoint_tasks_have_ordered_goals_on_requested_map():
    task = sample_navigation_task(np.random.default_rng(9), "zigzag", waypoint_count=3)
    assert task.map_name == "zigzag"
    assert len(task.goals) == 3
    assert all(np.linalg.norm(np.asarray(goal) - np.asarray(task.start)) > 0 for goal in task.goals)


def test_environment_can_advance_sequential_goals():
    env = PocketWorldEnv(walls=(), agent_start=(8, 8), goal=(8, 8))
    env.reset()
    env.set_goal((20, 20))
    assert np.allclose(env.goal, (20, 20))


def test_generalization_report_separates_unseen_maps_and_waypoint_tasks():
    report = evaluate_generalization(PocketWorldModel(), episodes=1, horizon=5, candidates=2, seed=4)
    assert [item["map_name"] for item in report["prediction"]["unseen"]] == ["cross", "zigzag"]
    assert set(report["prediction_summary"]) == {"train", "unseen"}
    assert report["waypoint_tasks"]["unseen"][0]["waypoint_count"] == 2


def test_generalization_preserves_requested_waypoint_execution_budget():
    report = evaluate_generalization(PocketWorldModel(), episodes=1, horizon=48, candidates=2, seed=4)
    assert report["waypoint_tasks"]["train"][0]["mean_actions_per_leg"] <= 48


def test_generalization_seed_report_contains_mean_and_std():
    report = evaluate_generalization_seeds(PocketWorldModel(), seeds=(4, 5), episodes=1, horizon=5, candidates=2)
    assert report["config"]["seeds"] == [4, 5]
    assert "mean" in report["summary"]["waypoint_task_success_rate"]["unseen"]
    assert "std" in report["summary"]["prediction_20_step_position_error_px"]["train"]
