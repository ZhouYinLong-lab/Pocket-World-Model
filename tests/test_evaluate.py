from pocketworld.evaluate import evaluate_planning, evaluate_prediction
from pocketworld.model import PocketWorldModel


def test_evaluation_reports_prediction_horizons():
    report = evaluate_prediction(PocketWorldModel(), episodes=1, seed=5)
    assert set(report) == {"image_mae", "position_error_px"}
    assert set(report["image_mae"]) == {"1", "5", "10", "20"}


def test_evaluation_reports_imagined_and_real_planning():
    report = evaluate_planning(PocketWorldModel(), episodes=1, horizon=3, candidates=4, seed=5)
    assert set(report) == {
        "imagined_success_rate",
        "real_success_rate",
        "planning_gap",
        "mean_imagined_final_distance_px",
        "mean_real_final_distance_px",
    }

