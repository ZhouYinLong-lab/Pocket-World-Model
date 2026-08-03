from pocketworld.evaluate import evaluate_action_effects, evaluate_collision_prediction, evaluate_planning, evaluate_prediction
from pocketworld.model import PocketWorldModel


def test_evaluation_reports_prediction_horizons():
    report = evaluate_prediction(PocketWorldModel(), episodes=1, seed=5)
    assert set(report) == {"image_mae", "position_error_px", "latent_position_error_px", "position_coverage"}
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


def test_action_effect_diagnostic_covers_all_actions():
    report = evaluate_action_effects(PocketWorldModel(), repeat=2)
    assert set(report) == {"up", "down", "left", "right"}
    assert all("position_error_px" in value for value in report.values())


def test_collision_prediction_report_has_classification_metrics():
    report = evaluate_collision_prediction(PocketWorldModel(), episodes=1, horizon=2, seed=5)
    assert set(report) == {"accuracy", "positive_rate", "predicted_positive_rate", "precision", "recall"}
