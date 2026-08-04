from pocketworld.evaluate import (
    evaluate_action_effects,
    evaluate_collision_prediction,
    evaluate_planning,
    evaluate_prediction,
    evaluate_temporal_velocity,
    evaluate_uncertainty_calibration,
    evaluate_uncertainty_calibration_matrix,
    evaluate_shift_detection_matrix,
)
from pocketworld.model import PocketWorldModel


def test_evaluation_reports_prediction_horizons():
    report = evaluate_prediction(PocketWorldModel(), episodes=1, seed=5)
    assert set(report) == {
        "image_mae",
        "composited_image_mae",
        "position_error_px",
        "composited_position_error_px",
        "latent_position_error_px",
        "position_coverage",
        "composited_position_coverage",
    }
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


def test_temporal_velocity_and_uncertainty_reports_are_calibratable():
    model = PocketWorldModel()
    velocity_report = evaluate_temporal_velocity(model, episodes=1, horizon=2, seed=5)
    uncertainty_report = evaluate_uncertainty_calibration(model, episodes=1, horizon=2, seed=5)

    assert "learned_velocity_mae_px" in velocity_report
    assert "finite_difference_velocity_mae_px" in velocity_report
    assert set(uncertainty_report["position_coverage"]) == {"0.50", "0.80", "0.90", "0.95"}
    assert len(uncertainty_report["calibration_scale"]) == 4


def test_uncertainty_calibration_matrix_covers_speed_and_map_shifts():
    report = evaluate_uncertainty_calibration_matrix(PocketWorldModel(), episodes=1, horizon=2, seed=5)

    assert set(report) == {
        "in_distribution",
        "ood_speed_slow",
        "ood_speed_fast",
        "ood_map",
        "ood_map_fast",
    }
    assert all("position_coverage" in condition for condition in report.values())


def test_shift_detection_matrix_reports_online_alarm_rates():
    report = evaluate_shift_detection_matrix(PocketWorldModel(), episodes=1, horizon=2, seed=5)

    assert "threshold" in report
    assert set(report["conditions"]) == {
        "in_distribution",
        "ood_speed_slow",
        "ood_speed_fast",
        "ood_map",
        "ood_map_fast",
    }
    assert 0.0 <= report["conditions"]["ood_map_fast"]["trigger_rate"] <= 1.0
