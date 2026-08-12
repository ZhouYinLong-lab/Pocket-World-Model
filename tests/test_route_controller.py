import pytest

from pocketworld.evaluate_route_controller import evaluate_route_controller


def test_route_controller_evaluator_returns_paired_rows_and_summary():
    report = evaluate_route_controller(
        scenarios=("single_barrier", "barrier_narrow_gap"),
        seeds=(11,),
        episodes=1,
        max_steps=32,
        variants=("cardinal_safe", "adaptive_locked_policy"),
    )

    assert report["protocol"]["controller_only"] is True
    assert set(report["variants"]) == {"cardinal_safe", "adaptive_locked_policy"}
    for variant in report["variants"].values():
        assert len(variant["rows"]) == 2
        assert 0.0 <= variant["summary"]["real_success"]["mean"] <= 1.0
        assert variant["summary"]["collision_count"]["mean"] >= 0.0


def test_route_controller_evaluator_rejects_unknown_variant():
    with pytest.raises(ValueError, match="unknown controller variant"):
        evaluate_route_controller(
            scenarios=("single_barrier",),
            seeds=(11,),
            episodes=1,
            max_steps=4,
            variants=("does_not_exist",),
        )
