import pytest

from pocketworld.procedural_routes import procedural_wall_layout, sample_procedural_route_cases


def test_procedural_layout_validates_gap_range():
    with pytest.raises(ValueError, match="gap_height"):
        procedural_wall_layout(7, gap_height=12)
    with pytest.raises(ValueError, match="episodes"):
        sample_procedural_route_cases(7, 0)
    with pytest.raises(ValueError, match="split"):
        sample_procedural_route_cases(7, 1, split="test")


def test_train_and_holdout_have_distinct_protocol_metadata():
    train = sample_procedural_route_cases(7, 2, split="train")
    holdout = sample_procedural_route_cases(7, 2, split="holdout")
    assert all(case.map_id.startswith("train-") for case in train)
    assert all(case.map_id.startswith("holdout-") for case in holdout)
    assert all(case.barrier_count in {2, 3} for case in train)
    assert all(case.barrier_count in {3, 4} for case in holdout)
