import numpy as np
import torch

from pocketworld.model import PocketWorldModel
from pocketworld.uncertainty import PocketWorldEnsemble, fit_conformal_upper_bound


def test_conformal_upper_bound_is_conservative_on_calibration_examples():
    calibration = fit_conformal_upper_bound(
        np.asarray([0.05, 0.10, 0.20, 0.25]),
        np.asarray([0.0, 0.0, 1.0, 0.0]),
        alpha=0.10,
    )

    assert calibration.samples == 4
    assert calibration.quantile >= 0.75
    assert calibration.collision_rate == 0.25


def test_ensemble_aggregates_collision_probabilities():
    first = PocketWorldModel()
    second = PocketWorldModel()
    for parameter in second.parameters():
        parameter.data.add_(0.001)
    ensemble = PocketWorldEnsemble([first, second], disagreement_weight=1.0)
    observation = torch.zeros((2, 3, 64, 64))
    actions = torch.zeros((2, 3), dtype=torch.long)

    probabilities = ensemble.imagine_collision_probabilities(observation, actions)

    assert probabilities.shape == (2, 3)
    assert torch.isfinite(probabilities).all()
    assert 0.0 <= ensemble.last_statistics["disagreement_std"] <= 1.0

