import numpy as np

from pocketworld.route_completion import (
    ROUTE_FEATURE_NAMES,
    RouteCompletionPredictor,
    extract_route_features,
)


def test_route_features_follow_the_planner_contract():
    positions = np.asarray(
        [
            [[8.0, 8.0], [12.0, 8.0], [18.0, 8.0]],
            [[8.0, 8.0], [8.0, 7.0], [8.0, 6.0]],
        ],
        dtype=np.float32,
    )
    prefix = np.asarray([[0.0, 0.0, 0.0], [0.0, 0.7, 0.7]], dtype=np.float32)

    features = extract_route_features(positions, (20.0, 8.0), prefix)

    assert features.shape == (2, len(ROUTE_FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert features[0, 1] < features[1, 1]


def test_route_completion_predictor_fits_and_predicts_probabilities():
    features = np.asarray(
        [
            [0.8, 0.05, 0.05, 0.75, 0.0, 1.0, 0.2, 0.1, 0.5],
            [0.8, 0.70, 0.65, 0.10, 0.8, 0.2, 0.8, 0.7, 0.5],
            [0.7, 0.10, 0.08, 0.60, 0.0, 1.0, 0.3, 0.1, 0.5],
            [0.7, 0.75, 0.70, -0.05, 0.9, 0.1, 0.9, 0.8, 0.5],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    predictor = RouteCompletionPredictor()
    metrics = predictor.fit(features, labels, epochs=20, seed=3)
    probabilities = predictor.predict_proba(features)

    assert metrics["samples"] == 4
    assert np.isfinite(probabilities).all()
    assert probabilities.shape == (4,)
    assert probabilities[0] > probabilities[1]

