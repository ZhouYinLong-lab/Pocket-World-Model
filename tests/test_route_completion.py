import numpy as np

from pocketworld.route_completion import (
    MAP_AWARE_ROUTE_FEATURE_NAMES,
    ROUTE_FEATURE_NAMES,
    RouteCompletionPredictor,
    extract_map_context_features,
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


def test_map_aware_route_features_append_observable_geometry():
    positions = np.asarray([[[8.0, 32.0], [12.0, 32.0], [16.0, 32.0]]], dtype=np.float32)
    prefix = np.zeros((1, 3), dtype=np.float32)
    wall_mask = np.zeros((64, 64), dtype=bool)
    wall_mask[10:54, 29:34] = True
    context = extract_map_context_features((8.0, 32.0), (56.0, 32.0), wall_mask)
    features = extract_route_features(
        positions,
        (56.0, 32.0),
        prefix,
        map_context=context,
    )

    assert features.shape == (1, len(MAP_AWARE_ROUTE_FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert features[0, len(ROUTE_FEATURE_NAMES) + 5] == 1.0


def test_route_predictor_roundtrips_map_aware_feature_contract(tmp_path):
    predictor = RouteCompletionPredictor(feature_names=MAP_AWARE_ROUTE_FEATURE_NAMES)
    values = np.zeros((4, len(MAP_AWARE_ROUTE_FEATURE_NAMES)), dtype=np.float32)
    values[1, 1] = 1.0
    labels = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
    predictor.fit(values, labels, epochs=1)
    path = tmp_path / "map-aware.pt"
    predictor.save(path)
    loaded = RouteCompletionPredictor.load(path)
    assert loaded.feature_names == MAP_AWARE_ROUTE_FEATURE_NAMES
    assert loaded.predict_proba(values).shape == (4,)


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
