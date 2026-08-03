from pocketworld.data import collect_random_rollouts


def test_rollout_collection_preserves_temporal_shape():
    batch = collect_random_rollouts(episodes=2, horizon=4, seed=3)
    assert batch.observations.shape == (2, 5, 3, 64, 64)
    assert batch.actions.shape == (2, 4)
    assert batch.positions.shape == (2, 5, 2)
    assert batch.velocities.shape == (2, 5, 2)
