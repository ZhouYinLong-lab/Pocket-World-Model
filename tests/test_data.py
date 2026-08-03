import numpy as np

from pocketworld.data import collect_random_rollouts


def test_rollout_collection_preserves_temporal_shape():
    batch = collect_random_rollouts(episodes=2, horizon=4, seed=3)
    assert batch.observations.shape == (2, 5, 3, 64, 64)
    assert batch.actions.shape == (2, 4)
    assert batch.positions.shape == (2, 5, 2)
    assert batch.velocities.shape == (2, 5, 2)
    assert batch.collisions.shape == (2, 4)


def test_full_state_range_samples_beyond_default_start_region():
    batch = collect_random_rollouts(episodes=32, horizon=2, seed=9, full_state_range=True)
    assert float(batch.positions[:, 0, 0].max()) > 20.0


def test_collision_labels_are_present_for_barrier_rollouts():
    batch = collect_random_rollouts(episodes=8, horizon=16, seed=4, full_state_range=True, barrier_probability=1.0)
    assert batch.collisions.shape == (8, 16)
    assert np.any(batch.collisions > 0)
