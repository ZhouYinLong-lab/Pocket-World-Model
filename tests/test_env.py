import numpy as np

from pocketworld.env import PocketWorldEnv, Rect


def test_reset_is_deterministic_and_image_is_chw():
    env = PocketWorldEnv()
    first, info = env.reset(seed=3)
    second, _ = env.reset(seed=3)
    assert first.shape == (3, 64, 64)
    assert first.dtype == np.uint8
    assert np.array_equal(first, second)
    assert info["position"].shape == (2,)


def test_wall_collision_keeps_agent_outside_wall():
    env = PocketWorldEnv(walls=(Rect(20, 0, 5, 64),), agent_start=(12, 32))
    env.reset()
    for _ in range(20):
        _, _, _, _, info = env.step(3)
    assert info["collision"] is True
    assert info["position"][0] < 20 - env.agent_radius


def test_goal_terminates_episode():
    env = PocketWorldEnv(agent_start=(10, 10), goal=(10, 10))
    _, _, terminated, _, _ = env.step(0)
    assert terminated is True

