import numpy as np

from pocketworld.env import PocketWorldEnv, Rect
from pocketworld.planner import _collision_prefix, extract_wall_mask


def test_wall_mask_detects_wall_but_not_grid_or_agent():
    env = PocketWorldEnv(walls=(Rect(24, 8, 5, 29),), agent_start=(8, 8))
    frame, _ = env.reset()
    mask = extract_wall_mask(frame)
    assert bool(mask[12, 26])
    assert not bool(mask[8, 8])
    assert not bool(mask[0, 0])


def test_collision_prefix_marks_wall_intersection_and_future_steps():
    wall = np.zeros((64, 64), dtype=bool)
    wall[20:40, 30:34] = True
    positions = np.asarray([[[10, 30], [26, 30], [31, 30], [40, 30]]], dtype=np.float32)
    prefix = _collision_prefix(positions, wall)
    assert prefix.tolist() == [[False, False, True, True]]
