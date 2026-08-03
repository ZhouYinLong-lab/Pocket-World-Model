import numpy as np

from pocketworld.env import PocketWorldEnv, Rect
from pocketworld.planner import extract_wall_mask


def test_wall_mask_detects_wall_but_not_grid_or_agent():
    env = PocketWorldEnv(walls=(Rect(24, 8, 5, 29),), agent_start=(8, 8))
    frame, _ = env.reset()
    mask = extract_wall_mask(frame)
    assert bool(mask[12, 26])
    assert not bool(mask[8, 8])
    assert not bool(mask[0, 0])
