import torch
import numpy as np

from pocketworld.model import PocketWorldModel
from pocketworld.planner import extract_agent_position


def test_model_preserves_image_shape_and_can_imagine():
    model = PocketWorldModel()
    observation = torch.rand(2, 3, 64, 64)
    actions = torch.randint(0, 4, (2, 5))
    next_frame = model.predict_next(observation, actions[:, 0])
    imagined = model.imagine(observation, actions)
    assert next_frame.shape == observation.shape
    assert imagined.shape == (2, 6, 3, 64, 64)


def test_agent_position_extractor_supports_batches():
    frames = np.zeros((2, 3, 64, 64), dtype=np.uint8)
    frames[0, 1, 10:14, 20:24] = 220
    frames[0, 0, 10:14, 20:24] = 30
    frames[1, 1, 40:44, 45:49] = 220
    frames[1, 0, 40:44, 45:49] = 30
    positions = extract_agent_position(frames)
    assert positions.shape == (2, 2)
    assert np.allclose(positions[0], [21.5, 11.5])
    assert np.allclose(positions[1], [46.5, 41.5])
