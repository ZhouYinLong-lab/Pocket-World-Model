import torch

from pocketworld.export_onnx import OneStepWrapper
from pocketworld.model import PocketWorldModel


def test_export_wrapper_returns_frame_and_position():
    wrapper = OneStepWrapper(PocketWorldModel()).eval()
    observation = torch.rand(2, 3, 64, 64)
    action = torch.tensor([0, 3])
    next_observation, next_position = wrapper(observation, action)
    assert next_observation.shape == observation.shape
    assert next_position.shape == (2, 2)
