import onnx
import torch

from pocketworld.export_onnx import OneStepWrapper, export
from pocketworld.model import PocketWorldModel


def test_export_wrapper_returns_frame_and_position():
    wrapper = OneStepWrapper(PocketWorldModel()).eval()
    observation = torch.rand(2, 3, 64, 64)
    action = torch.tensor([0, 3])
    next_observation, next_position = wrapper(observation, action)
    assert next_observation.shape == observation.shape
    assert next_position.shape == (2, 2)


def test_export_writes_loadable_onnx_with_stable_output_contract(tmp_path):
    checkpoint = tmp_path / "model.pt"
    destination = tmp_path / "model.onnx"
    torch.save({"model": PocketWorldModel().state_dict()}, checkpoint)

    export(str(checkpoint), str(destination))
    graph = onnx.load(destination)

    assert destination.stat().st_size > 0
    assert [output.name for output in graph.graph.output] == ["next_observation", "next_position"]
