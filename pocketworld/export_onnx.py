from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .model import PocketWorldModel


class OneStepWrapper(torch.nn.Module):
    def __init__(self, model: PocketWorldModel) -> None:
        super().__init__()
        self.model = model

    def forward(self, observation: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        next_observation, next_position = self.model.predict_next_state(observation, action)
        return next_observation, next_position


def export(checkpoint: str, output: str) -> Path:
    model = PocketWorldModel()
    payload = torch.load(checkpoint, map_location="cpu")
    missing, unexpected = model.load_state_dict(payload["model"], strict=False)
    if missing:
        print(f"warning: checkpoint is missing {len(missing)} optional keys; exporting the compatible image model")
    if unexpected:
        print(f"warning: checkpoint has {len(unexpected)} legacy keys")
    model.eval()
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        OneStepWrapper(model),
        (torch.zeros(1, 3, 64, 64), torch.zeros(1, dtype=torch.long)),
        destination,
        input_names=["observation", "action"],
        output_names=["next_observation", "next_position"],
        dynamic_axes={
            "observation": {0: "batch"},
            "action": {0: "batch"},
            "next_observation": {0: "batch"},
            "next_position": {0: "batch"},
        },
        opset_version=17,
    )
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a trained PocketWorld model for ONNX Runtime Web")
    parser.add_argument("checkpoint", nargs="?", default="artifacts/pocketworld.pt")
    parser.add_argument("--output", default="public/pocketworld.onnx")
    args = parser.parse_args()
    print(export(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
