import torch

from pocketworld.model import PocketWorldModel
from pocketworld.train import _agent_mask_loss, _agent_mask_targets, train


def test_agent_mask_target_and_loss_are_finite():
    positions = torch.tensor([[0.25, 0.75], [0.75, 0.25]])
    target = _agent_mask_targets(positions)
    loss = _agent_mask_loss(torch.zeros_like(target), target)

    assert target.shape == (2, 1, 64, 64)
    assert torch.all(target.flatten(1).sum(dim=1) > 0)
    assert torch.isfinite(loss)


def test_tiny_training_run_writes_a_loadable_checkpoint(tmp_path):
    destination = train(
        epochs=1,
        episodes=2,
        validation_episodes=1,
        batch_size=2,
        seed=13,
        output=str(tmp_path / "tiny.pt"),
        unroll_horizon=2,
        sticky_probability=0.5,
        full_state_range=True,
    )
    payload = torch.load(destination, map_location="cpu")
    model = PocketWorldModel()
    missing, unexpected = model.load_state_dict(payload["model"], strict=True)

    assert not missing
    assert not unexpected
    assert payload["episodes"] == 2
    assert payload["unroll_horizon"] == 2
    assert payload["collision_supervision"] is True
    assert payload["agent_rendering"] is True
