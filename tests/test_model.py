import torch
import numpy as np

from pocketworld.model import PocketWorldModel
from pocketworld.planner import extract_agent_position


def test_model_preserves_image_shape_and_can_imagine():
    model = PocketWorldModel()
    observation = torch.rand(2, 3, 64, 64)
    actions = torch.randint(0, 4, (2, 5))
    next_frame = model.predict_next(observation, actions[:, 0])
    next_frame_with_state, next_position = model.predict_next_state(observation, actions[:, 0])
    imagined = model.imagine(observation, actions)
    imagined_positions = model.imagine_positions(observation, actions)
    collision_probabilities = model.imagine_collision_probabilities(observation, actions)
    collision_response_positions = model.imagine_positions(observation, actions, collision_response=True)
    agent_masks = model.imagine_agent_masks(observation, actions)
    composed = model.compose_agent_rgb(model.decode(model.encode(observation)), model.encode(observation))
    wall_patch = model.wall_patch(observation, model.state_from_latent(model.encode(observation)))
    assert next_frame.shape == observation.shape
    assert next_frame_with_state.shape == observation.shape
    assert next_position.shape == (2, 2)
    assert imagined.shape == (2, 6, 3, 64, 64)
    assert imagined_positions.shape == (2, 5, 2)
    assert collision_probabilities.shape == (2, 5)
    assert collision_response_positions.shape == (2, 5, 2)
    assert agent_masks.shape == (2, 5, 1, 64, 64)
    assert composed.shape == observation.shape
    assert wall_patch.shape == (2, 49)
    assert torch.all((collision_probabilities >= 0) & (collision_probabilities <= 1))


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


def test_agent_position_extractor_does_not_confuse_yellow_goal():
    frame = np.zeros((3, 64, 64), dtype=np.uint8)
    frame[:, 50, 50] = [247, 190, 69]
    frame[:, 10, 10] = [93, 224, 183]
    assert np.allclose(extract_agent_position(frame), [10, 10])


def test_structured_state_dynamics_preserves_action_effect():
    model = PocketWorldModel()
    observation = torch.zeros(1, 3, 64, 64)
    state = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    left = model.state_transition(state, torch.tensor([2]))
    right = model.state_transition(state, torch.tensor([3]))
    assert right[0, 0] > state[0, 0]
    assert left[0, 0] < state[0, 0]
