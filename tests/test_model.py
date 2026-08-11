import torch
import numpy as np

from pocketworld.env import PocketWorldEnv, Rect
from pocketworld.model import PocketWorldModel, observable_velocity_from_frames
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
    guarded_collision_probabilities = model.imagine_collision_probabilities(observation, actions, visual_collision_guard=True)
    collision_response_positions = model.imagine_positions(observation, actions, collision_response=True)
    guarded_positions = model.imagine_positions(observation, actions, collision_response=True, visual_collision_guard=True)
    agent_masks = model.imagine_agent_masks(observation, actions)
    composed = model.compose_agent_rgb(model.decode(model.encode(observation)), model.encode(observation))
    wall_patch = model.wall_patch(observation, model.state_from_latent(model.encode(observation)))
    assert next_frame.shape == observation.shape
    assert next_frame_with_state.shape == observation.shape
    assert next_position.shape == (2, 2)
    assert imagined.shape == (2, 6, 3, 64, 64)
    assert imagined_positions.shape == (2, 5, 2)
    assert collision_probabilities.shape == (2, 5)
    assert guarded_collision_probabilities.shape == (2, 5)
    assert collision_response_positions.shape == (2, 5, 2)
    assert guarded_positions.shape == (2, 5, 2)
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


def test_observable_velocity_uses_only_rgb_history():
    frames = np.zeros((3, 3, 64, 64), dtype=np.uint8)
    for index, x in enumerate((10, 12, 14)):
        frames[index, 1, 20:24, x:x + 4] = 220
        frames[index, 0, 20:24, x:x + 4] = 30
    velocity = observable_velocity_from_frames(frames)
    assert velocity[0] > 0.0
    assert abs(velocity[1]) < 0.1
    assert np.linalg.norm(velocity) <= 2.3


def test_structured_state_dynamics_preserves_action_effect():
    model = PocketWorldModel()
    observation = torch.zeros(1, 3, 64, 64)
    state = torch.tensor([[0.5, 0.5, 0.0, 0.0]])
    left = model.state_transition(state, torch.tensor([2]))
    right = model.state_transition(state, torch.tensor([3]))
    assert right[0, 0] > state[0, 0]
    assert left[0, 0] < state[0, 0]


def test_temporal_velocity_representation_returns_mean_and_scale():
    model = PocketWorldModel()
    observations = torch.rand(2, 4, 3, 64, 64)
    velocity, scale = model.temporal_velocity_stats(observations)
    history_state = model.state_from_history(observations)

    assert velocity.shape == scale.shape == (2, 2)
    assert history_state.shape == (2, 4)
    assert torch.all(torch.isfinite(velocity))
    assert torch.all(scale > 0)


def test_calibrated_transition_uncertainty_is_positive_and_probabilistic():
    model = PocketWorldModel()
    observation = torch.rand(2, 3, 64, 64)
    action = torch.tensor([0, 3])
    latent = model.encode(observation)
    state = model.state_from_latent(latent)
    mean, scale = model.transition_state_stats(latent, state, action)
    probabilities = model.probabilistic_collision_probability(
        latent, state, action, observation, mean, scale, samples=8
    )

    assert mean.shape == scale.shape == (2, 4)
    assert probabilities.shape == (2,)
    assert torch.all(scale > 0)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))


def test_wall_patch_is_high_on_a_wall_and_low_in_free_space():
    env = PocketWorldEnv(walls=(Rect(24, 8, 5, 29),), agent_start=(8, 8))
    observation, _ = env.reset()
    frame = torch.from_numpy(observation[None]).float() / 255.0
    model = PocketWorldModel()
    wall_state = torch.tensor([[26 / 64, 20 / 64, 0.0, 0.0]])
    free_state = torch.tensor([[10 / 64, 20 / 64, 0.0, 0.0]])

    assert model.wall_patch(frame, wall_state).max() > 0.9
    assert model.wall_patch(frame, free_state).max() < 0.1


def test_wall_patch_treats_world_boundary_as_collision_geometry():
    observation = torch.zeros(1, 3, 64, 64)
    model = PocketWorldModel()
    edge_state = torch.tensor([[2 / 64, 20 / 64, 0.0, 0.0]])
    center_state = torch.tensor([[32 / 64, 32 / 64, 0.0, 0.0]])

    assert model.wall_patch(observation, edge_state).max() > 0.9
    assert model.wall_patch(observation, center_state).max() < 0.1


def test_predicted_collision_response_freezes_the_compact_state():
    model = PocketWorldModel()
    with torch.no_grad():
        model.collision_head[-1].weight.zero_()
        model.collision_head[-1].bias.fill_(20.0)
        model.spatial_collision_head[-1].weight.zero_()
        model.spatial_collision_head[-1].bias.fill_(20.0)
    observation = torch.zeros(1, 3, 64, 64)
    actions = torch.tensor([[3, 3]])
    initial_position = model.state_from_latent(model.encode(observation))[0, :2]
    imagined = model.imagine_positions(observation, actions, collision_response=True)

    assert torch.allclose(imagined[0, 0], initial_position)
    assert torch.allclose(imagined[0, 1], initial_position)


def test_collision_probability_rollout_stops_after_predicted_impact():
    model = PocketWorldModel()
    with torch.no_grad():
        model.spatial_collision_head[-1].weight.zero_()
        model.spatial_collision_head[-1].bias.fill_(20.0)
    observation = torch.zeros(1, 3, 64, 64)
    actions = torch.tensor([[3, 3, 3]])
    stopped = model.imagine_collision_probabilities(observation, actions)
    passthrough = model.imagine_collision_probabilities(observation, actions, collision_response=False)

    assert stopped.shape == passthrough.shape == (1, 3)


def test_robust_collision_probability_dominates_point_estimate():
    env = PocketWorldEnv(walls=(Rect(29, 10, 5, 44),), agent_start=(24, 30))
    observation, _ = env.reset()
    frame = torch.from_numpy(observation[None]).float() / 255.0
    model = PocketWorldModel()
    latent = model.encode(frame)
    state = torch.tensor([[24 / 64, 30 / 64, 0.0, 0.0]])
    action = torch.tensor([3])
    point = model.robust_collision_probability(latent, state, action, frame)
    robust = model.robust_collision_probability(latent, state, action, frame, uncertainty_radius_px=1.5)

    assert robust.shape == point.shape == (1,)
    assert torch.all(robust >= point)
